import asyncio
import hashlib
import io
import os
import re
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

# Tesseract uses OpenMP threads; one per worker process avoids oversubscription
# when several OCR jobs run in parallel. Must be set before OCR runs.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import httpx
import pymupdf
import pytesseract
from PIL import Image, ImageOps
from docx import Document
from fastapi import FastAPI, Form, HTTPException

VERSION = "4.0.0"

# --------------------------------------------------
# Configuration
# --------------------------------------------------

CPU = os.cpu_count() or 4

MAX_FILE_SIZE = int(os.getenv("ATS_MAX_FILE_SIZE", str(10 * 1024 * 1024)))
MAX_FILES = int(os.getenv("ATS_MAX_FILES", "1000"))
MAX_TEXT_CHARS = int(os.getenv("ATS_MAX_TEXT_CHARS", "20000"))
OCR_DPI = int(os.getenv("ATS_OCR_DPI", "130"))
OCR_MAX_PAGES = int(os.getenv("ATS_OCR_MAX_PAGES", "2"))
MIN_TEXT_BEFORE_OCR = int(os.getenv("ATS_MIN_TEXT_BEFORE_OCR", "40"))

# CPU-bound extraction (PDF parsing / OCR) runs in separate processes.
EXTRACT_WORKERS = max(1, int(os.getenv("ATS_EXTRACT_WORKERS", str(max(1, CPU - 1)))))
# CVs in flight (download -> extract -> score) at once. Bounds RAM.
PIPELINE_CONCURRENCY = max(1, int(os.getenv("ATS_PIPELINE_CONCURRENCY", str(EXTRACT_WORKERS * 2))))

CACHE_MAX_ITEMS = max(0, int(os.getenv("ATS_CACHE_MAX_ITEMS", "0")))      # 0 = cache off
CACHE_TTL_SECONDS = max(0, int(os.getenv("ATS_CACHE_TTL_SECONDS", "0")))
DOWNLOAD_TIMEOUT = float(os.getenv("ATS_DOWNLOAD_TIMEOUT", "30"))
ALLOW_FILE_URLS = os.getenv("ATS_ALLOW_FILE_URLS", "0").lower() in {"1", "true", "yes"}
MAX_JOBS = 50

SUPPORTED = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\s\-().]{8,16}\d)(?!\d)")

TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")
if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

pipeline_semaphore = asyncio.Semaphore(PIPELINE_CONCURRENCY)

_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_cache_lock = asyncio.Lock()
_jobs: OrderedDict[str, dict] = OrderedDict()

http_client: httpx.AsyncClient | None = None
process_pool: ProcessPoolExecutor | None = None


# --------------------------------------------------
# App lifecycle
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, process_pool
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(DOWNLOAD_TIMEOUT, connect=8.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        follow_redirects=True,
        headers={"User-Agent": f"ATS-Screening-API/{VERSION}"},
    )
    process_pool = ProcessPoolExecutor(max_workers=EXTRACT_WORKERS)
    try:
        yield
    finally:
        await http_client.aclose()
        http_client = None
        process_pool.shutdown(wait=False, cancel_futures=True)
        process_pool = None


app = FastAPI(
    title="ATS Screening API",
    description="Reads CV files, extracts text, and ranks candidates with an explainable (XAI) rule-based score",
    version=VERSION,
    lifespan=lifespan,
)


# --------------------------------------------------
# Cache helpers
# --------------------------------------------------

async def cache_get(key: str) -> dict | None:
    if CACHE_MAX_ITEMS <= 0 or CACHE_TTL_SECONDS <= 0:
        return None
    now = time.monotonic()
    async with _cache_lock:
        item = _cache.get(key)
        if item is None:
            return None
        expires_at, data = item
        if expires_at <= now:
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return dict(data)


async def cache_put(key: str, data: dict) -> None:
    if CACHE_MAX_ITEMS <= 0 or CACHE_TTL_SECONDS <= 0:
        return
    async with _cache_lock:
        _cache[key] = (time.monotonic() + CACHE_TTL_SECONDS, dict(data))
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX_ITEMS:
            _cache.popitem(last=False)


# --------------------------------------------------
# Download file
# --------------------------------------------------

async def download_file(client: httpx.AsyncClient, url: str) -> tuple[str, bytes]:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme == "file":
        if not ALLOW_FILE_URLS:
            raise ValueError("file:// URLs are disabled. Provide an HTTP(S) document URL.")
        file_path = unquote(parsed.path)
        if os.name == "nt" and file_path.startswith("/"):
            file_path = file_path[1:]
        path = Path(file_path).resolve()
        if not path.is_file():
            raise ValueError("The local document does not exist.")
        if path.stat().st_size > MAX_FILE_SIZE:
            raise ValueError("File exceeds the 10 MB limit.")
        return path.name or "local_file", await asyncio.to_thread(path.read_bytes)

    if scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Document URL must use HTTP or HTTPS.")

    filename = Path(unquote(parsed.path)).name or "downloaded_file"
    chunks: list[bytes] = []
    size = 0
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        content_length = response.headers.get("content-length", "")
        if content_length.isdigit() and int(content_length) > MAX_FILE_SIZE:
            raise ValueError("File exceeds the 10 MB limit.")
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                raise ValueError("File exceeds the 10 MB limit.")
            chunks.append(chunk)
    return filename, b"".join(chunks)


def detect_extension(filename: str, data: bytes) -> str:
    """Trust the file's content first, then its name (URLs often have no real extension)."""
    ext = Path(filename).suffix.lower()
    if data[:5] == b"%PDF-":
        return ".pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"PK\x03\x04" and ext in {"", ".docx"}:
        return ".docx"
    return ext


# --------------------------------------------------
# Text extraction (runs inside worker processes)
# --------------------------------------------------

def extract_cv_text(filename: str, file_bytes: bytes) -> str:
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        pages: list[str] = []
        total_chars = 0
        ocr_pages = 0
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            for page in pdf:
                page_text = page.get_text().strip()

                # OCR is slow, so only OCR a couple of near-empty (scanned) pages.
                if len(page_text) < MIN_TEXT_BEFORE_OCR and ocr_pages < OCR_MAX_PAGES:
                    ocr_pages += 1
                    try:
                        text_page = page.get_textpage_ocr(language="eng", dpi=OCR_DPI, full=True)
                        page_text = page.get_text(textpage=text_page).strip()
                    except Exception:
                        pass

                if page_text:
                    pages.append(page_text)
                    total_chars += len(page_text)
                if total_chars >= MAX_TEXT_CHARS:
                    break
        text = "\n".join(pages)

    elif extension == ".docx":
        document = Document(io.BytesIO(file_bytes))
        lines = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    lines.append(row_text)
        text = "\n".join(lines)

    elif extension in {".png", ".jpg", ".jpeg"}:
        with Image.open(io.BytesIO(file_bytes)) as image:
            image = ImageOps.exif_transpose(image).convert("L")
            image.thumbnail((2200, 2200))
            text = pytesseract.image_to_string(image, lang="eng")

    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, PNG, JPG or JPEG.")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    if not text:
        raise ValueError("No readable text was extracted from this CV.")
    return text[:MAX_TEXT_CHARS]


def regex_email(text: str) -> str:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else ""


def regex_phone(text: str) -> str:
    for match in PHONE_RE.finditer(text):
        candidate = match.group(1).strip()
        if 10 <= len(re.sub(r"\D", "", candidate)) <= 14:
            return candidate
    return ""


# --------------------------------------------------
# Rule-based parsing helpers (all matching ignores UPPER/lower case)
# --------------------------------------------------

EXP_NAMES = {
    "experience", "work experience", "professional experience", "employment",
    "employment history", "work history", "career history",
}
EDU_NAMES = {
    "education", "educational background", "education background",
    "academic background", "academic qualifications", "academic qualification",
    "qualifications", "educational qualifications",
}
OTHER_NAMES = {
    "skills", "technical skills", "projects", "project", "certifications",
    "certification", "summary", "profile", "objective", "languages",
    "references", "interests", "achievements", "awards", "training",
    "courses", "personal details", "personal information", "publications", "hobbies",
}


def split_sections(text: str) -> dict[str, str]:
    """Pull the experience and education sections out of the CV text."""
    sections: dict[str, list[str]] = {"experience": [], "education": []}
    current = None
    for line in text.split("\n"):
        key = re.sub(r"[^a-z ]", "", line.casefold()).strip()  # casefold -> "EXPERIENCE" == "experience"
        if len(line) <= 40:
            if key in EXP_NAMES:
                current = "experience"
                continue
            if key in EDU_NAMES:
                current = "education"
                continue
            if key in OTHER_NAMES:
                current = None
                continue
        if current:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


# "MS"/"BS" match in any case, but not "MS Office", "MS SQL" or "200 ms".
_NOT_MS_TOOL = (
    r"(?!\s*(?:office|word|excel|sql|access|project|teams|power|dynamics|"
    r"azure|visio|outlook|windows|dos)(?!\w))"
)
DEGREE_PATTERNS = [
    (5, re.compile(r"(?<!\w)(?:ph\.?\s?d|doctorate|doctoral)(?!\w)", re.I)),
    (4, re.compile(
        r"(?<!\w)(?:master'?s|masters|master of [a-z]+|postgraduate|mba|m\.?phil|"
        r"msc|m\.s\.|mscs|mcs)(?!\w)", re.I)),
    (4, re.compile(r"(?<!\w)(?<!\d )ms" + _NOT_MS_TOOL + r"(?!\w)", re.I)),
    (3, re.compile(
        r"(?<!\w)(?:bachelor'?s?|undergraduate|b\.?sc|bscs|bsit|bsse|bcs|bca|bba|"
        r"beng|b\.s\.|b\.e\.|b\.?tech|bs)(?!\w)", re.I)),
    (2, re.compile(r"(?<!\w)associate'?s? degree(?!\w)", re.I)),
    (1, re.compile(r"(?<!\w)(?:high school|secondary school|matric|hssc)(?!\w)", re.I)),
]
LEVEL_NAMES = {5: "Doctorate", 4: "Master's", 3: "Bachelor's", 2: "Associate", 1: "High school"}


def degree_levels(text: str) -> set[int]:
    return {level for level, pattern in DEGREE_PATTERNS if pattern.search(text)}


def best_degree(text: str) -> tuple[int, str]:
    best_level, best_line = 0, ""
    for line in text.split("\n"):
        levels = degree_levels(line)
        if levels and max(levels) > best_level:
            best_level, best_line = max(levels), line.strip()[:150]
    return best_level, best_line


# ---- Candidate name ---------------------------------------------------------

_BULLETS = "•-*▪●◦·➢►"
NAME_STOP = re.compile(
    r"resume|curriculum|vitae|\bcv\b|profile|objective|summary|contact|address|email|phone|page|"
    r"university|college|school|institute|science|technology|information|computer|engineering|"
    r"pakistan|karachi|hyderabad|lahore|islamabad|sindh|punjab|street|road|city|skills|education|experience",
    re.I,
)
TITLE_WORDS = (
    r"developer|engineer|designer|manager|analyst|intern|internship|consultant|administrator|officer|"
    r"assistant|specialist|architect|programmer|lead|full[- ]?stack|front[- ]?end|back[- ]?end|software|"
    r"student|fresher|trainee|instructor|teacher|trainer|executive|coordinator|director|head|supervisor|"
    r"technician|associate|freelance|freelancer|tester|qa|devops"
)
TITLE_RE = re.compile(rf"\b(?:{TITLE_WORDS})\b", re.I)
NOT_NAME_WORDS = re.compile(
    r"\b(?:and|the|with|for|from|of|in|at|on|to|an|is|am|my|built|developed|worked|using|seeking|"
    r"looking|motivated|experienced|passionate|hardworking|dedicated|oriented|driven|years|bachelor|"
    r"master|degree|certified|certificate|course|project|projects|mobile|web|app|apps|application|"
    r"applications|database|laravel|flutter|php|python|java|javascript|sql|mysql|api|apis|html|css|"
    r"react|node|asp|net|dart|present)\b",
    re.I,
)
_NAME_WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|[.'\-])*")  # letters (any language), dots, apostrophes, hyphens


def _valid_name(seg: str) -> bool:
    words = seg.split()
    return (
        2 <= len(words) <= 4
        and len(seg) <= 40
        and all(_NAME_WORD.fullmatch(w) for w in words)
        and not NAME_STOP.search(seg)
        and not TITLE_RE.search(seg)
        and not NOT_NAME_WORDS.search(seg)
    )


def _normalize_case(name: str) -> str:
    """'SADAM QURESHI' and 'sadam qureshi' both become 'Sadam Qureshi'; mixed case is kept."""
    name = re.sub(r"\s+", " ", name).strip()
    return name.title() if name.isupper() or name.islower() else name


def guess_name(text: str, email: str = "") -> tuple[str, str]:
    """Returns (name, source) where source is 'label', 'text' or 'email'."""
    m = re.search(r"(?im)^\s*(?:full\s+|candidate\s+)?name\s*[:\-]\s*(.+)$", text[:2500])
    if m and _valid_name(m.group(1).strip()):
        return _normalize_case(m.group(1)), "label"

    all_headers = EXP_NAMES | EDU_NAMES | OTHER_NAMES
    cands: list[tuple[int, str]] = []
    for i, line in enumerate(text.split("\n")[:15]):
        line = line.strip()
        if not line or line[:1] in _BULLETS:
            continue
        if re.sub(r"[^a-z ]", "", line.casefold()).strip() in all_headers:
            break  # the name comes before the first section
        line = EMAIL_RE.sub(" ", line)
        line = re.sub(r"https?://\S+|www\.\S+|linkedin\.com\S*|github\.com\S*", " ", line, flags=re.I)
        line = PHONE_RE.sub(" ", line)
        for seg in re.split(r"\s[|•·–—\-]\s|[|•·,]", line):
            seg = seg.strip(" :")
            if not seg:
                continue
            cands.append((i, seg))
            title = TITLE_RE.search(seg)
            if title and title.start() > 0:  # "Sadam Qureshi Full Stack Developer"
                cands.append((i, seg[: title.start()].strip()))

    for _, seg in cands:
        if _valid_name(seg):
            return _normalize_case(seg), "text"
    # first name and last name on two consecutive lines
    for (i1, a), (i2, b) in zip(cands, cands[1:]):
        if i2 == i1 + 1 and len(a.split()) == 1 and len(b.split()) == 1 and _valid_name(f"{a} {b}"):
            return _normalize_case(f"{a} {b}"), "text"

    # Last resort: first.last@example.com -> "First Last"
    if email:
        parts = [p for p in re.split(r"[._\-\d]+", email.split("@")[0]) if len(p) >= 2]
        if len(parts) >= 2:
            return " ".join(p.capitalize() for p in parts[:3]), "email"
    return "", ""


# ---- Work experience parsing ------------------------------------------------

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MONTH}\s*,?\s*\d{{4}}|\d{{1,2}}[/.\-]\d{{4}}|\d{{4}})"
RANGE_RE = re.compile(
    rf"({_DATE})\s*(?:-|–|—|to|till|until)\s*({_DATE}|present|current|now|till date|to date|ongoing)",
    re.I,
)
_ONGOING = re.compile(r"present|current|now|till date|to date|ongoing", re.I)


def _is_bullet(line: str) -> bool:
    return line[:1] in _BULLETS or len(line) > 90


def _segments(line: str) -> list[str]:
    parts = re.split(r"\s+(?:at|@)\s+|\s[|–—\-]\s|\s*\|\s*|,\s*", line)
    return [p.strip(" :•-") for p in parts if len(re.findall(r"[A-Za-z]", p)) >= 2]


def _year(value: str) -> int:
    m = re.search(r"\d{4}", value)
    return int(m.group(0)) if m else 0


def parse_experiences(exp_text: str, require_title: bool = False) -> list[dict]:
    """Rule-based extraction of job entries, newest first."""
    lines = [l.strip() for l in exp_text.split("\n") if l.strip()]
    hits = [(i, RANGE_RE.search(l)) for i, l in enumerate(lines)]
    hits = [(i, m) for i, m in hits if m]
    if not hits:
        return []

    def inline_text(i, m):
        return (lines[i][:m.start()] + " " + lines[i][m.end():]).strip(" |-–—,()")

    def header_start(i, m, floor):
        """First line of this entry's header: inline on the date line, or up to 2 non-bullet lines above."""
        if len(re.findall(r"[A-Za-z]", inline_text(i, m))) >= 4:
            return i
        j = i
        while j > floor and i - j < 2 and not _is_bullet(lines[j - 1]):
            j -= 1
        return j

    entries = []
    for k, (i, m) in enumerate(hits):
        prev_end = hits[k - 1][0] + 1 if k else 0
        h_start = header_start(i, m, prev_end)
        next_start = header_start(*hits[k + 1], i + 1) if k + 1 < len(hits) else len(lines)

        header = [inline_text(i, m)] if inline_text(i, m) else []
        header += [l for l in lines[h_start:i] if not _is_bullet(l)]
        body_start = i + 1
        # Title/Date/Company layouts: a short non-bullet line right after the date can still be header.
        if (body_start < next_start and h_start < i and len(header) < 2
                and not _is_bullet(lines[body_start]) and len(lines[body_start]) < 60):
            header.append(lines[body_start])
            body_start += 1

        segs = [s for h in header for s in _segments(h)]
        title = next((s for s in segs if TITLE_RE.search(s)), "")
        if require_title and not title:
            continue
        rest = [s for s in segs if s != title]
        if not title and segs:
            title, rest = segs[0], segs[1:]
        company = next((s for s in rest if not TITLE_RE.search(s)), rest[0] if rest else "")

        body = [l.lstrip(_BULLETS + " ") for l in lines[body_start:next_start]]
        summary = re.sub(r"\s+", " ", " ".join(body)).strip()[:250]

        end_raw = m.group(2)
        entries.append({
            "job_title": title[:100], "company": company[:100],
            "duration": f"{m.group(1).strip()} - {end_raw.strip()}", "summary": summary,
            "_key": (9999 if _ONGOING.search(end_raw) else _year(end_raw), _year(m.group(1))),
        })

    entries.sort(key=lambda e: e["_key"], reverse=True)
    for e in entries:
        e.pop("_key")
    return entries


# --------------------------------------------------
# Scoring with built-in explainability (XAI)
# --------------------------------------------------

SKILL_ALIASES = {
    "rest apis": ["rest api", "restful api", "restful apis", "rest services",
                  "restful services", "restful web services"],
    "fastapi": ["fastapi", "fast api"],
    "python": ["python 2", "python 3", "python programming", "python development"],
    "sql": ["structured query language"],
}

STOP_WORDS = {
    "the", "and", "for", "with", "from", "this", "that", "are", "you", "your",
    "will", "using", "have", "has", "into", "our", "their", "they", "can", "job",
    "role", "developer", "development", "experience", "work", "working", "years",
    "year", "required", "skills", "ability", "responsible", "responsibilities",
    "must", "should", "including", "include", "strong", "good", "knowledge",
    "proficient", "proficiency", "candidate", "team", "teams",
}
TERM_ALIASES = {
    "apis": "api", "restful": "api", "databases": "database",
    "integrations": "integrate", "integrated": "integrate", "integrating": "integrate",
    "integration": "integrate", "services": "service", "applications": "application",
    "apps": "application", "developed": "develop", "developing": "develop",
    "maintained": "maintain", "maintaining": "maintain", "back-end": "backend",
    "built": "build", "builds": "build", "building": "build",
    "designed": "design", "designing": "design", "designs": "design",
    "implemented": "implement", "implementing": "implement", "implements": "implement",
    "managed": "manage", "managing": "manage", "manages": "manage",
    "tested": "test", "testing": "test", "tests": "test",
}
EDU_WORDS = {
    "bachelor", "master", "doctorate", "doctoral", "degree", "undergraduate",
    "postgraduate", "education", "diploma", "certificate", "required", "qualification",
}
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z+#.]{2,}")


def terms(value) -> set[str]:
    out = set()
    for token in _TOKEN_RE.findall(str(value).casefold()):  # casefold -> UPPER and lower are the same term
        token = token.strip(".")
        if token and token not in STOP_WORDS:
            out.add(TERM_ALIASES.get(token, token))
    return out


@dataclass
class JobSpec:
    """Everything derived from the job posting, computed once per request."""
    skills: list[tuple[str, re.Pattern]]
    title_terms: set[str]
    desc_terms: set[str]
    resp_terms: set[str]
    required_edu_level: int | None


def build_job_spec(job_title: str, job_description: str, required_skills: str) -> JobSpec:
    skills = []
    for name in (s.strip() for s in re.split(r"[,;\n]", required_skills)):
        if not name:
            continue
        variants = [name] + SKILL_ALIASES.get(name.casefold(), [])
        pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(v) for v in variants) + r")(?!\w)", re.I
        )
        skills.append((name, pattern))

    title_terms = terms(job_title)
    desc_terms = terms(job_description) - EDU_WORDS  # degree wording is scored under education
    levels = degree_levels(job_description)
    return JobSpec(
        skills=skills,
        title_terms=title_terms,
        desc_terms=desc_terms,
        resp_terms=(desc_terms - title_terms) - EDU_WORDS,
        required_edu_level=min(levels) if levels else None,
    )


def _snippet(text: str, m: re.Match, pad: int = 45) -> str:
    start, end = max(0, m.start() - pad), min(len(text), m.end() + pad)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def score_cv(spec: JobSpec, cv_text: str, explain: bool = True) -> dict:
    """
    Preliminary 100-point score (40 skills / 30 experience / 20 education / 10 responsibilities).

    Explainability: the score is a sum of fixed-weight contributions, so every contribution below is
    exact (for an additive model this equals the SHAP value). Nothing is approximated or sampled.
    """
    notes: list[str] = []
    sections = split_sections(cv_text)
    has_exp_section = bool(sections["experience"])
    exp_text = sections["experience"] or cv_text
    if not has_exp_section:
        notes.append("No 'Experience' heading was found, so the whole CV text was used for experience matching.")
    exp_terms = terms(exp_text)

    # ---- Skills (40) ----
    n_skills = len(spec.skills)
    per_skill = 40 / n_skills if n_skills else 0.0
    skill_items, matched, missing = [], [], []
    for name, pattern in spec.skills:
        m = pattern.search(cv_text)
        if m:
            matched.append(name)
            skill_items.append({"skill": name, "found": True, "matched_text": m.group(0),
                                "evidence": _snippet(cv_text, m), "points": round(per_skill, 2)})
        else:
            missing.append(name)
            skill_items.append({"skill": name, "found": False, "matched_text": None,
                                "evidence": None, "points": 0.0})
    skill_points = round(40 * len(matched) / n_skills) if n_skills else None
    if not n_skills:
        notes.append("No required skills were given, so skills (40 pts) were not scored.")

    # ---- Experience (30) ----
    title_hit, title_miss = sorted(spec.title_terms & exp_terms), sorted(spec.title_terms - exp_terms)
    desc_hit, desc_miss = sorted(spec.desc_terms & exp_terms), sorted(spec.desc_terms - exp_terms)
    title_rel = len(title_hit) / len(spec.title_terms) if spec.title_terms else None
    desc_rel = len(desc_hit) / len(spec.desc_terms) if spec.desc_terms and exp_terms else None
    if title_rel is not None and desc_rel is not None:
        w_title, w_desc = 0.35, 0.65
    elif desc_rel is not None:
        w_title, w_desc = 0.0, 1.0
    elif title_rel is not None:
        w_title, w_desc = 1.0, 0.0
    else:
        w_title = w_desc = 0.0
    experience_points = (
        round(30 * (w_title * (title_rel or 0) + w_desc * (desc_rel or 0)))
        if (w_title or w_desc) else None
    )
    pt_title = 30 * w_title / len(spec.title_terms) if spec.title_terms else 0.0
    pt_desc = 30 * w_desc / len(spec.desc_terms) if spec.desc_terms else 0.0

    # ---- Education (20) ----
    cand_level, edu_line = best_degree(sections["education"])
    if not cand_level:
        cand_level, edu_line = best_degree(cv_text)
    required = spec.required_edu_level
    education_points = None
    if required is None:
        notes.append("The job description states no minimum degree, so education (20 pts) was not scored.")
    elif not cand_level:
        notes.append("No recognizable degree was found in the CV, so education (20 pts) was not scored.")
    else:
        education_points = 20 if cand_level >= required else 0

    # ---- Responsibilities (10) ----
    resp_hit, resp_miss = sorted(spec.resp_terms & exp_terms), sorted(spec.resp_terms - exp_terms)
    responsibility_points = (
        round(10 * len(resp_hit) / len(spec.resp_terms)) if spec.resp_terms and exp_terms else None
    )
    pt_resp = 10 / len(spec.resp_terms) if spec.resp_terms else 0.0

    # ---- Totals and recommendation ----
    categories = [
        (skill_points, 40), (experience_points, 30),
        (education_points, 20), (responsibility_points, 10),
    ]
    available_weight = sum(w for p, w in categories if p is not None)
    score = max(0, min(100, round(sum(p for p, _ in categories if p is not None))))

    if available_weight < 70:
        recommendation = "Needs Review"
    elif score >= 75 and available_weight >= 90 and not missing:
        recommendation = "Strong Match"
    elif score >= 45:
        recommendation = "Potential Match"
    else:
        recommendation = "Needs Review"

    # ---- Plain-language explanation ----
    def fmt(v):
        return v if v is not None else "unverified"

    explanation = (
        f"Preliminary screening signal: {score}/100 ({recommendation}). "
        f"Skills {fmt(skill_points)}/40, experience relevance {fmt(experience_points)}/30, "
        f"education {fmt(education_points)}/20, responsibilities {fmt(responsibility_points)}/10. "
        f"Assessable rubric weight: {available_weight}/100. "
        f"Required skills without explicit text evidence: {', '.join(missing) if missing else 'none'}. "
        "Keyword overlap and degree-name matching are approximate; this score is "
        "not validated and must not be used as the sole basis for hiring."
    )

    email = regex_email(cv_text)
    name, name_source = guess_name(cv_text, email)
    entries = (parse_experiences(sections["experience"]) if has_exp_section
               else parse_experiences(cv_text, require_title=True))
    last = entries[0] if entries else {"job_title": "", "company": "", "duration": "", "summary": ""}

    data = {
        "candidate_name": name,
        "name_source": name_source,
        "email": email,
        "phone": regex_phone(cv_text),
        "highest_education": edu_line,
        "last_experience": last,
        "previous_experiences": [
            {k: e[k] for k in ("job_title", "company", "duration")} for e in entries[1:6]
        ],
        "matched_skills": matched,
        "missing_or_unverified_skills": missing,
        "score": score,
        "recommendation": recommendation,
        "explanation": explanation,
    }
    if not explain:
        return data

    # ---- XAI: every factor with its exact point effect ----
    positive: list[tuple[str, float]] = []
    negative: list[tuple[str, float]] = []
    for it in skill_items:
        if it["found"]:
            positive.append((f"Skill found: {it['skill']}", per_skill))
        else:
            negative.append((f"Skill not found: {it['skill']}", -per_skill))
    for t in title_hit:
        positive.append((f"Job-title word in experience: {t}", pt_title))
    for t in title_miss:
        negative.append((f"Job-title word missing from experience: {t}", -pt_title))
    for t in desc_hit:
        positive.append((f"Job-description word in experience: {t}", pt_desc))
    for t in desc_miss:
        negative.append((f"Job-description word missing from experience: {t}", -pt_desc))
    for t in resp_hit:
        positive.append((f"Responsibility word in experience: {t}", pt_resp))
    for t in resp_miss:
        negative.append((f"Responsibility word missing from experience: {t}", -pt_resp))
    if education_points is not None:
        label = f"Degree {LEVEL_NAMES.get(cand_level)} vs required {LEVEL_NAMES.get(required)}"
        (positive if education_points else negative).append((label, 20.0 if education_points else -20.0))

    def top(items, reverse):
        items = [(f, p) for f, p in items if abs(p) > 0.001]
        items.sort(key=lambda x: x[1], reverse=reverse)
        return [{"factor": f, "points": round(p, 2)} for f, p in items[:5]]

    what_if = [
        f"Explicit evidence of '{s}' in the CV would add {per_skill:.0f} points." for s in missing[:5]
    ]
    if recommendation == "Needs Review":
        if available_weight < 70:
            what_if.append(
                f"Only {available_weight}/100 of the rubric could be verified; at least 70 is needed for "
                "'Potential Match' (for example, state a minimum degree in the job description)."
            )
        else:
            what_if.append(f"{45 - score} more points are needed to reach 'Potential Match'.")
    elif recommendation == "Potential Match":
        needs = []
        if score < 75:
            needs.append(f"{75 - score} more points")
        if missing:
            needs.append("evidence for every required skill")
        if available_weight < 90:
            needs.append(f"at least 90/100 verifiable rubric weight (currently {available_weight})")
        if needs:
            what_if.append("To reach 'Strong Match': " + ", ".join(needs) + ".")

    data["xai"] = {
        "method": (
            "Exact additive attribution: the score is a weighted sum of rule matches, so each factor's "
            "point effect is its exact contribution (equivalent to SHAP values for an additive model). "
            "Category points are rounded to whole numbers; factor points are shown to 2 decimals."
        ),
        "breakdown": {
            "skills": {"points": skill_points, "max": 40, "per_skill_points": round(per_skill, 2),
                       "items": skill_items},
            "experience": {"points": experience_points, "max": 30,
                           "weights": {"job_title_words": w_title, "job_description_words": w_desc},
                           "matched_title_words": title_hit, "missing_title_words": title_miss,
                           "matched_description_words": desc_hit, "missing_description_words": desc_miss,
                           "points_per_title_word": round(pt_title, 2),
                           "points_per_description_word": round(pt_desc, 2),
                           "section_used": "experience section" if has_exp_section else "whole CV"},
            "education": {"points": education_points, "max": 20,
                          "required": LEVEL_NAMES.get(required), "found": LEVEL_NAMES.get(cand_level),
                          "evidence": edu_line or None,
                          "status": "scored" if education_points is not None else "unverified"},
            "responsibilities": {"points": responsibility_points, "max": 10,
                                 "matched_words": resp_hit, "missing_words": resp_miss,
                                 "points_per_word": round(pt_resp, 2)},
        },
        "top_positive_factors": top(positive, True),
        "top_negative_factors": top(negative, False),
        "what_would_change_the_result": what_if,
        "data_quality_notes": notes,
        "assessable_weight": available_weight,
        "features_used": ["required skills", "experience-section wording", "highest degree",
                          "job-description responsibility wording"],
        "protected_attributes_used": False,
    }
    return data


# --------------------------------------------------
# Per-CV pipeline
# --------------------------------------------------

class DocumentDownloadTimeout(Exception):
    """Raised only when fetching the source CV times out."""


async def process_single_cv(http: httpx.AsyncClient, url: str, spec: JobSpec, explain: bool) -> dict:
    started = time.perf_counter()
    filename = ""
    timings: dict = {}
    try:
        # One slot covers download + extraction, so at most PIPELINE_CONCURRENCY
        # files are held in memory at any moment.
        async with pipeline_semaphore:
            stage = time.perf_counter()
            try:
                filename, file_bytes = await download_file(http, url)
            except httpx.TimeoutException as exc:
                raise DocumentDownloadTimeout from exc
            timings["download_seconds"] = round(time.perf_counter() - stage, 2)
            timings["file_bytes"] = len(file_bytes)

            ext = detect_extension(filename, file_bytes)
            if ext not in SUPPORTED:
                raise ValueError("Unsupported file type. Use PDF, DOCX, PNG, JPG or JPEG.")

            text_key = "txt:" + hashlib.sha256(file_bytes).hexdigest()
            stage = time.perf_counter()
            cached_text = await cache_get(text_key)
            if cached_text:
                cv_text = cached_text["text"]
            else:
                loop = asyncio.get_running_loop()
                cv_text = await loop.run_in_executor(process_pool, extract_cv_text, f"cv{ext}", file_bytes)
                await cache_put(text_key, {"text": cv_text})
            timings["extract_seconds"] = round(time.perf_counter() - stage, 2)
            timings["cv_chars"] = len(cv_text)
            del file_bytes

        stage = time.perf_counter()
        data = score_cv(spec, cv_text, explain)
        timings["score_seconds"] = round(time.perf_counter() - stage, 3)

        return {
            "success": True,
            "filename": filename,
            "url": url,
            "seconds": round(time.perf_counter() - started, 2),
            "timings": timings,
            "_text": cv_text,  # removed before the response is returned
            **data,
        }
    except Exception as exc:
        message = str(exc)
        if isinstance(exc, DocumentDownloadTimeout):
            message = "Document download timed out while fetching the CV URL."
        elif isinstance(exc, httpx.TimeoutException):
            message = "A network operation timed out during CV processing."
        elif isinstance(exc, httpx.HTTPStatusError):
            message = f"Document server returned HTTP {exc.response.status_code}."
        elif isinstance(exc, httpx.RequestError):
            message = "Could not download the document URL."
        return {
            "success": False,
            "filename": filename,
            "url": url,
            "error": message[:300] or "CV processing failed.",
            "seconds": round(time.perf_counter() - started, 2),
        }


async def run_screening(
    job_title: str,
    job_description: str,
    required_skills: str,
    paths: list[str],
    progress: dict | None = None,
    debug_text: bool = False,
    explain: bool = True,
) -> dict:
    started = time.perf_counter()
    spec = build_job_spec(job_title, job_description, required_skills)

    async def one(url: str) -> dict:
        result = await process_single_cv(http_client, url, spec, explain)
        if progress is not None:
            progress["done"] += 1
        return result

    results_list = await asyncio.gather(*(one(u) for u in paths))
    results = [r for r in results_list if r["success"]]
    errors = [{"url": r["url"], "error": r["error"]} for r in results_list if not r["success"]]
    results.sort(key=lambda r: (r["score"], len(r["matched_skills"])), reverse=True)

    for rank, item in enumerate(results, start=1):
        text = item.pop("_text", "")
        if debug_text:
            item["cv_preview"] = text[:2000]
        item["rank"] = rank

    return {
        "job_title": job_title,
        "total_urls": len(paths),
        "successful": len(results),
        "failed": len(errors),
        "total_seconds": round(time.perf_counter() - started, 1),
        "results": results,
        "errors": errors,
        "human_review_required": True,
    }


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

def validate_request(job_title, job_description, required_skills, document_paths) -> list[str]:
    paths = list(dict.fromkeys(p.strip() for p in document_paths.split(",") if p.strip()))
    if not paths:
        raise HTTPException(status_code=400, detail="Provide at least one document path.")
    if len(paths) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} documents per request.")
    if len(job_title) > 500 or len(job_description) > 20000 or len(required_skills) > 5000:
        raise HTTPException(status_code=400, detail="Job details exceed the allowed length.")
    if http_client is None:
        raise HTTPException(status_code=503, detail="HTTP client is not ready.")
    return paths


@app.get("/")
def home():
    return {
        "status": "running",
        "version": VERSION,
        "service": "ATS Screening API",
        "scoring": "rule-based with built-in explainability (XAI)",
        "extract_workers": EXTRACT_WORKERS,
    }


@app.get("/health")
async def health():
    return {"api": "running", "version": VERSION}


@app.post("/screen", response_model=dict)
async def screen_candidates(
    job_title: str = Form(...),
    job_description: str = Form(...),
    required_skills: str = Form(...),
    document_paths: str = Form(..., description="Comma-separated list of document URLs"),
    explain: bool = Form(True, description="Include the per-CV explainability report (xai)"),
    debug_text: bool = Form(False, description="Include the first 2000 extracted characters of each CV"),
):
    paths = validate_request(job_title, job_description, required_skills, document_paths)
    return await run_screening(job_title, job_description, required_skills, paths,
                               debug_text=debug_text, explain=explain)


@app.post("/screen/async", response_model=dict)
async def screen_async(
    job_title: str = Form(...),
    job_description: str = Form(...),
    required_skills: str = Form(...),
    document_paths: str = Form(...),
    explain: bool = Form(True),
    debug_text: bool = Form(False),
):
    """Starts a background job so 500-CV requests don't hit HTTP timeouts."""
    paths = validate_request(job_title, job_description, required_skills, document_paths)
    job_id = uuid.uuid4().hex
    job = {"status": "running", "done": 0, "total": len(paths), "result": None, "error": None}
    _jobs[job_id] = job
    while len(_jobs) > MAX_JOBS:
        _jobs.popitem(last=False)

    async def runner():
        try:
            job["result"] = await run_screening(
                job_title, job_description, required_skills, paths,
                progress=job, debug_text=debug_text, explain=explain,
            )
            job["status"] = "done"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)[:300]

    job["task"] = asyncio.create_task(runner())
    return {"job_id": job_id, "status": "running", "total": len(paths)}


@app.get("/screen/jobs/{job_id}", response_model=dict)
async def screen_job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return {k: v for k, v in job.items() if k != "task"}