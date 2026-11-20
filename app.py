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

VERSION = "4.1.1"

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
# Scanned CVs often have the name in a big decorative header that normal OCR skips.
# The top part of page 1 is OCR'd a second time (cheap) to catch it.
HEADER_STRIP_FRACTION = 0.22

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
PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\s\-().]{8,16}\d)(?!\d)")   # generic; only used to clean name lines

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
#
# extract_cv_text() returns a LIST of text variants of the same CV:
#   * "flow" = the PDF's natural reading order (good for 2-column CVs)
#   * "rows" = text rebuilt row by row, so "Company .......... Dates" stays on one line;
#              columns inside a row are separated by a TAB (good for table-like CVs)
# The parser runs on every variant and the one that yields the most complete result wins.

def _rows_text(page, gap: float = 18.0, ytol: float = 3.5) -> str:
    """Rebuild a PDF page row by row. Cells of one row (big horizontal gap) are joined with a TAB."""
    words = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words") if str(w[4]).strip()]
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list] = []                       # [mean_y, [words]]
    for w in words:
        yc = (w[1] + w[3]) / 2
        if rows and abs(rows[-1][0] - yc) <= ytol:
            rows[-1][1].append(w)
            n = len(rows[-1][1])
            rows[-1][0] = (rows[-1][0] * (n - 1) + yc) / n
        else:
            rows.append([yc, [w]])
    out = []
    for _, row in rows:
        row.sort(key=lambda w: w[0])
        cells = [[row[0]]]
        for prev, w in zip(row, row[1:]):
            if w[0] - prev[2] > gap:
                cells.append([w])
            else:
                cells[-1].append(w)
        out.append("\t".join(" ".join(x[4] for x in c) for c in cells))
    return "\n".join(out)


def _ocr_page_text(page) -> str:
    """OCR one page; column gaps (4+ spaces) become TABs."""
    textpage = page.get_textpage_ocr(language="eng", dpi=OCR_DPI, full=True)
    raw = page.get_text("text", textpage=textpage, sort=True)
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "camscanner" in line.casefold():
            continue
        lines.append(re.sub(r"\s{4,}", "\t", line))
    return "\n".join(lines)


def _ocr_header_lines(img: Image.Image) -> list[str]:
    """OCR only the top strip of the first page: this is where big decorative names live.

    Light-grey text on a dark banner (name + contact line) is missed by normal OCR, so the strip is
    also OCR'd with a few "light text -> black on white" thresholds. Lines are merged (no duplicates).
    """
    w, h = img.size
    strip = img.crop((0, 0, w, max(1, int(h * HEADER_STRIP_FRACTION)))).convert("L")
    lines: list[str] = []

    def add(text: str) -> None:
        for l in text.splitlines():
            l = l.strip()
            if l and "camscanner" not in l.casefold() and l not in lines:
                lines.append(l)

    add(pytesseract.image_to_string(strip, lang="eng", config="--psm 6"))
    right = strip.crop((int(w * 0.30), 0, w, strip.height))
    right = ImageOps.autocontrast(right.resize((right.width * 2, right.height * 2), Image.LANCZOS), cutoff=1)
    add(pytesseract.image_to_string(right, lang="eng", config="--psm 6"))
    lo, hi = strip.getextrema()
    dark = right.point(lambda p, t=lo + (hi - lo) * 0.30: 0 if p < t else 255)   # big dark name on a light banner
    add(pytesseract.image_to_string(dark, lang="eng", config="--psm 6"))
    if hi - lo > 40 and sum(strip.resize((32, 8)).getdata()) / 256 < 140:      # dark banner
        big = strip.crop((int(w * 0.22), 0, w, strip.height))
        if big.width < 1600:
            big = big.resize((big.width * 2, big.height * 2), Image.LANCZOS)
        for frac in (0.40, 0.33):
            th = lo + (hi - lo) * frac
            add(pytesseract.image_to_string(big.point(lambda p, t=th: 0 if p > t else 255),
                                            lang="eng", config="--psm 6"))
    return lines


def _pdf_variants(file_bytes: bytes) -> list[str]:
    flow_pages: list[str] = []
    row_pages: list[str] = []
    ocr_pages = 0
    total = 0
    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_no, page in enumerate(pdf):
            flow = page.get_text().strip()
            rows = ""
            if len(flow) >= MIN_TEXT_BEFORE_OCR:
                rows = _rows_text(page).strip()
            elif ocr_pages < OCR_MAX_PAGES:            # OCR is slow: only a couple of scanned pages
                ocr_pages += 1
                try:
                    flow = _ocr_page_text(page)
                    if page_no == 0:
                        pix = page.get_pixmap(dpi=200)
                        header = _ocr_header_lines(Image.open(io.BytesIO(pix.tobytes("png"))))
                        flow = "\n".join(header + [flow])
                except Exception:
                    flow = ""
                rows = flow
            if flow:
                flow_pages.append(flow)
                row_pages.append(rows or flow)
                total += len(flow)
            if total >= MAX_TEXT_CHARS:
                break
    variants = ["\n".join(flow_pages), "\n".join(row_pages)]
    return [v for i, v in enumerate(variants) if v.strip() and v not in variants[:i]]


def _despace_cell(cell: str) -> str:
    """'B A S H A M A' -> 'BASHAMA' (letter-spaced designer headings); other text is untouched."""
    toks = cell.split()
    if len(toks) >= 3 and sum(len(t) == 1 for t in toks) >= 0.75 * len(toks):
        return "".join(toks)
    return cell


def _normalize_text(text: str) -> str:
    text = text.replace("\r", "").replace("\u00a0", " ")
    text = "\n".join("\t".join(_despace_cell(c) for c in line.split("\t")) for line in text.split("\n"))
    text = re.sub(r"[…\.]{3,}", " … ", text)                          # leader dots  "Hospital ........ 2019"
    text = re.sub(r"(?<![0-9A-Za-z])2[kK](\d{2})(?![0-9A-Za-z])", r"20\1", text)   # 2K18 -> 2018
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"[ ]*\t[ ]*", "\t", text)
    text = re.sub(r"\t{2,}", "\t", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text[:MAX_TEXT_CHARS]


def extract_cv_text(filename: str, file_bytes: bytes) -> list[str]:
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        variants = _pdf_variants(file_bytes)

    elif extension == ".docx":
        document = Document(io.BytesIO(file_bytes))
        lines = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    value = cell.text.strip().replace("\n", " ")
                    if value and value not in cells:
                        cells.append(value)
                if cells:
                    lines.append("\t".join(cells))
        variants = ["\n".join(lines)]

    elif extension in {".png", ".jpg", ".jpeg"}:
        with Image.open(io.BytesIO(file_bytes)) as image:
            image = ImageOps.exif_transpose(image).convert("L")
            image.thumbnail((2200, 2200))
            header = _ocr_header_lines(image)
            body = pytesseract.image_to_string(image, lang="eng")
        variants = ["\n".join(header + [body])]

    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, PNG, JPG or JPEG.")

    cleaned = []
    for v in variants:
        v = _normalize_text(v)
        if v and v not in cleaned:
            cleaned.append(v)
    if not cleaned:
        raise ValueError("No readable text was extracted from this CV.")
    return cleaned


# --------------------------------------------------
# Email / phone
# --------------------------------------------------

def regex_email(text: str) -> str:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else ""


# CNIC (Pakistani national ID) looks like 41303-1917695-2 and must never be read as a phone number.
CNIC_RE = re.compile(r"(?<!\d)\d{5}-?\d{7}-?\d(?!\d)")
# Phone numbers must start with 0 or +92 (0092 is also accepted).  Mobile: 03XX XXXXXXX
_PK_MOBILE_RE = re.compile(
    r"(?<![\d+])(?:\+\s?92|0092|0)[\s\-().]*3\d{2}[\s\-().]*\d{3}[\s\-]?\d{4}(?!\d)"
)
# Landline fallback: 0 + 2-digit area code + 6-8 digits (e.g. 022-2780000)
_PK_LANDLINE_RE = re.compile(r"(?<![\d+])0\d{2}[\s\-]?\d{6,8}(?!\d)")


def regex_phone(text: str) -> str:
    # remove CNICs first (unless they are really a 0092... number)
    text = CNIC_RE.sub(lambda m: m.group(0) if m.group(0).startswith("0092") else " ", text)
    m = _PK_MOBILE_RE.search(text)
    if m:
        return re.sub(r"\s+", " ", m.group(0)).strip()
    m = _PK_LANDLINE_RE.search(text)
    return m.group(0).strip() if m else ""


# --------------------------------------------------
# Section headings
# --------------------------------------------------

EXP_NAMES = {
    "experience", "work experience", "professional experience", "employment",
    "employment history", "work history", "career history",
    "internship experience", "professional internships", "internship",
    "work experience and internships",
}
EDU_NAMES = {
    "education", "educational background", "education background",
    "academic background", "academic qualifications", "academic qualification",
    "qualifications", "educational qualifications", "degree",
}
OTHER_NAMES = {
    "skills", "technical skills", "skills competencies", "skills and competencies",
    "projects", "project", "certifications",
    "certification", "summary", "profile", "objective", "languages",
    "references", "interests", "achievements", "awards", "training",
    "courses", "personal details", "personal information", "publications", "hobbies",
}

_EXP_HEAD = re.compile(
    r"^(?:work |professional |internship |employment |career )?"
    r"(?:experi\w*|employment|history|internships?|professional)"
    r"(?: history| experience| and internships?)?$")
_EDU_HEAD = re.compile(
    r"^(?:formal |academic |educational |education )?(?:education|qualifications?|background|degree)"
    r"(?: (?:and )?(?:certifications?|training|qualifications?))?$")
_OTHER_HEAD = re.compile(
    r"^(?:key responsibilit\w*.*|responsibilit\w*|achievements?.*|declaration|references?|"
    r"personal (?:details|information|profile)|(?:computer |technical |soft |key |core |professional )?"
    r"(?:skills?|competenc\w+)(?: and competenc\w+| competenc\w+)?|"
    r"why should i be selected|projects?|certifications?|summary|profile|professional summary|objective|"
    r"career objective|professional objective|languages?|interests|hobbies|awards|training|courses|"
    r"publications|volunteer(?:ing)? (?:work|experience)|extra[- ]?curricular.*)$")


_COMPACT_HEADS = {
    "workexperience": "experience", "professionalexperience": "experience", "experience": "experience",
    "employmenthistory": "experience", "workhistory": "experience", "internshipexperience": "experience",
    "education": "education", "educationalbackground": "education", "academicbackground": "education",
    "skills": "other", "contact": "other", "professionalsummary": "other", "summary": "other",
    "profile": "other", "languages": "other", "references": "other", "projects": "other",
    "certifications": "other", "hobbies": "other", "interests": "other", "achievements": "other",
}


def _heading_kind(line: str):
    """'experience' | 'education' | 'other' | None for a short heading-like line."""
    line = line.strip()
    if not line or len(line) > 45:
        return None
    key = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", line.casefold())).strip()
    if not key:
        return None
    compact = _COMPACT_HEADS.get(key)                 # "workexperience" (letter-spaced heading)
    if compact:
        return compact
    if _EDU_HEAD.match(key):
        return "education"
    if _EXP_HEAD.match(key):
        return "experience"
    if _OTHER_HEAD.match(key) or key in OTHER_NAMES:
        return "other"
    return None


def _label_lines(lines: list[str]) -> list[str | None]:
    """Section of every line: 'experience' | 'education' | None | 'heading' (the heading line itself)."""
    labels: list[str | None] = []
    current = None
    for line in lines:
        kind = _heading_kind(line.split("\t")[0])
        if kind:
            current = kind if kind in ("experience", "education") else None
            labels.append("heading")
        else:
            labels.append(current)
    return labels


def split_sections(text: str) -> dict[str, str]:
    """Pull the experience and education sections out of the CV text."""
    lines = text.split("\n")
    labels = _label_lines(lines)
    out = {"experience": [], "education": []}
    for line, label in zip(lines, labels):
        if label in out:
            out[label].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


# --------------------------------------------------
# Degrees
# --------------------------------------------------

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
    # Pharm D is a 5-year professional degree; treated as bachelor-equivalent (change 3 -> 4 for master's)
    (3, re.compile(r"(?<!\w)(?:pharm[\s.\-]?d|doctor of pharmacy|b[\s.]?pharm(?:acy)?|mbbs|bds|dpt)(?!\w)", re.I)),
    (3, re.compile(
        r"(?<!\w)(?:bachelor'?s?|undergraduate|b\.?sc|bscs|bsit|bsse|bcs|bca|bba|"
        r"beng|b\.s\.|b\.e\.|b\.?tech|bs)(?!\w)", re.I)),
    (2, re.compile(r"(?<!\w)associate'?s? degree(?!\w)", re.I)),
    (1, re.compile(
        r"(?<!\w)(?:high school|secondary school|matric\w*|hssc|h\.?\s?s\.?\s?c|s\.?\s?s\.?\s?c|"
        r"intermediate)(?!\w)", re.I)),
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


# --------------------------------------------------
# Candidate name
# --------------------------------------------------

_BULLETS = "•-*▪●◦·➢►"
_BULLET_START = re.compile(r"^[•▪●◦·➢►*\-–]\s*")
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
    r"technician|associate|freelance|freelancer|tester|qa|devops|"
    r"pharmacist|pharmacy|doctor|nurse|lecturer|accountant|clerk|dispenser"
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
NAME_STOP_WORDS = re.compile(
    r"\b(?:cnic|nic|no|dob|birth|house|father|nationality|gender|religion|marital|licensed|licence|license|"
    r"qualified|about|organization|duration|therapy|management|patient|care|challenges|village|housing|"
    r"society|cottages|tower|sports|near|population|office|dadu|references?|declaration|personal|details?|"
    r"pharm|rph|pharmd)\b",
    re.I)
_NAME_WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|[.'\-])*")  # letters (any language), dots, apostrophes, hyphens
_REL_SPLIT = re.compile(r"(?i)[dsw]\s?/\s?o\b\.?|\b(?:daughter|son|wife) of\b")   # D/O, S/O, W/O
_TITLE_PREFIX = re.compile(r"^(?:dr|mr|mrs|ms|miss|engr|prof|sir|madam)\b\.?\s*", re.I)


def _valid_name(seg: str) -> bool:
    words = seg.split()
    letters = [re.sub(r"[^\w]", "", w) for w in words]
    return (
        2 <= len(words) <= 4
        and len(seg) <= 40
        and all(_NAME_WORD.fullmatch(w) for w in words)
        and all(len(l) >= 2 for l in letters) and any(len(l) >= 3 for l in letters)   # rejects "R p h", "A Ie"
        and not NAME_STOP.search(seg)
        and not NAME_STOP_WORDS.search(seg)
        and not TITLE_RE.search(seg)
        and not NOT_NAME_WORDS.search(seg)
    )


def _normalize_case(name: str) -> str:
    """'SADAM QURESHI' and 'sadam qureshi' both become 'Sadam Qureshi'; mixed case is kept."""
    name = re.sub(r"\s+", " ", name).strip()
    return name.title() if name.isupper() or name.islower() else name


def _clean_name_line(line: str) -> str:
    line = EMAIL_RE.sub(" ", line)
    line = re.sub(r"https?://\S+|www\.\S+|linkedin\.com\S*|github\.com\S*", " ", line, flags=re.I)
    line = CNIC_RE.sub(" ", line)
    line = PHONE_RE.sub(" ", line)
    line = _REL_SPLIT.split(line)[0]                 # "NAME D/O FATHER" -> "NAME"
    return re.sub(r"\([^)]*\)?|\)", " ", line)       # drop (...) and stray brackets


def _name_candidates(lines: list[str], stop_at_heading: bool) -> list[tuple[int, str]]:
    cands: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line[:1] in _BULLETS or len(line) > 90:
            continue
        if stop_at_heading and _heading_kind(line.split("\t")[0]):
            break                                    # the name comes before the first section
        parts = [p for p in line.split("\t") if p.strip()]
        if 2 <= len(parts) <= 3:
            cands.append((i, " ".join(parts)))
        for seg in re.split(r"\t|\s[|•·–—\-]\s|[|•·,]", _clean_name_line(line)):
            seg = _TITLE_PREFIX.sub("", seg.strip(" :"))
            stripped = re.sub(r"^(?:[a-z]{1,2}\s+)+", "", seg)      # OCR noise: "as AKASH Kumar"
            if len(stripped.split()) >= 2:
                seg = stripped
            if not seg:
                continue
            cands.append((i, seg))
            title = TITLE_RE.search(seg)
            if title and title.start() > 0:          # "Sadam Qureshi Full Stack Developer"
                cands.append((i, seg[: title.start()].strip()))
    return cands


def _email_score(name: str, email: str) -> int:
    """How many words of the name appear inside the e-mail address (shahjahanjarwar2@... -> Shahjahan, Jarwar)."""
    handle = re.sub(r"[^a-z]", "", email.split("@")[0].casefold())
    return sum(1 for t in re.findall(r"[a-z]+", name.casefold()) if len(t) >= 3 and t in handle)


def guess_name(text: str, email: str = "") -> tuple[str, str]:
    """Returns (name, source) where source is 'label', 'email-match', 'text' or 'email'."""
    m = re.search(r"(?im)^\s*(?:full\s+|candidate\s+)?name\s*[:\-]\s*(.+)$", text[:2500])
    if m:
        val = _TITLE_PREFIX.sub("", _clean_name_line(m.group(1)).strip())
        if _valid_name(val):
            return _normalize_case(val), "label"

    lines = text.split("\n")

    # 1) A line anywhere near the top whose words also appear in the e-mail address is almost surely the name.
    #    This works even when the PDF text order is scrambled and the name is not on the first line.
    if email:
        scored = [(_email_score(seg, email), -n, seg)
                  for n, (_, seg) in enumerate(_name_candidates(lines[:90], stop_at_heading=False))
                  if _valid_name(seg)]
        scored = [s for s in scored if s[0] >= 2]
        if scored:
            return _normalize_case(max(scored)[2]), "email-match"

    # 2) First valid name-looking text before the first section heading.
    cands = _name_candidates(lines[:15], stop_at_heading=True)
    for _, seg in cands:
        if _valid_name(seg):
            return _normalize_case(seg), "text"
    # first name and last name on two consecutive lines
    for (i1, a), (i2, b) in zip(cands, cands[1:]):
        if i2 == i1 + 1 and len(a.split()) == 1 and len(b.split()) == 1 and _valid_name(f"{a} {b}"):
            return _normalize_case(f"{a} {b}"), "text"

    # 3) Last resort: first.last@example.com -> "First Last"
    if email:
        parts = [p for p in re.split(r"[._\-\d]+", email.split("@")[0]) if len(p) >= 2]
        if len(parts) >= 2:
            return " ".join(p.capitalize() for p in parts[:3]), "email"
    return "", ""


# --------------------------------------------------
# Education
# --------------------------------------------------

def _clean_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t|•-–—▪●◦·➢►*")


_EDU_HEAD_LINE = re.compile(
    r"^(?:pharm[\s.\-]?d|doctor of pharmacy|b[\s.]?pharm\w*|mbbs|bds|dpt|ph\.?\s?d|m\.?\s?phil|m\.?\s?sc|mba|"
    r"bs(?:c|it|cs|se)?|b\.?\s?sc|bba|bca|bcs|beng|b\.e\.|b\.?\s?tech|mcs|"
    r"ms(?!\s*(?:office|word|excel|sql|access|project|teams|power))|"
    r"bachelor\w*|master\w*|intermediate|matric\w*|hssc|"
    r"h\.?\s?s\.?\s?c|s\.?\s?s\.?\s?c|a[- ]?levels?|o[- ]?levels?|dae|diploma|associate)"
    r"(?=[\s(,\-–|.:]|$)", re.I)
_INST_RE = re.compile(r"universit|college|school|institute|academy|board|faculty|campus|^govt|^government", re.I)
_GRADE_RE = re.compile(r"(?:g\.?\s?p\.?\s?a|cgpa|gpa|grade|percentage|division|marks|first class)\b.*"
                       r"|\d{2,3}(?:\.\d+)?\s*%", re.I)
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?:\s*[-–—]\s*(?:(?:19|20)\d{2}|present|ongoing))?(?!\d)", re.I)


def extract_education_records(text: str) -> list[dict]:
    """Degree rows from stacked lines, side-by-side table cells (TAB) or one-line rows."""
    raw = [x for x in text.splitlines() if x.strip()]
    lines = [_clean_line(x.split("\t")[0]) for x in raw]          # first column of each row
    labels = _label_lines(raw)
    heads = [i for i, l in enumerate(lines) if l and _EDU_HEAD_LINE.match(l) and labels[i] != "heading"]
    records = []
    for n, i in enumerate(heads):
        cells = [_clean_line(c) for c in raw[i].split("\t") if _clean_line(c)]
        if len(cells) >= 3:                                      # a table row: Degree | Year | Grade | Institution
            year = next((ym[0] for c in cells[1:] if (ym := _YEAR_RE.findall(c))), "")
            grade = next((_clean_line(g.group(0)) for c in cells[1:] if (g := _GRADE_RE.search(c))), "")
            inst = next((c for c in cells[1:] if _INST_RE.search(c)), "")
            records.append({"degree": cells[0][:150], "year": year, "grade": grade, "institution": inst})
            continue

        end = min(heads[n + 1] if n + 1 < len(heads) else len(lines), i + 8)
        block = []
        for k in range(i, end):
            if k > i and labels[k] == "heading":
                break
            block.append(lines[k])

        year = next((ym[0] for l in block if (ym := _YEAR_RE.findall(l))), "")
        grade = ""
        for l in block:
            g = _GRADE_RE.search(l)
            if g:
                grade = _clean_line(g.group(0))
                break
        inst = ""
        for k, l in enumerate(block):
            if not _INST_RE.search(l):
                continue
            seg = l
            if k == 0:
                head = _EDU_HEAD_LINE.match(l)
                rest = l[head.end():] if head else l
                rest = re.sub(r"^\s*\([^)]*\)\s*", "", rest)           # "Doctor of Pharmacy (Pharm D) ..."
                parts = [p.strip() for p in re.split(r"\s*[|,]\s*| - | – ", rest) if p.strip()]
                seg = next((p for p in parts if _INST_RE.search(p)), "")
                if not seg:
                    continue
            nxt = block[k + 1] if k + 1 < len(block) else ""
            if (nxt and len(nxt) <= 30 and not re.search(r"\d", nxt)
                    and not _GRADE_RE.search(nxt) and not _EDU_HEAD_LINE.match(nxt)
                    and not _INST_RE.search(nxt)):
                seg = f"{seg.rstrip('.,')} {nxt}"                      # "...Girls College" + "Hyderabad"
            inst = seg.strip(" .,")
            break
        records.append({"degree": block[0][:150], "year": year, "grade": grade, "institution": inst})

    # Merge records of the same degree ("PHARM-D" title + "Pharm D ... University" row), keep the richer one.
    merged: dict[str, dict] = {}
    order: list[str] = []
    for rec in records:
        key = re.sub(r"[^a-z]", "", rec["degree"].casefold())
        if key not in merged:
            merged[key] = rec
            order.append(key)
        elif sum(bool(v) for v in rec.values()) > sum(bool(v) for v in merged[key].values()):
            merged[key] = rec
    return [merged[k] for k in order]


def _best_education(records: list[dict], text: str, sections: dict) -> tuple[int, str, dict | None]:
    best, best_level = None, 0
    for rec in records:
        level = max(degree_levels(rec["degree"]) or {0})
        if level > best_level:
            best, best_level = rec, level
    if best:
        return best_level, best["degree"], best
    level, line = best_degree(sections["education"])
    if not level:
        level, line = best_degree(text)
    return level, line, None


# --------------------------------------------------
# Work experience
# --------------------------------------------------

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_YEAR4 = r"(?:19|20)\d{2}"
_DATE = (rf"(?:(?:\d{{1,2}}[\s\-/.]*)?{_MONTH}[\s,.\-/]*{_YEAR4}|"
         rf"\d{{1,2}}[/.\-]{_YEAR4}|{_YEAR4})")
_DATE_NO_YEAR = rf"(?:\d{{1,2}}[\s\-/.]*)?{_MONTH}"          # "6 March" in "6 March - 6 June 2023"
RANGE_RE = re.compile(
    rf"(?<![\d+])(?P<s>{_DATE}|{_DATE_NO_YEAR})\s*(?:-|–|—|…+|\.{{2,}}|to|till|until)\s*"
    rf"(?P<e>{_DATE}|present|current|now|(?:till\s+|to\s+)?date|ongoing|continu\w*)(?!\d)",
    re.I,
)
_ONGOING = re.compile(r"present|current|now|date|ongoing|continu", re.I)
_MONTH_TOKEN = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", re.I)
_PARTIAL_END = re.compile(
    rf"(?:-|–|—|to|till|until)\s*(?:\d{{1,2}}\s*)?{_MONTH}\s*$", re.I)   # "... - 6 September"  (year wrapped)

ROLE_RE = re.compile(
    r"\b(?:intern|interne|internee|internship|trainee|volunteer|locum|observer|pharmacist|dispenser|technician|"
    r"assistant|officer|manager|executive|analyst|developer|engineer|designer|consultant|nurse|doctor|"
    r"lecturer|teacher|instructor|accountant|clerk|supervisor|coordinator|specialist|in-?charge|hod|head|"
    r"director|resident|fellow|associate|representative|trainer|programmer|tester|administrator|"
    r"freelancer|owner|founder|intern)\b", re.I)
ORG_RE = re.compile(
    r"hospital|pharma\w*|clinic|company|\bltd\b|limited|\bpvt\b|\binc\b|corporation|university|institute|"
    r"college|\bcent(?:er|re)\b|laborator\w*|industries|foundation|\btrust\b|program|school|academy|"
    r"dispensary|health ?care|medical|\bhyd\b|solutions|technolog\w*|systems|\bgroup\b|\bbank\b|"
    r"enterprises|services|nicvd|\bagency\b|\bstore\b|\bmart\b", re.I)

# "1 Month", "6 months", "300 hours": a length of time instead of a date range.
# A lone "August,2025" / "Aug 2025" (one month + year, no range) = a single-date entry.
_SINGLE_DATE_RE = re.compile(rf"^{_MONTH}[\s,.\-/]*{_YEAR4}$", re.I)
_LENGTH_RE = re.compile(r"(?<!\d)\d+\s*(?:months?|weeks?|years?|hours?|days?)\b", re.I)
# A lone label such as "Internship" is a tag next to a job, never a job title by itself.
_BARE_LABEL = re.compile(r"(?:internship|intern|training|volunteer)", re.I)


def _year(value: str) -> int:
    m = re.search(_YEAR4, value)
    return int(m.group(0)) if m else 0


def _cells(line: str) -> list[str]:
    """Cells of a row: TAB-separated columns, and 'A | B' pipe-separated parts."""
    out: list[str] = []
    for c in line.split("\t"):
        out.extend(p.strip() for p in re.split(r"\s*\|\s*", c) if p.strip())
    if (len(out) >= 2 and re.fullmatch(r"[A-Z]{2,6}", out[0])
            and re.match(r"(?:hospital|clinic|university|institute|college)\b", out[1], re.I)):
        out[:2] = [f"{out[0]} {out[1]}"]
    return out


def _classify(cell: str) -> str:
    if ROLE_RE.search(cell) and len(cell.split()) <= 5:
        return "role"
    if ORG_RE.search(cell):
        return "org"
    return "other"


def _title_company(desc: str) -> tuple[str, str]:
    """'... as Site Pharmacist in IRD Program of Indus hospital' -> (title, company)."""
    d = re.sub(r"\([^)]*\)", " ", desc)
    d = re.sub(r"\s+", " ", re.sub(r"…+", " ", d)).strip(" .:-–")
    title = company = ""
    m = re.search(r"\bas\s+(?:an?\s+|the\s+)?(?P<t>[A-Za-z][A-Za-z &/\-]{1,45}?)\s+"
                  r"(?:in|at|with|for)\s+(?P<c>.+)$", d, re.I)
    if m:
        title, rest = m.group("t"), m.group("c")
        later = re.search(r"\bas\s+(?:an?\s+)?([A-Za-z][A-Za-z &/\-]{2,45})$", rest, re.I)
        if later:                                   # "... Hospital as hospital pharmacist"
            title, rest = later.group(1), rest[:later.start()]
        at = re.search(r"\bat\s+(.+)$", rest, re.I)
        company = at.group(1) if at else rest
    else:
        m = re.search(r"\b(?:at|in|with)\s+(?P<c>.+?)\s+as\s+(?:an?\s+)?"
                      r"(?P<t>[A-Za-z][A-Za-z &/\-]{2,45})$", d, re.I)
        if m:
            company, title = m.group("c"), m.group("t")
        else:
            m = re.search(r"\b(?:at|in|with)\s+(.+)$", d, re.I)
            company = m.group(1) if m else ""
    return title.strip(" .,:-"), company.strip(" .,:-")


def _sort_entries(entries: list[dict]) -> list[dict]:
    entries.sort(key=lambda e: e["_key"], reverse=True)
    for e in entries:
        e.pop("_key")
    return entries


def parse_experiences(text: str) -> list[dict]:
    """
    Rule-based job extraction, newest first. Handles three layouts:
      1. bullets that end with the dates:   '• Worked as Site Pharmacist in X Hospital (Jun-2020 to Nov-2021)'
      2. a table-like row:                  'Company Name <TAB> 1 March 2022 - 28 Feb 2023' + role on the next line
      3. role/company lines above the date; and, when no dates exist at all, role + company lines.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    n = len(lines)
    labels = _label_lines(lines)
    rows = [_cells(l) for l in lines]
    consumed = [False] * n
    entries: list[dict] = []

    # ---- 1. bullet items (wrapped lines are joined) ----
    i = 0
    while i < n:
        if not (labels[i] == "experience" and rows[i] and _BULLET_START.match(rows[i][0])):
            i += 1
            continue
        j = i + 1
        while (j < n and labels[j] == "experience" and len(rows[j]) == 1
               and not _BULLET_START.match(rows[j][0])):
            j += 1
        item = re.sub(r"\s+", " ", " ".join(" ".join(rows[k]) for k in range(i, j))).strip()
        item = _BULLET_START.sub("", item)
        m = RANGE_RE.search(item)
        if m:
            start, end = m.group("s").strip(), m.group("e").strip()
            duration = f"{start} - {end}"
            key = (9999 if _ONGOING.search(end) else _year(end), _year(start) or _year(end))
            desc = item[:m.start()] + " " + item[m.end():]
        else:
            ym = re.search(rf"(?<!\d)({_YEAR4})\W*$", item)
            if not ym:
                i = j
                continue
            duration, key = ym.group(1), (int(ym.group(1)), int(ym.group(1)))
            desc = item[:ym.start()]
        title, company = _title_company(desc)
        if title or company:
            summary = re.sub(r"\s+", " ", re.sub(r"…+|\(\s*\)", " ", desc)).strip(" .:-–")[:250]
            entries.append({"job_title": title[:100], "company": company[:100], "duration": duration,
                            "location": "", "summary": summary, "_key": key})
            for k in range(i, j):
                consumed[k] = True
        i = j

    # ---- 2/3. date rows with role/company beside, below or above them ----
    for r in range(n - 1):                          # a date whose year wrapped onto the next row
        for c_idx, cell in enumerate(rows[r]):
            if _PARTIAL_END.search(cell):
                nxt = next((k for k, c in enumerate(rows[r + 1]) if re.fullmatch(_YEAR4, c)), None)
                if nxt is not None:
                    rows[r][c_idx] = f"{cell} {rows[r + 1][nxt]}"
                    rows[r + 1].pop(nxt)

    anchors: list[tuple[int, int, re.Match]] = []
    for r in range(n):
        if consumed[r]:
            continue
        if labels[r] == "education" and any(
                _EDU_HEAD_LINE.match(c) or degree_levels(c)
                or re.search(r"matric|intermediate|pharm[\s.\-]?d|bachelor|master|mbbs|degree|diploma", c, re.I)
                for c in rows[r]):
            continue                                     # a degree row: its dates are not a job
        for c_idx, cell in enumerate(rows[r]):
            m = RANGE_RE.search(cell)
            if not m and labels[r] == "experience" and _SINGLE_DATE_RE.match(cell.strip()):
                d = cell.strip()                         # "August,2025" -> start = end
                m = RANGE_RE.search(f"{d} - {d}")
            if not m:
                continue
            if not (_MONTH_TOKEN.search(m.group(0)) or labels[r] == "experience"):
                continue                            # year-only ranges outside the experience section = education
            if c_idx == 0 and _EDU_HEAD_LINE.match(cell):
                continue
            anchors.append((r, c_idx, m))
            break

    anchor_rows = {a[0] for a in anchors}
    claimed: set[tuple[int, int]] = set()

    def eligible(r: int, k: int, c: str) -> bool:
        return ((r, k) not in claimed and 2 <= len(c) <= 70 and len(c.split()) <= 9
                and not _BULLET_START.match(c) and not RANGE_RE.search(c)
                and not re.fullmatch(r"[\W\d_]+", c) and not _EDU_HEAD_LINE.match(c)
                and not _heading_kind(c) and not re.search(r":\s+\S", c))

    alone = sum(1 for (r, j, _) in anchors
                if not any(eligible(r, k, c) for k, c in enumerate(rows[r]) if k != j))
    header_above = bool(anchors) and alone > len(anchors) / 2     # "Title / Company / Date" layout

    built = []
    for (r, j, m) in anchors:
        found = {"role": "", "org": ""}

        def take(cells):
            for rr, k, c in cells:
                kind = _classify(c)
                if kind in found and not found[kind]:
                    found[kind] = c
                    claimed.add((rr, k))

        def row_cells(rr):
            return [(rr, k, c) for k, c in enumerate(rows[rr]) if eligible(rr, k, c)]

        def look_forward():
            for rr in range(r + 1, min(n, r + 3)):
                if rr in anchor_rows:
                    break
                cells = row_cells(rr)
                if not cells:
                    break
                take(cells)

        def look_back():
            for rr in range(r - 1, max(-1, r - 4), -1):
                if rr in anchor_rows:
                    break
                if labels[rr] == "heading":
                    continue
                take(row_cells(rr))

        same = [(r, k, c) for k, c in enumerate(rows[r]) if k != j and eligible(r, k, c)]
        take(same)
        if not found["org"]:                        # "Google <TAB> 2020 - 2022"
            for rr, k, c in same:
                if (rr, k) not in claimed and _classify(c) == "other":
                    found["org"] = c
                    claimed.add((rr, k))
                    break
        if header_above:
            look_back()
            if not (found["role"] and found["org"]):
                look_forward()          # "Title / Date / Company" layouts
        else:
            if not (found["role"] and found["org"]):
                look_forward()
            if not (found["role"] and found["org"]):
                look_back()

        start, end = m.group("s").strip(), m.group("e").strip()
        key = (9999 if _ONGOING.search(end) else _year(end), _year(start) or _year(end))
        built.append({"r": r, "role": found["role"], "org": found["org"],
                      "duration": f"{start} - {end}", "_key": key})

    for idx, b in enumerate(built):
        if not b["org"] and b["role"] and idx > 0 and built[idx - 1]["org"]:
            b["org"] = built[idx - 1]["org"]        # 2nd stint at the same place: "Volunteer <TAB> dates"
        if not (b["role"] or b["org"]):
            continue
        stop = built[idx + 1]["r"] if idx + 1 < len(built) else n
        body = []
        for rr in range(b["r"] + 1, stop):
            if labels[rr] == "heading" or (labels[rr] == "education" and labels[b["r"]] != "education"):
                break
            for k, c in enumerate(rows[rr]):
                if (rr, k) not in claimed and not RANGE_RE.search(c):
                    body.append(_BULLET_START.sub("", c))
        entries.append({"job_title": b["role"][:100], "company": b["org"][:100], "duration": b["duration"],
                        "location": "", "summary": re.sub(r"\s+", " ", " ".join(body)).strip()[:250],
                        "_key": b["_key"]})

    # ---- 4. no dates anywhere: role line + company line (scanned CVs, "Role | Company" lines) ----
    if not entries:
        entries = _undated_experiences(rows, labels)

    return _sort_entries(entries)


def _undated_experiences(rows: list[list[str]], labels: list) -> list[dict]:
    has_section = any(l == "experience" for l in labels)
    out: list[dict] = []
    used_org: set[tuple[int, int]] = set()
    for r, cells in enumerate(rows):
        if has_section and labels[r] != "experience":
            continue
        for k, c in enumerate(cells):
            if not (2 <= len(c) <= 60 and len(c.split()) <= 5 and _classify(c) == "role"):
                continue
            if (_BULLET_START.match(c) or _EDU_HEAD_LINE.match(c) or _heading_kind(c)
                    or _BARE_LABEL.fullmatch(c) or re.search(_YEAR4, c)):
                continue
            org = ""
            for rr in (r, r - 1, r - 2):
                if rr < 0:
                    continue
                for kk, cc in enumerate(rows[rr]):
                    if (rr, kk) in used_org or (rr, kk) == (r, k):
                        continue
                    if (_classify(cc) == "org" and not _EDU_HEAD_LINE.match(cc) and len(cc) <= 60
                            and cc[:1].isupper() and not cc.endswith(".")
                            and not re.search(r"universit", cc, re.I)):
                        org = cc
                        used_org.add((rr, kk))
                        break
                if org:
                    break
            if not org:
                continue
            # month / year words printed near the entry (vertical dates like "Jun - July - 2024")
            near = [x for rr in range(max(0, r - 2), min(len(rows), r + 5)) for x in rows[rr]]
            months = [x for x in near if re.fullmatch(_MONTH, x, re.I)]
            year = next((m.group(0) for x in near if (m := re.fullmatch(_YEAR4, x))), "")
            duration = (" - ".join(months[:2]) + (f" {year}" if year else "")).strip()
            if not duration:                       # "1 Month", "6 months" ...
                duration = next((x for rr in range(r, min(len(rows), r + 3))
                                 for x in rows[rr] if _LENGTH_RE.fullmatch(x)), "")
            out.append({"job_title": c[:100], "company": org[:100], "duration": duration,
                        "location": "", "summary": "", "_key": (_year(year), 0), "_row": r})
            break

    # summary = bullet / sentence lines below each entry, up to the next entry or heading
    for idx, e in enumerate(out):
        stop = out[idx + 1]["_row"] if idx + 1 < len(out) else len(rows)
        body = []
        for rr in range(e["_row"] + 1, stop):
            if labels[rr] == "heading":
                break
            body += [_BULLET_START.sub("", c) for c in rows[rr]
                     if len(c.split()) >= 4 or _BULLET_START.match(c)]
        e["summary"] = re.sub(r"\s+", " ", " ".join(body)).strip()[:250]
        e.pop("_row")
    return out


# --------------------------------------------------
# Structured CV extraction (independent of job scoring)
# --------------------------------------------------

def extract_skills_by_category(text: str) -> dict:
    """Read category/value pairs from common skills tables and stacked PDF text."""
    lines = [_clean_line(x.replace("\t", " | ")) for x in text.splitlines()]
    category_names = {
        "pharmaceutical", "analytical", "computer skills", "communication",
        "interpersonal", "technical skills", "soft skills", "clinical skills",
    }
    result = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        if "|" in line:
            left, right = line.split("|", 1)
            if _clean_line(left).casefold() in category_names:
                result[_clean_line(left)] = [x.strip() for x in re.split(r"[,;]", right) if x.strip()]
                i += 1
                continue
        if line.casefold() in category_names and i + 1 < len(lines):
            values = lines[i + 1]
            if values and values.casefold() not in category_names:
                result[line] = [x.strip() for x in re.split(r"[,;]", values) if x.strip()]
                i += 2
                continue
        i += 1
    return result


def extract_personal_details(text: str) -> dict:
    email = regex_email(text)
    phone = regex_phone(text)
    name, name_source = guess_name(text, email)
    return {"candidate_name": name, "name_source": name_source,
            "email": email or None, "phone": phone or None}


def build_profile(text: str) -> dict:
    sections = split_sections(text)
    personal = extract_personal_details(text)
    education = extract_education_records(sections["education"]) or extract_education_records(text)
    return {
        "text": text,
        "sections": sections,
        "personal": personal,
        "education": education,
        "experiences": parse_experiences(text),
        "skills": extract_skills_by_category(text),
    }


def _profile_quality(p: dict) -> int:
    per = p["personal"]
    score = 0
    if per["candidate_name"]:
        score += 1 if per["name_source"] == "email" else 3
    score += bool(per["email"]) + bool(per["phone"])
    score += 2 if p["education"] else 0
    # a job with both a title and a company is worth more than a half-parsed one
    score += sum(2 if (e["job_title"] and e["company"]) else 1 for e in p["experiences"][:4])
    return score


def choose_profile(texts: list[str]) -> dict:
    """Parse every text variant of the CV and keep the most complete result (ties: first variant).
    A name / e-mail / phone that only another variant found is copied into the winner."""
    profiles = [build_profile(t) for t in texts]
    best = profiles[0]
    for p in profiles[1:]:
        if _profile_quality(p) > _profile_quality(best):
            best = p
    for key in ("candidate_name", "name_source", "email", "phone"):
        if not best["personal"][key]:
            for p in profiles:
                if p["personal"][key] and (key != "name_source" or p["personal"]["candidate_name"]
                                           == best["personal"]["candidate_name"] or not best["personal"]["candidate_name"]):
                    best["personal"][key] = p["personal"][key]
                    break
    return best


# --------------------------------------------------
# Scoring with built-in explainability (XAI)
# --------------------------------------------------

SKILL_ALIASES = {
    "rest apis": ["rest api", "restful api", "restful apis", "rest services",
                  "restful services", "restful web services"],
    "fastapi": ["fastapi", "fast api"],
    "python": ["python 2", "python 3", "python programming", "python development"],
    "sql": ["structured query language"],
    "pharm d": ["pharmd", "pharm-d", "doctor of pharmacy"],
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
        # a space inside a skill also matches "-", "." or nothing:  "Pharm D" == "PHARM-D" == "PharmD"
        body = "|".join(re.escape(v).replace(r"\ ", r"[\s\-.]*") for v in variants)
        skills.append((name, re.compile(r"(?<!\w)(?:" + body + r")(?!\w)", re.I)))

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


def score_cv(spec: JobSpec, profile: dict, explain: bool = True) -> dict:
    """
    Preliminary 100-point score (40 skills / 30 experience / 20 education / 10 responsibilities).

    Explainability: the score is a sum of fixed-weight contributions, so every contribution below is
    exact (for an additive model this equals the SHAP value). Nothing is approximated or sampled.
    """
    cv_text = profile["text"].replace("\t", " ")
    sections = {k: v.replace("\t", " ") for k, v in profile["sections"].items()}
    notes: list[str] = []
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
    education_records = profile["education"]
    cand_level, edu_line, best_edu = _best_education(education_records, cv_text, sections)
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

    experiences = profile["experiences"]
    last = experiences[0] if experiences else {
        "job_title": "", "company": "", "duration": "", "summary": ""
    }
    previous = experiences[1:6]

    data = {
        **profile["personal"],
        "highest_education": (best_edu["degree"] if best_edu else edu_line) or None,
        "highest_education_details": best_edu,
        "all_education": education_records,
        "skills": profile["skills"],
        "latest_experience": last,
        "last_experience": last,
        "previous_experiences": previous,
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

            text_key = "txt2:" + hashlib.sha256(file_bytes).hexdigest()
            stage = time.perf_counter()
            cached_text = await cache_get(text_key)
            if cached_text:
                texts = cached_text["texts"]
            else:
                loop = asyncio.get_running_loop()
                texts = await loop.run_in_executor(process_pool, extract_cv_text, f"cv{ext}", file_bytes)
                await cache_put(text_key, {"texts": texts})
            timings["extract_seconds"] = round(time.perf_counter() - stage, 2)
            del file_bytes

        stage = time.perf_counter()
        profile = choose_profile(texts)
        cv_text = profile["text"]
        timings["cv_chars"] = len(cv_text)
        data = score_cv(spec, profile, explain)
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