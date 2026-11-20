# ATS Screening API

**Version:** 4.1.1
**Developer:** Sadam Qureshi

An Applicant Tracking System (ATS) Screening API built with Python and FastAPI. It evaluates resumes against job requirements, calculates candidate matching scores, and returns screening results through REST API endpoints — synchronously, asynchronously, and with explainable results.

## Features

- Resume screening against job descriptions and required skills
- Candidate matching and scoring
- Explainable screening results
- Synchronous screening endpoint
- Asynchronous screening with background job tracking
- REST API documentation via Swagger UI
- Linux deployment with aaPanel, Nginx, and systemd
- Optional Ollama integration for legacy versions or future LLM-powered features

## Technology Stack

- Python 3.10+
- FastAPI
- Uvicorn
- Pydantic
- Document parsing: PyMuPDF, python-docx, pytesseract, Pillow
- Ubuntu / Linux, Git, aaPanel, Nginx, systemd
- Ollama (optional; not required by v4.0.0 scoring)

## Project Structure

```text
AI-ATS-Screener/
├── app.py              # FastAPI application (v4.0.0)
├── requirements.txt
├── run.bat
├── .venv/
└── README.md
```

## Requirements

- Ubuntu Linux server (or any Python-capable environment)
- Python 3.10 or later
- pip and virtual environment support
- Git
- For production: a domain name, Nginx reverse proxy, and HTTPS

# Installation

## 1. System dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git curl
```

## 2. Clone the repository

```bash
git clone https://github.com/SaDaMQureshi/ats-screening-api.git
cd ats-screening-api
```

## 3. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

On Windows, use `.venv\Scripts\activate` instead.

## 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 5. Run locally

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Interactive documentation is available at `http://127.0.0.1:8000/docs`.

# API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API root |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI documentation |
| `/openapi.json` | GET | OpenAPI specification |
| `/screen` | POST | Synchronous screening |
| `/screen/async` | POST | Start asynchronous screening job |
| `/screen/jobs/{job_id}` | GET | Retrieve screening job status |

Verify health:

```bash
curl -sS http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "api": "running",
  "version": "4.0.0"
}
```

## Asynchronous screening

`/screen/async` accepts form fields including `job_title`, `job_description`, `required_skills`, `document_paths`, and optional `explain` and `debug_text`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/screen/async \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "job_title=Python Developer" \
  --data-urlencode "job_description=Python developer with API experience" \
  --data-urlencode "required_skills=Python, FastAPI, SQL" \
  --data-urlencode "document_paths=https://example.com/resume.pdf" \
  --data-urlencode "explain=true"
```

Poll the returned job ID:

```bash
curl http://127.0.0.1:8000/screen/jobs/YOUR_JOB_ID
```

Use `/docs` for exact request/response schemas.

# Ollama (Optional)

v4.0.0 uses rule-based candidate scoring and does **not** require Ollama. It is only needed for the legacy v3 application or future LLM-powered features.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b   # model used by the legacy v3 config
curl -sS http://127.0.0.1:11434/api/tags
```

Installing Ollama does not automatically connect it to the app — the application must explicitly call the Ollama API.

# Production Deployment (aaPanel)

## Systemd service

Create `/etc/systemd/system/ats-screening-api.service`:

```ini
[Unit]
Description=ATS Screening API
After=network.target

[Service]
Type=simple
WorkingDirectory=/www/wwwroot/ats-screening-api
ExecStart=/www/wwwroot/ats-screening-api/venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ats-screening-api
sudo systemctl status ats-screening-api --no-pager -l
sudo journalctl -u ats-screening-api -n 100 --no-pager
```

## aaPanel / Nginx reverse proxy

1. In aaPanel, add your API domain and point its DNS A record to the server IP.
2. Configure reverse proxy upstream: `http://127.0.0.1:8000`
3. Preserve request paths and HTTP methods; enable SSL/HTTPS.

Keep Uvicorn bound to `127.0.0.1` behind the proxy.

# Updating the Application

```bash
git pull
./venv/bin/pip install -r requirements.txt
./venv/bin/python -c "import app; print('Import successful')"
sudo systemctl restart ats-screening-api
```

Keep a known-good backup for rollback.

# Security Recommendations

- Never commit API keys, passwords, tokens, or private configuration.
- Do not commit the virtual environment or Python cache files.
- Do not publish candidate resumes or personally identifiable information.
- Use HTTPS for public traffic; keep Uvicorn on localhost.
- Validate document URLs and file types; apply request-size and timeout limits.
- Use human review for important hiring decisions — do not rely solely on automated scores.

Suggested `.gitignore`:

```gitignore
venv/
.venv/
__pycache__/
*.py[cod]
.env
.env.*
!.env.example
.pytest_cache/
.mypy_cache/
*.log
```

# License

Choose a license before publishing (e.g., MIT). Only add a license file if you intend to release the project under those terms.

# Developer Contact

**Sadam Qureshi**

- Email: pirsadam05@gmail.com
- Phone: 03035609341

For questions or bug reports, use the repository's GitHub Issues section.

---

**ATS Screening API v4.0.0 — Resume screening and candidate matching through a REST API.**
