
# ATS Screening API

**Version:** 4.0.0  
**Developer:** Sadam Qureshi

An Applicant Tracking System (ATS) Screening API built with Python and FastAPI. It helps evaluate resumes against job requirements, calculate candidate matching scores, and return screening results through REST API endpoints.

The API supports synchronous screening, asynchronous screening jobs, job-status tracking, and explainable screening results.

## Features

- Resume screening against job descriptions and required skills
- Candidate matching and scoring
- Explainable screening results
- Synchronous screening endpoint
- Asynchronous screening endpoint
- Background screening job status tracking
- REST API documentation with Swagger UI
- Linux deployment using aaPanel and systemd
- Optional Ollama integration for legacy versions or future LLM-powered features

## Technology Stack

- Python
- FastAPI
- Uvicorn
- Pydantic
- Ubuntu / Linux
- Git and GitHub
- aaPanel
- Nginx reverse proxy
- Ollama (optional; not required by v4.0.0 scoring)

## Project Structure

```text
ats-screening-api/
├── app.py
├── app_optimized_v3.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

The exact project structure may vary. The `venv/` directory should not be committed to GitHub.

## Requirements

- Ubuntu Linux server
- Python 3.10 or later, compatible with the project dependencies
- pip and Python virtual environment support
- Git
- aaPanel or another web server management panel
- A domain name for public deployment
- Nginx reverse proxy and HTTPS for production

# 1. Install System Dependencies

Update the Ubuntu package list:

```bash
sudo apt update
```

Install Python, Git, and required utilities:

```bash
sudo apt install -y python3 python3-pip python3-venv git curl
```

Verify the installations:

```bash
python3 --version
pip3 --version
git --version
```

# 2. Clone the Repository

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

```bash
git clone https://github.com/SaDaMQureshi/ats-screening-api.git
```

Navigate into the project directory:

```bash
cd ats-screening-api
```

# 3. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

# 4. Install Python Dependencies

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, create one based on the actual application imports and tested dependency versions before proceeding.

# 5. Run the API Locally

Start the v4.0.0 application:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Available local endpoints:

| Endpoint | Description |
|---|---|
| `/` | API root |
| `/health` | Health check |
| `/docs` | Interactive Swagger documentation |
| `/openapi.json` | OpenAPI specification |
| `/screen` | Synchronous screening |
| `/screen/async` | Start asynchronous screening |
| `/screen/jobs/{job_id}` | Retrieve screening job status |

Open Swagger UI in your browser:

```text
http://127.0.0.1:8000/docs
```

Check the health endpoint:

```bash
curl -sS http://127.0.0.1:8000/health
```

Expected v4.0.0 response:

```json
{
  "api": "running",
  "version": "4.0.0"
}
```

Press `Ctrl+C` to stop the development server.

# 6. Ollama Installation (Optional)

## Does v4.0.0 require Ollama?

No. The current v4.0.0 implementation uses rule-based candidate scoring and does not require Ollama for that workflow.

Ollama was used by the legacy v3 application. You can install it if you run that version or add LLM-powered functionality that explicitly calls Ollama.

## Install Ollama on Ubuntu

Official website:

https://ollama.com/download

Standard Linux installation command:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Review the installation script and security implications before executing it on a production server.

Verify the installation:

```bash
ollama --version
```

Check the service:

```bash
sudo systemctl status ollama --no-pager
```

Enable and start the service if needed:

```bash
sudo systemctl enable --now ollama
```

## Download the Recommended Legacy Model

The previous v3 configuration used:

```text
qwen2.5:1.5b
```

Download the model:

```bash
ollama pull qwen2.5:1.5b
```

Run it interactively:

```bash
ollama run qwen2.5:1.5b
```

List downloaded models:

```bash
ollama list
```

Check the local Ollama API:

```bash
curl -sS http://127.0.0.1:11434/api/tags
```

### Model Selection

- `qwen2.5:1.5b` is a relatively small model for initial experiments on resource-constrained servers.
- Larger models may offer better results on some tasks but require more memory and processing capacity.
- Evaluate models using representative resumes and job descriptions before production use.

Installing Ollama does not automatically connect it to v4.0.0. The application must explicitly implement and invoke the Ollama API.

# 7. Test the Asynchronous Endpoint

The `/screen/async` endpoint accepts form fields including:

- `job_title`
- `job_description`
- `required_skills`
- `document_paths`
- `explain` (optional)
- `debug_text` (optional)

Example request:

```bash
curl -X POST http://127.0.0.1:8000/screen/async \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "job_title=Python Developer" \
  --data-urlencode "job_description=Python developer with API experience" \
  --data-urlencode "required_skills=Python, FastAPI, SQL" \
  --data-urlencode "document_paths=https://example.com/resume.pdf" \
  --data-urlencode "explain=true"
```

Replace the example URL with a real, accessible document URL supported by the application.

A successful request returns a job ID. Retrieve the status using:

```bash
curl http://127.0.0.1:8000/screen/jobs/YOUR_JOB_ID
```

Replace `YOUR_JOB_ID` with the ID returned by the API.

The job-status response provides the current processing state and, when available, the results or errors.

Use Swagger UI at `/docs` to inspect the precise request and response schemas.

# 8. Deploy on aaPanel

These instructions describe a typical Ubuntu deployment.

## Step 1: Prepare the project directory

Example production directory:

```text
/www/wwwroot/ats-screening-api
```

Navigate to it:

```bash
cd /www/wwwroot/ats-screening-api
```

If the repository has not been cloned yet, clone it into the intended location.

## Step 2: Install dependencies

Create a virtual environment if one does not already exist:

```bash
python3 -m venv venv
```

Install the dependencies:

```bash
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

Verify that the application imports:

```bash
./venv/bin/python -c "import app; print('Application import successful')"
```

## Step 3: Test before production

First confirm that port `8003` is available.

Start a temporary test instance:

```bash
./venv/bin/uvicorn app:app --host 127.0.0.1 --port 8003
```

In another terminal:

```bash
curl -sS http://127.0.0.1:8003/health
```

Stop the temporary instance with `Ctrl+C` after testing.

Do not use a port already occupied by another service.

## Step 4: Configure systemd

If you are creating a new service, create the service file:

```bash
sudo nano /etc/systemd/system/ats-screening-api.service
```

Example configuration:

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

Save the file.

For an existing installation, back up and adapt the current service configuration rather than creating a duplicate service. Configure an unprivileged service account and appropriate file permissions for production.

Reload systemd:

```bash
sudo systemctl daemon-reload
```

Enable the service at startup:

```bash
sudo systemctl enable ats-screening-api
```

Start the service:

```bash
sudo systemctl start ats-screening-api
```

Check the status:

```bash
sudo systemctl status ats-screening-api --no-pager -l
```

View the latest logs:

```bash
sudo journalctl -u ats-screening-api -n 100 --no-pager
```

Follow live logs:

```bash
sudo journalctl -u ats-screening-api -f
```

## Step 5: Configure the website in aaPanel

1. Open aaPanel.
2. Go to **Website** and add your API domain.
3. Configure the domain's DNS A record to point to the server's public IP.
4. Open the website's reverse proxy settings.
5. Configure the upstream address as:

   ```text
   http://127.0.0.1:8000
   ```

6. Preserve request paths and HTTP methods.
7. Configure SSL and enable HTTPS.
8. Test the domain after saving the configuration.

Example domain:

```text
https://ats-api.example.com
```

Replace this placeholder with your actual domain.

Uvicorn should remain bound to `127.0.0.1` when accessed through a reverse proxy on the same server.

# 9. Production Health Checks

Check the local API:

```bash
curl -sS http://127.0.0.1:8000/health
```

Check the public API:

```bash
curl -sS https://YOUR_API_DOMAIN/health
```

Verify the published API version:

```bash
curl -fsS https://YOUR_API_DOMAIN/openapi.json
```

Inspect the service status:

```bash
sudo systemctl status ats-screening-api --no-pager -l
```

Replace `YOUR_API_DOMAIN` with your configured domain.

# 10. Update the Application

Back up the current application and configuration before updating a production deployment.

Fetch the latest repository changes:

```bash
git pull
```

Install updated dependencies:

```bash
./venv/bin/pip install -r requirements.txt
```

Validate the application import:

```bash
./venv/bin/python -c "import app; print('Import successful')"
```

Restart the service only after the new version has been checked:

```bash
sudo systemctl restart ats-screening-api
```

Verify the status and health endpoint afterward.

For production updates, maintain a known-good release or backup so you can roll back if the new version fails.

# 11. Security Recommendations

- Never commit API keys, passwords, tokens, or private configuration.
- Do not commit the virtual environment or Python cache files.
- Do not publish candidate resumes or personally identifiable information.
- Use HTTPS for public traffic.
- Keep Uvicorn bound to localhost behind the reverse proxy.
- Keep Ollama bound to localhost unless secure remote access is specifically required.
- Restrict access to administrative endpoints and server logs.
- Validate document URLs and file types.
- Apply request-size and timeout limits appropriate to the application.
- Review uploaded documents and external URLs as untrusted input.
- Use human review for important hiring decisions.
- Do not rely solely on automated scores to accept or reject candidates.
- Test changes before restarting production services.

## Suggested `.gitignore`

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
.DS_Store
*.log
```

Do not commit private resumes, credentials, server backups, or production-only configuration files.

# 12. License

Choose a license before publishing the repository.

For example, the MIT License permits reuse, modification, and distribution subject to its terms. Only add an MIT license if you intend to release the project under that license.

# Developer Contact

**Sadam Qureshi**

- Email: pirsadam05@gmail.com
- Phone: 03035609341

For questions, bug reports, or technical discussions, please use the repository's GitHub Issues section.

---

**ATS Screening API v4.0.0 — Resume screening and candidate matching through a REST API.**
