@echo off
cd /d "E:\My Data\Project\AI-ATS-Screener"
python -m uvicorn app-2:app --reload --host 0.0.0.0 --port 8000
