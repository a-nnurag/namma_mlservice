# NammaKelsa — ML Service

AI-powered candidate evaluation service for the NammaKelsa skill verification platform. Receives video/audio streams via Kafka, assembles them, and runs a parallel Celery pipeline for fraud detection, interview scoring, work experience verification, and degree validation.

## Tech Stack

- **Python 3.11** + **FastAPI** (async)
- **Celery** with chord pattern (parallel tasks → verdict callback)
- **Redis DB 0** (Celery broker + result backend)
- **Redis DB 1** (Kafka chunk buffer)
- **PostgreSQL 16** with **pgvector** (face/voice embeddings for duplicate detection)
- **Kafka / Redpanda** (receives chunked video/audio from backend)
- **DeepFace** (ArcFace 512-d face embeddings)
- **Resemblyzer** (speaker 256-d voice embeddings)
- **MediaPipe** (liveness detection — blink, head movement)
- **Tesseract OCR** (degree certificate verification)

## Pipeline

```
Kafka chunks → Redis buffer → Assemble media files
    ↓
Celery chord:
  ┌─ fraud_task    (liveness + duplicate face/voice check)
  ├─ interview_task (ASR transcription + domain scoring)
  └─ workexp_task  (claimed role verification from audio)
    ↓ (all three complete)
  verdict_task  (composite score + PASS/MANUAL_REVIEW/REJECT)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/status/{candidate_id}` | Pipeline status (polled by backend every 30s) |
| `GET` | `/verdict/{candidate_id}` | Final verdict + composite score |
| `POST` | `/degree/{candidate_id}` | Degree certificate OCR verification |
| `POST` | `/face-check` | ArcFace uniqueness check (called at registration) |
| `GET` | `/admin/runs` | List all pipeline runs |
| `GET` | `/admin/run/{run_id}` | Single run details |
| `DELETE` | `/admin/run/{run_id}/cancel` | Cancel a run |
| `POST` | `/admin/trigger/{candidate_id}` | Re-trigger pipeline for a candidate |

## Related Repositories

| Service | Repository |
|---------|-----------|
| Backend API (FastAPI) | https://github.com/a-nnurag/namma_backend |
| Frontend (Next.js) | https://github.com/a-nnurag/namma_frontend |
| Infrastructure (Docker + DB init) | https://github.com/a-nnurag/namma_infra |

## Local Development

```bash
# 1. Spin up dependencies
cd ../  # go to infra root
docker compose up postgres-ml redis redpanda -d

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your values

# 5. Start Celery worker (separate terminal)
celery -A app.tasks.celery_app.celery_app worker --loglevel=info --concurrency=2

# 6. Run API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Redis DB Allocation

| DB | Used by | Purpose |
|----|---------|---------|
| 0 | Celery | Task broker + result backend |
| 1 | ML Service | Kafka chunk buffer |
| 2 | Backend | OTP, sessions, caches (separate service) |
