"""
Modal deployment for NammaKelsa ML Service.

Commands:
  modal serve ml_service/modal_app.py   # local dev with hot-reload
  modal deploy ml_service/modal_app.py  # production deploy

Secrets: create a secret group named "ml-service-env" in the Modal dashboard
with all vars from ml_service/.env.example.
"""
import modal

APP_NAME = "nammakelsa-ml-service"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "build-essential",
        "python3-dev",
        "libglib2.0-0",
        "libgl1",
        "libgomp1",
        "ffmpeg",
        "tesseract-ocr",
        "tesseract-ocr-eng",
        "libsm6",
        "libxext6",
    ])
    .copy_local_file("ml_service/requirements.txt", "/app/requirements.txt")
    .pip_install_from_requirements("/app/requirements.txt")
    .copy_local_dir("ml_service", "/app")
    .workdir("/app")
)

# Persistent volume — DeepFace downloads ArcFace + Facenet weights (~350 MB)
# on first run. This volume caches them so redeploys don't re-download.
model_cache = modal.Volume.from_name("deepface-weights", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={"/root/.deepface": model_cache},
    memory=4096,       # 4 GB — needed for TF + PyTorch + models in memory
    cpu=2.0,
    timeout=600,       # 10 min max per request (long video processing)
    keep_warm=1,       # keep 1 container alive for the Kafka consumer loop
    secrets=[modal.Secret.from_name("ml-service-env")],
)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/app")
    from app.main import app as fastapi_app
    return fastapi_app
