FROM python:3.11-slim

# System deps for OpenCV, pytesseract, pydub, mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download DeepFace model weights during build (optional — speeds up startup)
# RUN python -c "from deepface import DeepFace; import numpy as np; DeepFace.represent(np.zeros((160,160,3),dtype=np.uint8), model_name='Facenet', enforce_detection=False)"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
