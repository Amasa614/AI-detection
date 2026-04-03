FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface \
    BINO_PRELOAD_MODELS=true

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY daniela_bino_backend.py ./daniela_bino_backend.py
COPY hf-start.sh ./hf-start.sh

RUN chmod +x /app/hf-start.sh && mkdir -p /data/huggingface

EXPOSE 7860

CMD ["/app/hf-start.sh"]
