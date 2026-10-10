FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/hf-cache/sentence-transformers \
    DEVICE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY bot /app/bot
COPY indexer /app/indexer

ENV PYTHONPATH=/app/bot

CMD ["uvicorn", "api:app", "--app-dir", "/app/bot", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
