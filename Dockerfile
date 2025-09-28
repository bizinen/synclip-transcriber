FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_PORT=3000 \
    HF_HOME=/app/.cache/huggingface \
    WHISPER_PRELOAD_MODEL=tiny

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3000

CMD ["streamlit", "run", "ui.py", "--server.port=3000", "--server.address=0.0.0.0"]

