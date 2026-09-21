FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLAGS_use_cuda=0
ENV FLAGS_use_mkldnn=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cron \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /app/storage

COPY docker/ocr-recovery-cron /etc/cron.d/ocr-recovery-cron

RUN sed -i 's/\r$//' /etc/cron.d/ocr-recovery-cron \
    && chmod 0644 /etc/cron.d/ocr-recovery-cron \
    && printf '\n' >> /etc/cron.d/ocr-recovery-cron

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]