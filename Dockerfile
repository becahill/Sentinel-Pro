FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV SENTINEL_DISABLE_TOXICITY=1
ENV SENTINEL_REDACT_PII=1

EXPOSE 8000 8501
