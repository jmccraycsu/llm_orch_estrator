FROM python:3.12-slim

WORKDIR /app

# Only install what pip needs for C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY llm_orchestrator ./llm_orchestrator

RUN useradd --create-home appuser
USER appuser

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "llm_orchestrator.api:app", "--host", "0.0.0.0", "--port", "8000"]
