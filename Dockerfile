FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config ./config
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "churnxgb.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
