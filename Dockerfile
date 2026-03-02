# Use slim Python image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /

# Install build deps for some python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements if present
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Copy source code
COPY /app /app
WORKDIR /app

# Default port; Render provides $PORT env var at runtime
EXPOSE 8000

# Start command (expects FastAPI app at api.main:app). Uses $PORT if set.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# docker build -t my-fastapi-app .
# docker run -p 8000:8000 my-fastapi-app