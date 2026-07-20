FROM python:3.10-slim

WORKDIR /app

# system deps for pillow + basic build
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU wheels first (faster/more deterministic)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Project deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY src /app/src

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]