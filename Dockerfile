FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies

RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin libtesseract-dev build-essential && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for Docker layer caching

COPY requirements.txt .

# Install Python dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project

COPY . .

# Pre-download FastEmbed sparse model

RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25', cache_dir='./models')"

# Expose Render port

EXPOSE 10000

# Start FastAPI app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]
