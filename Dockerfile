
# 1. Use a slim Python base image
FROM python:3.12-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Install Tesseract OCR and system dependencies
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libtesseract-dev \
    libleptonica-dev \
    build-essential \
    libsqlite3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory
WORKDIR /app

# 5. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the application code
COPY . .

# 7. Expose the port
EXPOSE 5056

# 8. Define the command to run the application
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "5056"] 