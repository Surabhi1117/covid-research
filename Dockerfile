FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose ports (FastAPI on 8000, Streamlit on 8501)
EXPOSE 8000 8501

# We will use docker-compose to override the command for different services
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
