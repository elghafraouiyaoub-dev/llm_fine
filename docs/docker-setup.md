# Docker Setup

## Overview

This document outlines the containerization strategy for our LLM-based movie recommendation system, including multi-stage Docker builds, Docker Compose configurations, and testing procedures for local development and production deployment.

## Container Architecture

### Service Architecture
```mermaid
graph TB
    A[Load Balancer/Nginx] --> B[API Gateway]
    B --> C[Recommendation API]
    B --> D[Model Inference Service]
    
    C --> E[PostgreSQL]
    C --> F[Redis Cache]
    D --> G[Ollama Service]
    
    H[MLflow] --> I[Model Registry]
    J[Prometheus] --> K[Grafana]
    L[Jupyter Notebooks] --> M[Development]
```

## Docker Images

### 1. Base Image
```dockerfile
# docker/Dockerfile.base
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser
```

### 2. API Service
```dockerfile
# docker/Dockerfile.api
FROM base as api

# Install API-specific dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy API code
COPY src/api/ ./src/api/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Model Training Service
```dockerfile
# docker/Dockerfile.training
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as training

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3.11 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy and install requirements
COPY requirements-training.txt .
RUN pip install --no-cache-dir -r requirements-training.txt

# Copy training code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Create directories for data and models
RUN mkdir -p /app/data /app/models

# Default command
CMD ["python", "scripts/train_model.py"]
```

### 4. Ollama Service
```dockerfile
# docker/Dockerfile.ollama
FROM ollama/ollama:latest

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0
ENV OLLAMA_PORT=11434

# Create directory for models
RUN mkdir -p /root/.ollama

# Copy custom models (if any)
COPY models/ /root/.ollama/models/

# Expose port
EXPOSE 11434

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:11434/api/tags || exit 1

# Start Ollama service
CMD ["ollama", "serve"]
```

### 5. Monitoring Stack
```dockerfile
# docker/Dockerfile.monitoring
FROM prom/prometheus:latest as prometheus

COPY config/prometheus.yml /etc/prometheus/prometheus.yml

---

FROM grafana/grafana:latest as grafana

# Copy custom dashboards
COPY config/grafana/dashboards/ /var/lib/grafana/dashboards/
COPY config/grafana/provisioning/ /etc/grafana/provisioning/

# Set environment variables
ENV GF_SECURITY_ADMIN_PASSWORD=admin123
ENV GF_USERS_ALLOW_SIGN_UP=false
```

## Docker Compose Configurations

### Development Environment
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  # Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: recommendations
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init_db.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Ollama Service
  ollama:
    build:
      context: .
      dockerfile: docker/Dockerfile.ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # API Service
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/recommendations
      - REDIS_URL=redis://redis:6379
      - OLLAMA_URL=http://ollama:11434
      - ENVIRONMENT=development
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      ollama:
        condition: service_healthy
    volumes:
      - ./src:/app/src
      - ./config:/app/config
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

  # MLflow
  mlflow:
    image: python:3.11-slim
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://postgres:postgres@postgres:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow_data:/mlflow
    depends_on:
      postgres:
        condition: service_healthy
    command: >
      bash -c "
        pip install mlflow psycopg2-binary &&
        mlflow server 
          --backend-store-uri postgresql://postgres:postgres@postgres:5432/mlflow
          --default-artifact-root /mlflow/artifacts
          --host 0.0.0.0
          --port 5000
      "

  # Jupyter Notebook
  jupyter:
    build:
      context: .
      dockerfile: docker/Dockerfile.jupyter
    ports:
      - "8888:8888"
    environment:
      - JUPYTER_ENABLE_LAB=yes
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./src:/home/jovyan/work/src
      - ./data:/home/jovyan/work/data
    command: start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  ollama_data:
  mlflow_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: recommendation_network
```

### Production Environment
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./config/nginx/ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

  # API Service (Multiple replicas)
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - OLLAMA_URL=${OLLAMA_URL}
      - ENVIRONMENT=production
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Ollama Service
  ollama:
    build:
      context: .
      dockerfile: docker/Dockerfile.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    volumes:
      - ollama_data:/root/.ollama

  # External services (managed separately in production)
  # postgres, redis, monitoring stack would typically be managed
  # by cloud providers or separate infrastructure

volumes:
  ollama_data:
    external: true

networks:
  default:
    external:
      name: production_network
```

## Multi-Stage Build Optimization

### Optimized API Dockerfile
```dockerfile
# docker/Dockerfile.api.optimized
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt requirements-api.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt -r requirements-api.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Set PATH to include user packages
ENV PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Testing Procedures

### Container Testing Script
```bash
#!/bin/bash
# scripts/test_containers.sh

set -e

echo "🧪 Starting container testing..."

# Function to wait for service to be ready
wait_for_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1

    echo "⏳ Waiting for $service to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "✅ $service is ready!"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts failed, retrying in 10s..."
        sleep 10
        ((attempt++))
    done
    
    echo "❌ $service failed to start after $max_attempts attempts"
    return 1
}

# Start services
echo "🚀 Starting services..."
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to be ready
wait_for_service "PostgreSQL" "http://localhost:5432"
wait_for_service "Redis" "http://localhost:6379"
wait_for_service "API" "http://localhost:8000/health"
wait_for_service "Ollama" "http://localhost:11434/api/tags"
wait_for_service "MLflow" "http://localhost:5000"

# Run integration tests
echo "🔍 Running integration tests..."
docker-compose -f docker-compose.dev.yml exec -T api python -m pytest tests/integration/ -v

# Run API tests
echo "🌐 Running API tests..."
python scripts/test_api_endpoints.py

# Test model inference
echo "🤖 Testing model inference..."
python scripts/test_model_inference.py

# Performance tests
echo "⚡ Running performance tests..."
python scripts/performance_tests.py --duration=60

echo "✅ All tests passed!"

# Cleanup
echo "🧹 Cleaning up..."
docker-compose -f docker-compose.dev.yml down -v
```

### API Testing Script
```python
# scripts/test_api_endpoints.py
import requests
import json
import time
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_endpoint(self) -> bool:
        """Test health check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def test_recommendation_endpoint(self) -> bool:
        """Test recommendation endpoint."""
        payload = {
            "user_id": 1,
            "num_recommendations": 5
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/recommend",
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                return False
            
            data = response.json()
            return len(data.get('movies', [])) == 5
            
        except requests.RequestException:
            return False
    
    def test_user_profile_endpoint(self) -> bool:
        """Test user profile endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/users/1/profile")
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all API tests."""
        tests = {
            'health': self.test_health_endpoint,
            'recommendations': self.test_recommendation_endpoint,
            'user_profile': self.test_user_profile_endpoint
        }
        
        results = {}
        for test_name, test_func in tests.items():
            print(f"Running {test_name} test...")
            results[test_name] = test_func()
            print(f"✅ {test_name}: {'PASSED' if results[test_name] else 'FAILED'}")
        
        return results

if __name__ == "__main__":
    tester = APITester()
    results = tester.run_all_tests()
    
    if all(results.values()):
        print("🎉 All API tests passed!")
        exit(0)
    else:
        print("❌ Some tests failed!")
        exit(1)
```

## Container Security

### Security Best Practices
```dockerfile
# Security-hardened Dockerfile example
FROM python:3.11-slim

# Update packages and remove package manager
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove

# Create non-root user with specific UID/GID
RUN groupadd -g 1001 appuser && useradd -r -u 1001 -g appuser appuser

# Set secure file permissions
COPY --chown=appuser:appuser --chmod=755 src/ /app/src/

# Use non-root user
USER appuser

# Remove unnecessary packages and files
RUN find /app -name "*.pyc" -delete \
    && find /app -name "__pycache__" -delete

# Set security headers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random
```

## Performance Optimization

### Build Cache Optimization
```dockerfile
# Optimized layer caching
FROM python:3.11-slim

# Install system dependencies (changes rarely)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first (changes less frequently than code)
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy configuration (changes less frequently than code)
COPY config/ ./config/

# Copy source code last (changes most frequently)
COPY src/ ./src/
```

## Next Steps

1. **Kubernetes Migration**: Prepare Kubernetes manifests for production
2. **Image Scanning**: Implement automated security scanning
3. **Multi-Architecture**: Support ARM64 for Apple Silicon
4. **Registry Management**: Set up private container registry
5. **Resource Optimization**: Fine-tune resource limits and requests

This Docker setup provides a robust, scalable, and secure containerization strategy for our recommendation system.
