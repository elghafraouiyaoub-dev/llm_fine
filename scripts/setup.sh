#!/bin/bash

# Movie Recommendation System Setup Script
# This script sets up the complete development environment

set -e  # Exit on any error

echo "🎬 Setting up Movie Recommendation System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.11+ first."
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    required_version="3.11"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_error "Python 3.11+ is required. Current version: $python_version"
        exit 1
    fi
    
    print_success "Python $python_version is installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."
    
    directories=(
        "data/raw"
        "data/processed"
        "data/external"
        "models/checkpoints"
        "models/artifacts"
        "logs"
        "notebooks"
        "tests/unit"
        "tests/integration"
        "config/grafana/dashboards"
        "config/grafana/provisioning/dashboards"
        "config/grafana/provisioning/datasources"
        "config/nginx"
        "scripts"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
    
    print_success "All directories created"
}

# Create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Environment
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/recommendations
POSTGRES_DB=recommendations
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Redis
REDIS_URL=redis://localhost:6379

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama2:7b

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API Keys (add your own)
TMDB_API_KEY=your_tmdb_api_key_here
WANDB_API_KEY=your_wandb_api_key_here

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=admin123
EOF
        print_success "Environment file created"
    else
        print_warning "Environment file already exists"
    fi
}

# Create configuration files
create_config_files() {
    print_status "Creating configuration files..."
    
    # Prometheus configuration
    cat > config/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'recommendation-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

    # Grafana datasource configuration
    mkdir -p config/grafana/provisioning/datasources
    cat > config/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Nginx configuration
    cat > config/nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://api_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        location /health {
            access_log off;
            proxy_pass http://api_backend/health;
        }
    }
}
EOF

    print_success "Configuration files created"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Download sample data
download_sample_data() {
    print_status "Downloading sample MovieLens data..."
    
    if [ ! -f "data/raw/ml-25m.zip" ]; then
        curl -o data/raw/ml-25m.zip "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
        cd data/raw
        unzip ml-25m.zip
        cd ../..
        print_success "MovieLens dataset downloaded and extracted"
    else
        print_warning "MovieLens dataset already exists"
    fi
}

# Initialize database
init_database() {
    print_status "Initializing database..."
    
    cat > scripts/init_db.sql << EOF
-- Create databases
CREATE DATABASE IF NOT EXISTS recommendations;
CREATE DATABASE IF NOT EXISTS mlflow;

-- Create extensions
\c recommendations;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create basic tables
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS movies (
    id SERIAL PRIMARY KEY,
    movie_id INTEGER UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    genres TEXT,
    year INTEGER,
    imdb_id VARCHAR(20),
    tmdb_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ratings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    rating FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    score FLOAT NOT NULL,
    algorithm VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_movie_id ON ratings(movie_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations(user_id);
CREATE INDEX IF NOT EXISTS idx_movies_movie_id ON movies(movie_id);
EOF

    print_success "Database initialization script created"
}

# Setup Ollama models
setup_ollama() {
    print_status "Setting up Ollama models..."
    
    # This will be done after containers are running
    cat > scripts/setup_ollama.sh << EOF
#!/bin/bash
echo "Pulling Ollama models..."
docker exec recommendation_ollama ollama pull llama2:7b
docker exec recommendation_ollama ollama pull mistral:7b
echo "Ollama models ready"
EOF
    
    chmod +x scripts/setup_ollama.sh
    print_success "Ollama setup script created"
}

# Create test scripts
create_test_scripts() {
    print_status "Creating test scripts..."
    
    cat > scripts/test_system.sh << EOF
#!/bin/bash
echo "🧪 Testing Movie Recommendation System..."

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Test API health
echo "Testing API health..."
curl -f http://localhost:8000/health || exit 1

# Test database connection
echo "Testing database connection..."
curl -f http://localhost:8000/ready || exit 1

# Test Ollama
echo "Testing Ollama..."
curl -f http://localhost:11434/api/tags || exit 1

# Test MLflow
echo "Testing MLflow..."
curl -f http://localhost:5000 || exit 1

echo "✅ All services are healthy!"
EOF

    chmod +x scripts/test_system.sh
    print_success "Test scripts created"
}

# Main setup function
main() {
    print_status "Starting Movie Recommendation System setup..."
    
    # Check prerequisites
    check_docker
    check_python
    
    # Setup project structure
    create_directories
    create_env_file
    create_config_files
    init_database
    setup_ollama
    create_test_scripts
    
    # Install dependencies (optional for Docker-only setup)
    read -p "Do you want to install Python dependencies locally? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_python_deps
    fi
    
    # Download data (optional)
    read -p "Do you want to download the MovieLens dataset? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_sample_data
    fi
    
    print_success "Setup completed successfully!"
    
    echo
    echo "🚀 Next steps:"
    echo "1. Start the system: docker-compose up -d"
    echo "2. Setup Ollama models: ./scripts/setup_ollama.sh"
    echo "3. Test the system: ./scripts/test_system.sh"
    echo "4. Access the API docs: http://localhost:8000/docs"
    echo "5. Access Grafana: http://localhost:3000 (admin/admin123)"
    echo "6. Access MLflow: http://localhost:5000"
    echo "7. Access Jupyter: http://localhost:8888"
    echo
    echo "📚 Check the docs/ folder for detailed documentation"
}

# Run main function
main "$@"
