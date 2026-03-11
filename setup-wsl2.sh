#!/bin/bash
# QLoRA Setup Script for Windows WSL2
# Script ini menginstal semua dependencies dan mengkonfigurasi aplikasi

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running in WSL2
check_wsl2() {
    log_step "Checking WSL2 environment..."
    
    if ! grep -q Microsoft /proc/version; then
        log_error "This script is designed for WSL2. Use setup-linux.sh for native Linux."
        exit 1
    fi
    
    if ! grep -q 2 /proc/version; then
        log_warn "WSL1 detected. Some features may not work properly."
    fi
    
    log_info "WSL2 environment detected."
}

# Update system packages
update_system() {
    log_step "Updating system packages..."
    
    sudo apt update
    sudo apt upgrade -y
    
    log_info "System packages updated."
}

# Install essential packages
install_essentials() {
    log_step "Installing essential packages..."
    
    sudo apt install -y \
        curl \
        wget \
        git \
        unzip \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common \
        build-essential \
        python3 \
        python3-pip \
        python3-venv
    
    log_info "Essential packages installed."
}

# Install Docker
install_docker() {
    log_step "Installing Docker..."
    
    # Remove old versions
    sudo apt-get remove -y docker docker-engine docker.io containerd runc || true
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Set up the repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    log_info "Docker installed. You may need to restart WSL2 for group changes to take effect."
}

# Configure Docker for WSL2
configure_docker_wsl2() {
    log_step "Configuring Docker for WSL2..."
    
    # Create docker daemon config
    sudo mkdir -p /etc/docker
    sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "hosts": ["unix:///var/run/docker.sock", "tcp://0.0.0.0:2375"],
  "iptables": false,
  "ip-forward": false,
  "bridge": "none",
  "storage-driver": "overlay2"
}
EOF
    
    # Start Docker service
    sudo systemctl enable docker
    sudo systemctl start docker
    
    log_info "Docker configured for WSL2."
}

# Check GPU availability
check_gpu() {
    log_step "Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        GPU_AVAILABLE=true
    else
        log_warn "No NVIDIA GPU detected. Training will run in simulation mode."
        GPU_AVAILABLE=false
    fi
}

# Install NVIDIA Container Toolkit (if GPU available)
install_nvidia_docker() {
    if [ "$GPU_AVAILABLE" = true ]; then
        log_step "Installing NVIDIA Container Toolkit..."
        
        # Add NVIDIA package repositories
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        # Update and install
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        
        # Configure Docker runtime
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        
        log_info "NVIDIA Container Toolkit installed."
    fi
}

# Create necessary directories
create_directories() {
    log_step "Creating necessary directories..."
    
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/models"
    mkdir -p "$PROJECT_DIR/checkpoints"
    mkdir -p "$PROJECT_DIR/data"
    mkdir -p "$PROJECT_DIR/docker/mongodb"
    mkdir -p "$PROJECT_DIR/docker/prometheus"
    mkdir -p "$PROJECT_DIR/docker/grafana/provisioning/dashboards"
    mkdir -p "$PROJECT_DIR/docker/grafana/provisioning/datasources"
    
    log_info "Directories created."
}

# Generate configuration files
generate_configs() {
    log_step "Generating configuration files..."
    
    # MongoDB init script
    cat > "$PROJECT_DIR/docker/mongodb/init.js" << 'EOF'
db = db.getSiblingDB('admin');
db.createUser({
  user: 'qlora_user',
  pwd: 'qlora_password',
  roles: [
    { role: 'readWrite', db: 'qlora_db' },
    { role: 'dbAdmin', db: 'qlora_db' }
  ]
});
EOF
    
    # Prometheus config
    cat > "$PROJECT_DIR/docker/prometheus/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'qlora-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
EOF
    
    # Grafana datasource config
    cat > "$PROJECT_DIR/docker/grafana/provisioning/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    
    log_info "Configuration files generated."
}

# Setup environment file
setup_environment() {
    log_step "Setting up environment file..."
    
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        log_warn ".env file not found. Creating from template..."
        
        cat > "$PROJECT_DIR/.env" << 'EOF'
# QLoRA Application Environment Configuration - WSL2
# Database Configuration
DATABASE_URL=mongodb://admin:password@localhost:27017/qlora_db?authSource=admin
MONGO_USERNAME=admin
MONGO_PASSWORD=your-secure-mongodb-password
MONGO_DATABASE=qlora_db

# Redis Configuration  
REDIS_URL=redis://:your-secure-redis-password@localhost:6379/0
REDIS_PASSWORD=your-secure-redis-password

# Security Configuration
SECRET_KEY=your-32-character-secret-key-here-change-this-in-production
JWT_SECRET_KEY=your-32-character-jwt-secret-key-here-change-this-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
ENABLE_FILE_LOGGING=true
LOG_FILE_PATH=logs/app.log

# GPU Configuration
GPU_MEMORY_THRESHOLD=0.9
ENABLE_GPU_MONITORING=true

# Performance Configuration
MAX_WORKERS=4
ENABLE_PERFORMANCE_MONITORING=true

# Security Configuration
ENABLE_SECURITY_AUDIT=true

# ML Platform Configuration (Optional)
WANDB_API_KEY=your-wandb-api-key
HUGGINGFACE_TOKEN=your-huggingface-token

# Grafana Configuration (Optional)
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
EOF
        
        log_warn "Please edit .env file with your secure passwords and keys before proceeding."
        log_info "Template .env file created."
    else
        log_info ".env file already exists."
    fi
}

# Build and start services
deploy_application() {
    log_step "Building and starting application..."
    
    cd "$PROJECT_DIR"
    
    # Validate environment
    if command -v python3 &> /dev/null; then
        log_info "Validating environment configuration..."
        python3 backend/core/environment_validator.py --env-file .env || {
            log_error "Environment validation failed. Please fix .env file."
            exit 1
        }
    fi
    
    # Build and start services
    if [ "$GPU_AVAILABLE" = true ]; then
        log_info "Starting with GPU support..."
        docker-compose -f docker-compose.yml up -d
    else
        log_info "Starting without GPU support..."
        docker-compose -f docker-compose.yml up -d mongodb redis backend frontend
    fi
    
    log_info "Application deployed successfully."
}

# Wait for services and check health
check_deployment() {
    log_step "Checking deployment health..."
    
    cd "$PROJECT_DIR"
    
    # Wait for services to start
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check service health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "✅ Backend is healthy"
    else
        log_warn "⚠️ Backend not responding yet"
    fi
    
    if curl -f http://localhost > /dev/null 2>&1; then
        log_info "✅ Frontend is healthy"
    else
        log_warn "⚠️ Frontend not responding yet"
    fi
    
    # Show service status
    docker-compose ps
}

# Show access information
show_access_info() {
    log_step "Access Information:"
    echo ""
    echo "🌐 Frontend: http://localhost"
    echo "🔧 Backend API: http://localhost:8000"
    echo "📊 Grafana: http://localhost:3000 (admin/admin)"
    echo "📈 Prometheus: http://localhost:9090"
    echo "🗄️ MongoDB: localhost:27017"
    echo "🗃️ Redis: localhost:6379"
    echo ""
    echo "📋 Useful Commands:"
    echo "  docker-compose logs -f          # Show all logs"
    echo "  docker-compose logs backend     # Show backend logs"
    echo "  docker-compose ps               # Show service status"
    echo "  docker-compose down             # Stop all services"
    echo ""
}

# Main execution
main() {
    echo "🚀 QLoRA Setup for Windows WSL2"
    echo "================================"
    echo ""
    
    check_wsl2
    update_system
    install_essentials
    install_docker
    configure_docker_wsl2
    check_gpu
    install_nvidia_docker
    create_directories
    generate_configs
    setup_environment
    
    echo ""
    log_info "Setup completed successfully!"
    echo ""
    
    # Ask if user wants to deploy now
    read -p "Do you want to deploy the application now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        deploy_application
        check_deployment
        show_access_info
    else
        log_info "Setup completed. Run './deploy.sh setup' to deploy the application later."
    fi
}

# Run main function
main "$@"
