#!/bin/bash
# QLoRA Setup Script for Native Linux
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

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    else
        log_error "Cannot detect Linux distribution"
        exit 1
    fi
}

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

# Check if running as root for system operations
check_permissions() {
    if [ "$EUID" -eq 0 ]; then
        log_error "This script should not be run as root. Run as regular user."
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log_step "Checking system requirements..."
    
    # Check RAM
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_RAM" -lt 16 ]; then
        log_warn "System has less than 16GB RAM (${TOTAL_RAM}GB detected). Performance may be limited."
    else
        log_info "System RAM: ${TOTAL_RAM}GB ✓"
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 100 ]; then
        log_warn "Less than 100GB disk space available (${AVAILABLE_SPACE}GB detected)."
    else
        log_info "Available disk space: ${AVAILABLE_SPACE}GB ✓"
    fi
    
    # Check if running in WSL (should use setup-wsl2.sh instead)
    if grep -q Microsoft /proc/version 2>/dev/null; then
        log_error "WSL detected. Please use setup-wsl2.sh instead."
        exit 1
    fi
}

# Update system packages
update_system() {
    log_step "Updating system packages..."
    
    case $DISTRO in
        ubuntu|debian)
            sudo apt update
            sudo apt upgrade -y
            ;;
        fedora|centos|rhel)
            if command -v dnf &> /dev/null; then
                sudo dnf update -y
            else
                sudo yum update -y
            fi
            ;;
        arch)
            sudo pacman -Syu --noconfirm
            ;;
        *)
            log_error "Unsupported distribution: $DISTRO"
            exit 1
            ;;
    esac
    
    log_info "System packages updated."
}

# Install essential packages
install_essentials() {
    log_step "Installing essential packages..."
    
    case $DISTRO in
        ubuntu|debian)
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
                python3-venv \
                htop \
                iotop \
                net-tools
            ;;
        fedora|centos|rhel)
            if command -v dnf &> /dev/null; then
                sudo dnf groupinstall -y "Development Tools"
                sudo dnf install -y \
                    curl \
                    wget \
                    git \
                    unzip \
                    python3 \
                    python3-pip \
                    htop \
                    iotop \
                    net-tools
            else
                sudo yum groupinstall -y "Development Tools"
                sudo yum install -y \
                    curl \
                    wget \
                    git \
                    unzip \
                    python3 \
                    python3-pip \
                    htop \
                    iotop \
                    net-tools
            fi
            ;;
        arch)
            sudo pacman -S --noconfirm \
                base-devel \
                curl \
                wget \
                git \
                unzip \
                python \
                python-pip \
                htop \
                iotop \
                net-tools
            ;;
    esac
    
    log_info "Essential packages installed."
}

# Install Docker
install_docker() {
    log_step "Installing Docker..."
    
    # Remove old versions
    case $DISTRO in
        ubuntu|debian)
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
            ;;
        fedora|centos|rhel)
            sudo dnf remove -y docker docker-client docker-client-latest docker-common docker-latest docker-latest-logrotate docker-logrotate docker-engine || true
            
            # Add Docker repository
            sudo dnf config-manager --add-repo=https://download.docker.com/linux/fedora/docker-ce.repo
            
            # Install Docker Engine
            sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            ;;
        arch)
            sudo pacman -R --noconfirm docker || true
            sudo pacman -S --noconfirm docker docker-compose
            ;;
    esac
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Start and enable Docker service
    sudo systemctl enable docker
    sudo systemctl start docker
    
    log_info "Docker installed and started. You may need to log out and log back in for group changes to take effect."
}

# Check GPU availability
check_gpu() {
    log_step "Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
        GPU_AVAILABLE=true
        
        # Check NVIDIA driver version
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
        log_info "NVIDIA Driver Version: $DRIVER_VERSION"
        
        # Check CUDA toolkit
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            log_info "CUDA Version: $CUDA_VERSION"
        else
            log_warn "CUDA toolkit not found. GPU acceleration may be limited."
        fi
    else
        log_warn "No NVIDIA GPU detected. Training will run in simulation mode."
        GPU_AVAILABLE=false
    fi
}

# Install NVIDIA Container Toolkit (if GPU available)
install_nvidia_docker() {
    if [ "$GPU_AVAILABLE" = true ]; then
        log_step "Installing NVIDIA Container Toolkit..."
        
        case $DISTRO in
            ubuntu|debian)
                # Add NVIDIA package repositories
                curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
                curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
                
                sudo apt-get update
                sudo apt-get install -y nvidia-container-toolkit
                ;;
            fedora|centos|rhel)
                curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
                    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
                
                if command -v dnf &> /dev/null; then
                    sudo dnf install -y nvidia-container-toolkit
                else
                    sudo yum install -y nvidia-container-toolkit
                fi
                ;;
            arch)
                sudo pacman -S --noconfirm nvidia-container-toolkit
                ;;
        esac
        
        # Configure Docker runtime
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        
        log_info "NVIDIA Container Toolkit installed."
    fi
}

# Configure system optimizations
configure_system() {
    log_step "Configuring system optimizations..."
    
    # Increase file descriptor limits
    echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
    echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
    
    # Configure sysctl for better performance
    sudo tee -a /etc/sysctl.conf > /dev/null <<EOF

# QLoRA Performance Optimizations
vm.max_map_count = 262144
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
fs.file-max = 2097152
EOF
    
    # Apply sysctl changes
    sudo sysctl -p
    
    log_info "System optimizations configured."
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
    
    # Set proper permissions
    chmod 755 "$PROJECT_DIR/logs"
    chmod 755 "$PROJECT_DIR/models"
    chmod 755 "$PROJECT_DIR/checkpoints"
    chmod 755 "$PROJECT_DIR/data"
    
    log_info "Directories created with proper permissions."
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

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
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
# QLoRA Application Environment Configuration - Linux Native
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
ENVIRONMENT=production
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
    echo "  htop                            # Monitor system resources"
    echo "  nvidia-smi                      # Monitor GPU usage"
    echo ""
}

# Main execution
main() {
    echo "🚀 QLoRA Setup for Native Linux"
    echo "==============================="
    echo ""
    
    detect_distro
    check_permissions
    check_requirements
    update_system
    install_essentials
    install_docker
    check_gpu
    install_nvidia_docker
    configure_system
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
