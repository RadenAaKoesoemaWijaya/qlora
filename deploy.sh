#!/bin/bash
# Deployment script untuk QLoRA Application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

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

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Python (for environment validation)
    if ! command -v python3 &> /dev/null; then
        log_warn "Python3 is not installed. Environment validation will be skipped."
    fi
    
    log_info "Dependencies check completed."
}

validate_environment() {
    log_info "Validating environment configuration..."
    
    if [ ! -f "$ENV_FILE" ]; then
        log_error "Environment file $ENV_FILE not found. Please create it first."
        exit 1
    fi
    
    # Run Python environment validator if available
    if command -v python3 &> /dev/null; then
        cd "$PROJECT_DIR"
        if python3 backend/core/environment_validator.py --env-file "$ENV_FILE"; then
            log_info "Environment validation passed."
        else
            log_error "Environment validation failed. Please fix the issues above."
            exit 1
        fi
    else
        log_warn "Skipping environment validation (Python not available)."
    fi
}

create_directories() {
    log_info "Creating necessary directories..."
    
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

generate_config_files() {
    log_info "Generating configuration files..."
    
    # Generate MongoDB init script
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
    
    # Generate Prometheus configuration
    cat > "$PROJECT_DIR/docker/prometheus/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'qlora-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'qlora-mongodb'
    static_configs:
      - targets: ['mongodb:27017']
    metrics_path: '/metrics'
EOF
    
    # Generate Grafana datasource configuration
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

build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_DIR"
    
    # Build backend image
    docker-compose build backend
    
    # Build frontend image
    docker-compose build frontend
    
    log_info "Docker images built successfully."
}

deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_DIR"
    
    # Start core services
    docker-compose up -d mongodb redis
    
    log_info "Waiting for database services to be ready..."
    sleep 30
    
    # Start backend and frontend
    docker-compose up -d backend frontend
    
    log_info "Services deployed successfully."
}

deploy_with_monitoring() {
    log_info "Deploying with monitoring stack..."
    
    cd "$PROJECT_DIR"
    
    # Start all services including monitoring
    docker-compose --profile monitoring up -d
    
    log_info "Monitoring stack deployed successfully."
    log_info "Grafana available at: http://localhost:3000 (admin/admin)"
    log_info "Prometheus available at: http://localhost:9090"
}

deploy_fullstack() {
    log_info "Deploying full-stack application..."
    
    cd "$PROJECT_DIR"
    
    # Start all services
    docker-compose --profile fullstack up -d
    
    log_info "Full-stack application deployed successfully."
    log_info "Application available at: http://localhost:8080"
    log_info "API available at: http://localhost:8081"
}

check_health() {
    log_info "Checking service health..."
    
    # Check backend health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "✅ Backend service is healthy"
    else
        log_error "❌ Backend service is not responding"
    fi
    
    # Check frontend health
    if curl -f http://localhost > /dev/null 2>&1; then
        log_info "✅ Frontend service is healthy"
    else
        log_error "❌ Frontend service is not responding"
    fi
    
    # Check database connectivity
    if docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
        log_info "✅ MongoDB is healthy"
    else
        log_error "❌ MongoDB is not responding"
    fi
    
    if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
        log_info "✅ Redis is healthy"
    else
        log_error "❌ Redis is not responding"
    fi
}

show_status() {
    log_info "Service Status:"
    docker-compose ps
    
    log_info "Resource Usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

stop_services() {
    log_info "Stopping services..."
    
    cd "$PROJECT_DIR"
    docker-compose down
    
    log_info "Services stopped."
}

show_logs() {
    log_info "Showing logs for: $1"
    
    cd "$PROJECT_DIR"
    
    if [ -n "$1" ]; then
        docker-compose logs -f "$1"
    else
        docker-compose logs -f
    fi
}

show_help() {
    echo "QLoRA Deployment Script"
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy          Deploy core services (backend, frontend, database)"
    echo "  deploy-monitoring Deploy with monitoring stack (Prometheus, Grafana)"
    echo "  deploy-fullstack Deploy full-stack application"
    echo "  build           Build Docker images"
    echo "  health          Check service health"
    echo "  status          Show service status and resource usage"
    echo "  logs [service]  Show logs (optionally for specific service)"
    echo "  stop            Stop all services"
    echo "  validate        Validate environment configuration"
    echo "  setup           Run full setup (validate, build, deploy)"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup                    # Full setup and deployment"
    echo "  $0 deploy-monitoring        # Deploy with monitoring"
    echo "  $0 logs backend             # Show backend logs"
    echo "  $0 health                   # Check service health"
}

# Main execution
main() {
    case "${1:-help}" in
        setup)
            check_dependencies
            validate_environment
            create_directories
            generate_config_files
            build_images
            deploy_services
            sleep 10
            check_health
            show_status
            ;;
        deploy)
            check_dependencies
            validate_environment
            create_directories
            generate_config_files
            deploy_services
            sleep 10
            check_health
            show_status
            ;;
        deploy-monitoring)
            check_dependencies
            validate_environment
            create_directories
            generate_config_files
            build_images
            deploy_with_monitoring
            sleep 15
            check_health
            show_status
            ;;
        deploy-fullstack)
            check_dependencies
            validate_environment
            create_directories
            generate_config_files
            build_images
            deploy_fullstack
            sleep 15
            check_health
            show_status
            ;;
        build)
            check_dependencies
            build_images
            ;;
        health)
            check_health
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "$2"
            ;;
        stop)
            stop_services
            ;;
        validate)
            validate_environment
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"