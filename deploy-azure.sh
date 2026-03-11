#!/bin/bash
# Azure Deployment Script untuk QLoRA Application
# Script ini mendeploy aplikasi ke Azure Container Instances dan Azure Database

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

# Azure Configuration
RESOURCE_GROUP="qlora-rg"
LOCATION="eastus"
ACR_NAME="qloraregistry$(date +%s | tail -c 6)"
CONTAINER_APP_NAME="qlora-app"
COSMOS_DB_NAME="qlora-cosmos"
REDIS_CACHE_NAME="qlora-redis"

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

# Check Azure CLI
check_azure_cli() {
    log_step "Checking Azure CLI..."
    
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed. Please install it first:"
        echo "curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"
        exit 1
    fi
    
    # Check if logged in
    if ! az account show &> /dev/null; then
        log_warn "Not logged in to Azure. Please login:"
        az login
    fi
    
    log_info "Azure CLI is ready."
}

# Create Resource Group
create_resource_group() {
    log_step "Creating Resource Group..."
    
    if ! az group show --name "$RESOURCE_GROUP" &> /dev/null; then
        az group create \
            --name "$RESOURCE_GROUP" \
            --location "$LOCATION"
        log_info "Resource Group '$RESOURCE_GROUP' created."
    else
        log_info "Resource Group '$RESOURCE_GROUP' already exists."
    fi
}

# Create Azure Container Registry
create_container_registry() {
    log_step "Creating Azure Container Registry..."
    
    if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        az acr create \
            --resource-group "$RESOURCE_GROUP" \
            --name "$ACR_NAME" \
            --sku Basic \
            --admin-enabled true
        log_info "Container Registry '$ACR_NAME' created."
    else
        log_info "Container Registry '$ACR_NAME' already exists."
    fi
    
    # Get ACR credentials
    ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query loginServer -o tsv)
    ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query username -o tsv)
    ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query passwords[0].value -o tsv)
    
    log_info "ACR Login Server: $ACR_LOGIN_SERVER"
}

# Build and Push Docker Images
build_and_push_images() {
    log_step "Building and Pushing Docker Images..."
    
    # Login to ACR
    az acr login --name "$ACR_NAME"
    
    # Build backend image
    log_info "Building backend image..."
    docker build -t "$ACR_LOGIN_SERVER/qlora-backend:latest" --target backend .
    
    # Build frontend image
    log_info "Building frontend image..."
    docker build -t "$ACR_LOGIN_SERVER/qlora-frontend:latest" --target frontend .
    
    # Build fullstack image
    log_info "Building fullstack image..."
    docker build -t "$ACR_LOGIN_SERVER/qlora-fullstack:latest" --target fullstack .
    
    # Push images
    log_info "Pushing images to ACR..."
    docker push "$ACR_LOGIN_SERVER/qlora-backend:latest"
    docker push "$ACR_LOGIN_SERVER/qlora-frontend:latest"
    docker push "$ACR_LOGIN_SERVER/qlora-fullstack:latest"
    
    log_info "All images pushed to ACR successfully."
}

# Create Cosmos DB (MongoDB compatible)
create_cosmos_db() {
    log_step "Creating Cosmos DB (MongoDB compatible)..."
    
    if ! az cosmosdb show --name "$COSMOS_DB_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        az cosmosdb create \
            --name "$COSMOS_DB_NAME" \
            --resource-group "$RESOURCE_GROUP" \
            --locations "East US"=0 "West US"=1 \
            --kind MongoDB \
            --server-version "4.0" \
            --default-consistency-level "Session"
        
        # Create database
        az cosmosdb mongodb database create \
            --account-name "$COSMOS_DB_NAME" \
            --resource-group "$RESOURCE_GROUP" \
            --name "qlora_db"
        
        log_info "Cosmos DB '$COSMOS_DB_NAME' created."
    else
        log_info "Cosmos DB '$COSMOS_DB_NAME' already exists."
    fi
    
    # Get connection string
    COSMOS_CONNECTION_STRING=$(az cosmosdb keys list \
        --name "$COSMOS_DB_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --type connection-strings \
        --query connectionStrings[0].connectionString -o tsv)
    
    log_info "Cosmos DB connection string obtained."
}

# Create Redis Cache
create_redis_cache() {
    log_step "Creating Redis Cache..."
    
    if ! az redis show --name "$REDIS_CACHE_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        az redis create \
            --name "$REDIS_CACHE_NAME" \
            --resource-group "$RESOURCE_GROUP" \
            --location "$LOCATION" \
            --sku Basic \
            --vm-size C0 \
            --redis-version 6
        
        log_info "Redis Cache '$REDIS_CACHE_NAME' created."
    else
        log_info "Redis Cache '$REDIS_CACHE_NAME' already exists."
    fi
    
    # Get Redis connection details
    REDIS_HOST=$(az redis show --name "$REDIS_CACHE_NAME" --resource-group "$RESOURCE_GROUP" --query hostName -o tsv)
    REDIS_PORT=$(az redis show --name "$REDIS_CACHE_NAME" --resource-group "$RESOURCE_GROUP" --query port -o tsv)
    REDIS_PASSWORD=$(az redis list-keys --name "$REDIS_CACHE_NAME" --resource-group "$RESOURCE_GROUP" --query primaryKey -o tsv)
    
    log_info "Redis connection details obtained."
}

# Create Container App Environment
create_container_app_environment() {
    log_step "Creating Container App Environment..."
    
    if ! az containerapp env show --name "qlora-env" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        az containerapp env create \
            --name "qlora-env" \
            --resource-group "$RESOURCE_GROUP" \
            --location "$LOCATION"
        
        log_info "Container App Environment created."
    else
        log_info "Container App Environment already exists."
    fi
}

# Deploy Backend Container App
deploy_backend_app() {
    log_step "Deploying Backend Container App..."
    
    az containerapp create \
        --name "qlora-backend" \
        --resource-group "$RESOURCE_GROUP" \
        --environment "qlora-env" \
        --image "$ACR_LOGIN_SERVER/qlora-backend:latest" \
        --registry-server "$ACR_LOGIN_SERVER" \
        --registry-username "$ACR_USERNAME" \
        --registry-password "$ACR_PASSWORD" \
        --target-port 8000 \
        --ingress external \
        --cpu 2 \
        --memory 4Gi \
        --min-replicas 1 \
        --max-replicas 3 \
        --env-vars \
            DATABASE_URL="$COSMOS_CONNECTION_STRING" \
            REDIS_URL="redis://:$REDIS_PASSWORD@$REDIS_HOST:$REDIS_PORT/0" \
            REDIS_PASSWORD="$REDIS_PASSWORD" \
            SECRET_KEY="your-32-character-secret-key-here-change-this-in-production" \
            JWT_SECRET_KEY="your-32-character-jwt-secret-key-here-change-this-in-production" \
            ENVIRONMENT="production" \
            LOG_LEVEL="INFO" \
            GPU_MEMORY_THRESHOLD="0.9" \
            ENABLE_GPU_MONITORING="false" \
            MAX_WORKERS="4" \
            ENABLE_PERFORMANCE_MONITORING="true" \
            ENABLE_SECURITY_AUDIT="true"
    
    # Get backend URL
    BACKEND_URL=$(az containerapp show --name "qlora-backend" --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)
    log_info "Backend deployed at: https://$BACKEND_URL"
}

# Deploy Frontend Container App
deploy_frontend_app() {
    log_step "Deploying Frontend Container App..."
    
    az containerapp create \
        --name "qlora-frontend" \
        --resource-group "$RESOURCE_GROUP" \
        --environment "qlora-env" \
        --image "$ACR_LOGIN_SERVER/qlora-frontend:latest" \
        --registry-server "$ACR_LOGIN_SERVER" \
        --registry-username "$ACR_USERNAME" \
        --registry-password "$ACR_PASSWORD" \
        --target-port 80 \
        --ingress external \
        --cpu 0.5 \
        --memory 1Gi \
        --min-replicas 1 \
        --max-replicas 2
    
    # Get frontend URL
    FRONTEND_URL=$(az containerapp show --name "qlora-frontend" --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)
    log_info "Frontend deployed at: https://$FRONTEND_URL"
}

# Setup Application Insights (Monitoring)
setup_monitoring() {
    log_step "Setting up Application Insights..."
    
    if ! az monitor app-insights component show --app "qlora-insights" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        az monitor app-insights component create \
            --app "qlora-insights" \
            --location "$LOCATION" \
            --resource-group "$RESOURCE_GROUP" \
            --application-type web
        
        log_info "Application Insights created."
    else
        log_info "Application Insights already exists."
    fi
    
    # Get instrumentation key
    INSTRUMENTATION_KEY=$(az monitor app-insights component show \
        --app "qlora-insights" \
        --resource-group "$RESOURCE_GROUP" \
        --query instrumentationKey -o tsv)
    
    log_info "Application Insights instrumentation key obtained."
}

# Create Azure Files for persistent storage
create_storage() {
    log_step "Creating Azure Files for persistent storage..."
    
    STORAGE_ACCOUNT_NAME="qlorastorage$(date +%s | tail -c 6)"
    
    az storage account create \
        --name "$STORAGE_ACCOUNT_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --sku Standard_LRS \
        --kind StorageV2
    
    # Create file shares
    az storage share create \
        --account-name "$STORAGE_ACCOUNT_NAME" \
        --name "models"
    
    az storage share create \
        --account-name "$STORAGE_ACCOUNT_NAME" \
        --name "checkpoints"
    
    az storage share create \
        --account-name "$STORAGE_ACCOUNT_NAME" \
        --name "data"
    
    log_info "Azure Files created for persistent storage."
}

# Show deployment summary
show_deployment_summary() {
    log_step "Deployment Summary"
    echo ""
    echo "🎉 QLoRA Application deployed successfully to Azure!"
    echo ""
    echo "📊 Resource Group: $RESOURCE_GROUP"
    echo "🐳 Container Registry: $ACR_LOGIN_SERVER"
    echo "🗄️ Cosmos DB: $COSMOS_DB_NAME"
    echo "🗃️ Redis Cache: $REDIS_CACHE_NAME"
    echo "💾 Storage Account: $STORAGE_ACCOUNT_NAME"
    echo "📈 Application Insights: qlora-insights"
    echo ""
    echo "🌐 Application URLs:"
    echo "  Frontend: https://$FRONTEND_URL"
    echo "  Backend:  https://$BACKEND_URL"
    echo ""
    echo "🔧 Useful Commands:"
    echo "  az containerapp logs show --name qlora-backend --resource-group $RESOURCE_GROUP --follow"
    echo "  az containerapp logs show --name qlora-frontend --resource-group $RESOURCE_GROUP --follow"
    echo "  az containerapp revision list --name qlora-backend --resource-group $RESOURCE_GROUP"
    echo "  az containerapp revision list --name qlora-frontend --resource-group $RESOURCE_GROUP"
    echo ""
    echo "⚠️  Important Notes:"
    echo "  - Update SECRET_KEY and JWT_SECRET_KEY with secure values"
    echo "  - Configure WANDB_API_KEY and HUGGINGFACE_TOKEN if needed"
    echo "  - Set up proper networking and security rules"
    echo "  - Consider using Azure Key Vault for secrets management"
    echo ""
}

# Cleanup function
cleanup() {
    log_warn "Cleaning up..."
    # Add any cleanup logic here
}

# Main execution
main() {
    echo "🚀 QLoRA Azure Deployment Script"
    echo "================================="
    echo ""
    
    # Set up error handling
    trap cleanup EXIT
    
    check_azure_cli
    create_resource_group
    create_container_registry
    build_and_push_images
    create_cosmos_db
    create_redis_cache
    create_container_app_environment
    create_storage
    setup_monitoring
    deploy_backend_app
    deploy_frontend_app
    show_deployment_summary
    
    log_info "Azure deployment completed successfully!"
}

# Run main function
main "$@"
