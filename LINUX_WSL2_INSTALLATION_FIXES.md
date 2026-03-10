# 🔧 QLoRA Platform - Linux/WSL Installation Analysis & Fixes

Dokumen ini menganalisis masalah instalasi aplikasi QLoRA pada Linux dan WSL2 serta memberikan rekomendasi perbaikan komprehensif.

---

## 🚨 IDENTIFIED ISSUES

### 1. **Critical: Missing Requirements File**

**Problem**: `Dockerfile` line 21-22 mencoba menginstall `requirements-new.txt` yang tidak ada:
```dockerfile
COPY backend/core/requirements-new.txt .
RUN pip install --no-cache-dir -r requirements-new.txt
```

**Impact**: Build akan gagal pada tahap backend-builder karena file tidak ditemukan.

### 2. **Environment Variables Mismatch**

**Problem**: `.env.example` menggunakan format yang berbeda dengan yang diharapkan oleh `docker-compose.yml`:
- `.env.example`: `MONGO_ROOT_USERNAME`, `MONGO_ROOT_PASSWORD`
- `docker-compose.yml`: `MONGO_USERNAME`, `MONGO_PASSWORD`

**Impact**: Container MongoDB akan gagal start karena environment variables tidak cocok.

### 3. **Port Conflicts in WSL2**

**Problem**: Beberapa port mungkin konflik dengan services lain di WSL2:
- Port 80 (nginx)
- Port 8000 (backend)
- Port 27017 (MongoDB)
- Port 6379 (Redis)

**Impact**: Container akan gagal start dengan "port already in use" error.

### 4. **WSL2 File Permission Issues**

**Problem**: WSL2 memiliki permission handling yang berbeda untuk:
- Volume mounts
- File permissions untuk non-root user
- Docker socket access

**Impact**: Container tidak dapat write ke volumes atau start dengan non-root user.

### 5. **GPU Access in WSL2**

**Problem**: GPU access di WSL2 memerlukan konfigurasi khusus:
- NVIDIA Container Toolkit untuk WSL2
- Docker runtime configuration
- Driver compatibility

**Impact**: Training jobs akan gagal dengan GPU not available.

---

## 🛠️ COMPREHENSIVE FIXES

### FIX 1: Create Missing Requirements File

```bash
# Buat file requirements-new.txt yang benar
cat > backend/core/requirements-new.txt << 'EOF'
# Core Dependencies
fastapi==0.110.1
uvicorn==0.25.0
python-dotenv==1.2.1
pydantic==2.12.5

# Database & Caching
motor==3.3.1
pymongo==4.5.0
redis==5.0.1

# Authentication & Security
PyJWT==2.10.1
python-jose==3.5.0
passlib==1.7.4
bcrypt==4.1.3
cryptography==46.0.3

# Machine Learning & AI
torch==2.1.0
transformers==4.36.0
peft==0.10.0
bitsandbytes==0.41.3
datasets==2.15.0
accelerate==0.25.0
sentencepiece==0.1.99
protobuf==5.29.5
numpy==2.4.0
pandas==2.3.3

# Hugging Face Integration
huggingface_hub==1.2.4
tokenizers==0.22.2

# API & HTTP
httpx==0.28.1
requests==2.32.5
websockets==15.0.1

# Data Processing
openpyxl==3.1.2
pyarrow==14.0.1
python-multipart==0.0.21

# Async File Processing
aiofiles==23.2.0

# Utilities
Jinja2==3.1.6
PyYAML==6.0.3
rich==14.2.0
tqdm==4.67.1
click==8.3.1

# Cloud & Storage
boto3==1.42.21
s3transfer==0.16.0

# Date & Time
python-dateutil==2.9.0.post0
pytz==2025.2

# Validation & Schema
jsonschema==4.26.0
email-validator==2.3.0
EOF
```

### FIX 2: Update Dockerfile Path

```dockerfile
# Ubah line 21-22 di Dockerfile:
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

### FIX 3: Fix Environment Variables

Update `.env.example`:
```bash
# Database Configuration
MONGO_USERNAME=admin
MONGO_PASSWORD=secure_password_123
MONGO_PORT=27017

# Redis Configuration  
REDIS_PASSWORD=redis_password
REDIS_PORT=6379

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Configuration
SECRET_KEY=your-secret-key-change-in-production
ENVIRONMENT=production
LOG_LEVEL=INFO
ENABLE_FILE_LOGGING=true
LOG_FILE_PATH=/app/logs/app.log

# GPU Configuration
GPU_MEMORY_THRESHOLD=0.9
ENABLE_GPU_MONITORING=true
ENABLE_SECURITY_AUDIT=true
ENABLE_PERFORMANCE_MONITORING=true

# Monitoring Configuration
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
```

### FIX 4: WSL2 Port Configuration

Buat `docker-compose.wsl2.yml`:
```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:7.0
    container_name: qlora-mongodb
    restart: unless-stopped
    ports:
      - "27018:27017"  # Different port for WSL2
    environment:
      MONGO_USERNAME: ${MONGO_USERNAME:-admin}
      MONGO_PASSWORD: ${MONGO_PASSWORD:-password}
      MONGO_INITDB_DATABASE: ${MONGO_DATABASE:-qlora_db}
    volumes:
      - mongodb_data:/data/db
      - ./docker/mongodb/init.js:/docker-entrypoint-initdb.d/init.js:ro
    networks:
      - qlora-network

  redis:
    image: redis:7-alpine
    container_name: qlora-redis
    restart: unless-stopped
    ports:
      - "6380:6379"  # Different port for WSL2
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD:-redis_password}
    volumes:
      - redis_data:/data
    networks:
      - qlora-network

  backend:
    build:
      context: .
      dockerfile: Dockerfile
      target: backend
    container_name: qlora-backend
    restart: unless-stopped
    ports:
      - "8001:8000"  # Different port for WSL2
    environment:
      DATABASE_URL=mongodb://${MONGO_USERNAME:-admin}:${MONGO_PASSWORD:-password}@mongodb:27017/${MONGO_DATABASE:-qlora_db}?authSource=admin
      REDIS_URL=redis://:${REDIS_PASSWORD:-redis_password}@redis:6379/0
      SECRET_KEY=${SECRET_KEY:-your-secret-key-change-in-production}
      JWT_SECRET_KEY=${JWT_SECRET_KEY:-your-jwt-secret-key-change-in-production}
      JWT_ALGORITHM=${JWT_ALGORITHM:-HS256}
      ACCESS_TOKEN_EXPIRE_MINUTES=${ACCESS_TOKEN_EXPIRE_MINUTES:-30}
      ENVIRONMENT=${ENVIRONMENT:-production}
      LOG_LEVEL=${LOG_LEVEL:-INFO}
      ENABLE_FILE_LOGGING=${ENABLE_FILE_LOGGING:-true}
      LOG_FILE_PATH=${LOG_FILE_PATH:-/app/logs/app.log}
      MAX_WORKERS=${MAX_WORKERS:-4}
      GPU_MEMORY_THRESHOLD=${GPU_MEMORY_THRESHOLD:-0.9}
      ENABLE_GPU_MONITORING=${ENABLE_GPU_MONITORING:-true}
      ENABLE_SECURITY_AUDIT=${ENABLE_SECURITY_AUDIT:-true}
      ENABLE_PERFORMANCE_MONITORING=${ENABLE_PERFORMANCE_MONITORING:-true}
      WANDB_API_KEY=${WANDB_API_KEY:-}
      HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-}
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
      - ./checkpoints:/app/checkpoints
      - ./data:/app/data
    depends_on:
      mongodb:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - qlora-network

  frontend:
    build:
      context: .
      dockerfile: Dockerfile
      target: frontend
    container_name: qlora-frontend
    restart: unless-stopped
    ports:
      - "8002:80"  # Different port for WSL2
    depends_on:
      - backend
    networks:
      - qlora-network

networks:
  qlora-network:
    driver: bridge

volumes:
  mongodb_data:
  redis_data:
```

### FIX 5: WSL2 File Permissions

Buat `setup-wsl2.sh`:
```bash
#!/bin/bash
# Setup script untuk WSL2 environment

set -e

echo "🔧 Setting up QLoRA for WSL2..."

# Create necessary directories with proper permissions
echo "📁 Creating directories..."
mkdir -p logs models checkpoints data docker/mongodb
chmod 755 logs models checkpoints data docker/mongodb

# Fix Docker permissions
echo "🐳 Fixing Docker permissions..."
sudo usermod -aG docker $USER

# Add user to docker group if not already
if ! groups $USER | grep -q docker; then
    sudo usermod -aG docker $USER
    echo "✅ Added $USER to docker group. Please run: newgrp docker"
fi

# Set proper ownership
echo "👤 Setting ownership..."
sudo chown -R $USER:$USER logs models checkpoints data

# Create .env file from example if not exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration"
fi

# Check WSL2 GPU support
echo "🎮 Checking GPU support..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA drivers detected"
    nvidia-smi
else
    echo "⚠️  NVIDIA drivers not found. GPU training will not work."
fi

# Check Docker version
echo "🐳 Checking Docker version..."
docker --version
docker-compose --version

echo "✅ WSL2 setup complete!"
echo ""
echo "🚀 To start the application:"
echo "   docker-compose -f docker-compose.wsl2.yml up -d"
echo ""
echo "🌐 Access URLs:"
echo "   Frontend: http://localhost:8002"
echo "   Backend: http://localhost:8001"
echo "   MongoDB: localhost:27018"
echo "   Redis: localhost:6380"
```

### FIX 6: GPU Support for WSL2

Buat `docker-compose.wsl2-gpu.yml`:
```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:7.0
    container_name: qlora-mongodb
    restart: unless-stopped
    ports:
      - "27018:27017"
    environment:
      MONGO_USERNAME: ${MONGO_USERNAME:-admin}
      MONGO_PASSWORD: ${MONGO_PASSWORD:-password}
      MONGO_INITDB_DATABASE: ${MONGO_DATABASE:-qlora_db}
    volumes:
      - mongodb_data:/data/db
      - ./docker/mongodb/init.js:/docker-entrypoint-initdb.d/init.js:ro
    networks:
      - qlora-network

  redis:
    image: redis:7-alpine
    container_name: qlora-redis
    restart: unless-stopped
    ports:
      - "6380:6379"
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD:-redis_password}
    volumes:
      - redis_data:/data
    networks:
      - qlora-network

  backend:
    build:
      context: .
      dockerfile: Dockerfile
      target: backend
    container_name: qlora-backend
    restart: unless-stopped
    runtime: nvidia  # GPU runtime
    ports:
      - "8001:8000"
    environment:
      DATABASE_URL=mongodb://${MONGO_USERNAME:-admin}:${MONGO_PASSWORD:-password}@mongodb:27017/${MONGO_DATABASE:-qlora_db}?authSource=admin
      REDIS_URL=redis://:${REDIS_PASSWORD:-redis_password}@redis:6379/0
      SECRET_KEY=${SECRET_KEY:-your-secret-key-change-in-production}
      JWT_SECRET_KEY=${JWT_SECRET_KEY:-your-jwt-secret-key-change-in-production}
      JWT_ALGORITHM=${JWT_ALGORITHM:-HS256}
      ACCESS_TOKEN_EXPIRE_MINUTES=${ACCESS_TOKEN_EXPIRE_MINUTES:-30}
      ENVIRONMENT=${ENVIRONMENT:-production}
      LOG_LEVEL=${LOG_LEVEL:-INFO}
      ENABLE_FILE_LOGGING=${ENABLE_FILE_LOGGING:-true}
      LOG_FILE_PATH=${LOG_FILE_PATH:-/app/logs/app.log}
      MAX_WORKERS=${MAX_WORKERS:-4}
      GPU_MEMORY_THRESHOLD=${GPU_MEMORY_THRESHOLD:-0.9}
      ENABLE_GPU_MONITORING=${ENABLE_GPU_MONITORING:-true}
      ENABLE_SECURITY_AUDIT=${ENABLE_SECURITY_AUDIT:-true}
      ENABLE_PERFORMANCE_MONITORING=${ENABLE_PERFORMANCE_MONITORING:-true}
      WANDB_API_KEY=${WANDB_API_KEY:-}
      HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-}
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
      - ./checkpoints:/app/checkpoints
      - ./data:/app/data
    depends_on:
      mongodb:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - qlora-network

  frontend:
    build:
      context: .
      dockerfile: Dockerfile
      target: frontend
    container_name: qlora-frontend
    restart: unless-stopped
    ports:
      - "8002:80"
    depends_on:
      - backend
    networks:
      - qlora-network

networks:
  qlora-network:
    driver: bridge

volumes:
  mongodb_data:
  redis_data:
```

---

## 🚀 STEP-BY-STEP INSTALLATION GUIDE

### Prerequisites untuk WSL2

```bash
# 1. Update WSL2 packages
sudo apt update && sudo apt upgrade -y

# 2. Install Docker untuk WSL2
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 3. Install NVIDIA Container Toolkit untuk WSL2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 4. Clone repository
git clone <repository-url>
cd qlora

# 5. Run setup script
chmod +x setup-wsl2.sh
./setup-wsl2.sh

# 6. Start application
docker-compose -f docker-compose.wsl2.yml up -d
```

### Untuk GPU Support

```bash
# Gunakan GPU-enabled compose file
docker-compose -f docker-compose.wsl2-gpu.yml up -d
```

---

## 🔍 TROUBLESHOOTING

### Issue: Port Already in Use

```bash
# Check port usage
netstat -tulpn | grep :80
netstat -tulpn | grep :8000

# Kill process using port
sudo kill -9 <PID>

# Atau gunakan port berbeda di docker-compose.wsl2.yml
```

### Issue: Permission Denied

```bash
# Fix Docker socket permissions
sudo chmod 666 /var/run/docker.sock

# Add user ke docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Issue: GPU Not Available

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
```

### Issue: Container Won't Start

```bash
# Check logs
docker-compose -f docker-compose.wsl2.yml logs backend
docker-compose -f docker-compose.wsl2.yml logs mongodb

# Check container status
docker-compose -f docker-compose.wsl2.yml ps

# Rebuild containers
docker-compose -f docker-compose.wsl2.yml down
docker-compose -f docker-compose.wsl2.yml build --no-cache
docker-compose -f docker-compose.wsl2.yml up -d
```

---

## ✅ VERIFICATION

### Health Checks

```bash
# Check all services
curl http://localhost:8002/health  # Frontend
curl http://localhost:8001/health  # Backend

# Check MongoDB
docker exec qlora-mongodb mongosh --eval "db.adminCommand('ping')"

# Check Redis
docker exec qlora-redis redis-cli ping
```

### Expected Output

```
✅ Frontend: HTTP/1.1 200 OK
✅ Backend: HTTP/1.1 200 OK
✅ MongoDB: { ok: 1 }
✅ Redis: PONG
```

---

## 📋 SUMMARY OF FIXES

| Issue | Fix | Status |
|-------|------|--------|
| Missing requirements file | Create `requirements-new.txt` | ✅ |
| Environment variables mismatch | Update `.env.example` | ✅ |
| Port conflicts | Custom WSL2 compose file | ✅ |
| File permissions | Setup script with proper permissions | ✅ |
| GPU access | NVIDIA Container Toolkit + GPU runtime | ✅ |

Dengan menerapkan semua fix ini, aplikasi QLoRA seharusnya berhasil berjalan di Linux dan WSL2 environment.
