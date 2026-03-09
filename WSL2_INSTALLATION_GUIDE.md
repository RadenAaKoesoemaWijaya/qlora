# Panduan Lengkap Instalasi QLoRA di Windows Subsystem Linux (WSL2)

## 📋 Prasyarat Sistem

### Hardware Requirements
- **RAM**: Minimum 16GB (32GB+ direkomendasikan untuk model besar)
- **Storage**: 100GB+ ruang disk kosong
- **GPU**: NVIDIA GPU dengan 8GB+ VRAM (RTX 3080/4070 atau lebih baik)
- **OS**: Windows 10/11 dengan WSL2

### Software yang Dibutuhkan
1. **Windows 10/11** (versi 2004 atau lebih baru)
2. **WSL2** terinstall
3. **Docker Desktop** dengan WSL2 integration
4. **Git** untuk cloning repository
5. **NVIDIA GPU Driver** (jika menggunakan GPU)

---

## 🚀 Langkah 1: Setup WSL2

### 1.1 Install WSL2 (jika belum)
Buka PowerShell sebagai Administrator dan jalankan:
```powershell
# Enable WSL
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Virtual Machine Platform
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart Windows
Restart-Computer

# Set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu (recommended)
wsl --install -d Ubuntu
```

### 1.2 Konfigurasi WSL2
Setelah instalasi selesai, buka Ubuntu dan setup:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install basic dependencies
sudo apt install -y curl wget git unzip python3 python3-pip
```

---

## 🐳 Langkah 2: Install Docker

### 2.1 Install Docker Desktop
1. Download Docker Desktop for Windows dari [docker.com](https://www.docker.com/products/docker-desktop)
2. Install dengan opsi "Use WSL 2 based engine"
3. Enable WSL2 integration di Settings > Resources > WSL Integration

### 2.2 Verifikasi Docker di WSL2
```bash
# Test Docker
docker --version
docker-compose --version

# Test Docker run
docker run hello-world
```

---

## 🎯 Langkah 3: Setup Project QLoRA

### 3.1 Clone Repository
```bash
# Clone project (ganti dengan repository URL yang sesuai)
git clone <repository-url>
cd qlora

# Atau jika sudah ada di Windows, copy ke WSL2:
# cp -r /mnt/e/qlora ~/qlora
# cd ~/qlora
```

### 3.2 Setup Environment File
```bash
# Copy environment template
cp .env.example .env

# Edit environment file
nano .env
```

**Konfigurasi minimal yang diperlukan di `.env`:**
```env
# Database
MONGO_USERNAME=admin
MONGO_PASSWORD=secure_password_123
MONGO_DATABASE=qlora_db

# Security
SECRET_KEY=your-32-character-secret-key-here
JWT_SECRET_KEY=your-32-character-jwt-secret-key-here

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO

# GPU (jika ada)
GPU_MEMORY_THRESHOLD=0.9
ENABLE_GPU_MONITORING=true

# Opsional: API keys untuk ML platforms
WANDB_API_KEY=your-wandb-api-key
HUGGINGFACE_TOKEN=your-huggingface-token
```

---

## 🔧 Langkah 4: Setup NVIDIA GPU Support (Opsional)

### 4.1 Install NVIDIA Container Toolkit
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 4.2 Verifikasi GPU di Docker
```bash
# Check GPU availability
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

---

## 🚀 Langkah 5: Deploy Aplikasi

### 5.1 Menggunakan Deployment Script (Recommended)
```bash
# Make script executable
chmod +x deploy.sh

# Full setup dan deployment
./deploy.sh setup
```

**Commands yang tersedia:**
```bash
./deploy.sh setup              # Setup lengkap
./deploy.sh deploy             # Deploy basic services
./deploy.sh deploy-monitoring  # Deploy dengan monitoring
./deploy.sh deploy-fullstack   # Deploy full-stack
./deploy.sh health             # Check service health
./deploy.sh status             # Show status
./deploy.sh logs [service]     # Show logs
./deploy.sh stop               # Stop services
```

### 5.2 Manual Deployment (Alternative)
```bash
# Create required directories
mkdir -p logs models checkpoints data
mkdir -p docker/mongodb docker/prometheus docker/grafana/provisioning

# Build Docker images
docker-compose build

# Deploy services
docker-compose up -d mongodb redis
sleep 30
docker-compose up -d backend frontend
```

---

## 🌐 Langkah 6: Akses Aplikasi

Setelah deployment berhasil, aplikasi dapat diakses melalui:

### Access Points
- **Frontend UI**: http://localhost (port 80)
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Jika menggunakan monitoring:
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### Jika menggunakan full-stack:
- **Application**: http://localhost:8080
- **API**: http://localhost:8081

---

## 🔍 Langkah 7: Verifikasi dan Troubleshooting

### 7.1 Check Service Health
```bash
# Check all services
./deploy.sh health

# Check service status
./deploy.sh status

# View logs
./deploy.sh logs backend
./deploy.sh logs frontend
./deploy.sh logs mongodb
```

### 7.2 Manual Health Check
```bash
# Check backend
curl http://localhost:8000/health

# Check frontend
curl http://localhost

# Check database
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"

# Check Redis
docker-compose exec redis redis-cli ping
```

### 7.3 Common Issues and Solutions

#### Issue 1: Docker tidak bisa diakses dari WSL2
```bash
# Restart Docker service
sudo service docker restart

# Check Docker daemon
sudo docker info
```

#### Issue 2: Port conflict
```bash
# Check ports in use
netstat -tulpn | grep :80
netstat -tulpn | grep :8000

# Kill processes using ports
sudo fuser -k 80/tcp
sudo fuser -k 8000/tcp
```

#### Issue 3: Memory issues
```bash
# Check memory usage
free -h
docker stats

# Increase WSL2 memory (edit %USERPROFILE%\.wslconfig)
# Create file .wslconfig:
[wsl2]
memory=32GB
swap=8GB
```

#### Issue 4: GPU not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Reinstall NVIDIA Container Toolkit
sudo apt remove nvidia-docker2
sudo apt install nvidia-docker2
sudo systemctl restart docker
```

#### Issue 5: Database connection failed
```bash
# Check MongoDB logs
./deploy.sh logs mongodb

# Restart MongoDB
docker-compose restart mongodb

# Check network
docker network ls
docker network inspect qlora_qlora-network
```

---

## 📊 Langkah 8: Monitoring dan Maintenance

### 8.1 View Logs
```bash
# All logs
docker-compose logs

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# Real-time monitoring
watch -n 2 'docker-compose ps'
```

### 8.2 Backup Data
```bash
# Backup database
docker-compose exec mongodb mongodump --out /backup --gzip

# Backup models and checkpoints
tar -czf qlora-backup-$(date +%Y%m%d).tar.gz models/ checkpoints/ data/
```

### 8.3 Update Application
```bash
# Pull latest changes
git pull

# Rebuild and redeploy
docker-compose down
docker-compose build --no-cache
./deploy.sh deploy
```

---

## 🛠️ Advanced Configuration

### 9.1 Custom Ports
Edit `.env` file:
```env
BACKEND_PORT=8001
FRONTEND_PORT=8080
NGINX_PORT=8888
```

### 9.2 Resource Limits
Edit `docker-compose.yml`:
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
```

### 9.3 SSL/TLS Setup
```bash
# Generate SSL certificates
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/privkey.pem -out ssl/fullchain.pem

# Update nginx configuration
# Edit nginx.conf untuk SSL
```

---

## 📝 Tips dan Best Practices

### Performance Optimization
1. **SSD Storage**: Gunakan SSD untuk model storage
2. **Memory Allocation**: Allocate sufficient RAM untuk WSL2
3. **GPU Optimization**: Gunakan NVIDIA GPU untuk training
4. **Network**: Gunakan koneksi internet stabil untuk model downloads

### Security
1. **Change Default Passwords**: Update password di `.env`
2. **Firewall**: Configure firewall untuk port yang dibutuhkan
3. **SSL**: Enable SSL untuk production
4. **Regular Updates**: Keep Docker dan dependencies updated

### Development Workflow
```bash
# Development mode
cd backend && pip install -r requirements.txt && python server.py
cd frontend && npm install && npm run dev

# Production mode
./deploy.sh setup
```

---

## 🆘 Bantuan dan Support

### Resources
- **Documentation**: Lihat `README.md` dan `DEPLOYMENT.md`
- **API Documentation**: http://localhost:8000/docs
- **Monitoring**: Grafana dashboards (jika di-enable)

### Troubleshooting Commands
```bash
# System info
docker system df
docker system prune

# Service restart
docker-compose restart [service-name]

# Full reset
docker-compose down -v
docker system prune -a
./deploy.sh setup
```

### Community Support
- Check logs di `./logs/` directory
- Use `./deploy.sh health` untuk diagnostic
- Monitor resource usage dengan `./deploy.sh status`

---

## ✅ Checklist Selesai

Sebelum aplikasi siap digunakan, pastikan:

- [ ] WSL2 terinstall dengan benar
- [ ] Docker Desktop dengan WSL2 integration
- [ ] Project berhasil di-clone
- [ ] Environment file (.env) sudah dikonfigurasi
- [ ] Docker images berhasil di-build
- [ ] Semua services running (MongoDB, Redis, Backend, Frontend)
- [ ] Health checks passed
- [ ] Aplikasi dapat diakses via browser
- [ ] GPU support (jika diperlukan) sudah terkonfigurasi

---

**🎉 Selamat! Aplikasi QLoRA Anda sekarang sudah running di WSL2!**

Untuk pertanyaan atau masalah, cek bagian Troubleshooting atau lihat logs untuk detail error.
