# QLoRA Fine-tuning Platform

Platform web-based canggih untuk fine-tuning Large Language Models (LLM) menggunakan metode QLoRA (Quantized Low-Rank Adaptation). Menyediakan antarmuka yang user-friendly untuk melakukan fine-tuning model AI secara efisien dengan teknik kuantisasi 4-bit dan monitoring real-time.

## 🚀 Fitur Utama

### 1. **Model Selection & Management**
- Pilihan model pre-trained: Llama 2, Llama 3, Mistral, Gemma, Mixtral
- Informasi detail tentang ukuran model dan karakteristik
- Support untuk model 7B hingga 47B parameters
- Hugging Face integration dengan token authentication

### 2. **Advanced Dataset Management**
- Upload dataset dalam format JSON, JSONL, CSV, TXT, Parquet, XLSX
- Validasi dataset otomatis dengan quality metrics
- Manajemen dataset dengan informasi jumlah baris dan ukuran file
- Data preprocessing dan cleaning otomatis

### 3. **Training Configuration**
- Konfigurasi parameter QLoRA (rank, alpha, dropout)
- Pengaturan learning rate, batch size, dan epochs
- Custom target modules untuk adaptation
- GPU memory management dan optimization
- Advanced hyperparameter tuning

### 4. **Real-time Training Monitor**
- Real-time monitoring progress training via WebSocket
- Visualisasi loss, learning rate, dan metrics
- Checkpoint management otomatis
- Estimasi waktu completion
- GPU resource usage monitoring

### 5. **Advanced Evaluation & Analytics**
- **Evaluation Engine**: Menggunakan library `evaluate` dari Hugging Face
- **Metrics**:
  - **BERTScore**: Semantic similarity (F1)
  - **ROUGE-L**: Longest common subsequence untuk summarization
  - **BLEU**: N-gram precision untuk translation
  - **Perplexity**: Model fluency/uncertainty
- Perbandingan performa model
- History training dan evaluasi

### 6. **Enhanced Training Engine**
- **Smart Dispatcher**: Otomatis mendeteksi ketersediaan GPU dan library ML
- **Real Training**: Menjalankan fine-tuning QLoRA asli menggunakan `bitsandbytes`, `peft`, dan `transformers`
- **Multi-GPU Support**: Distribusi training di multiple GPU
- **Error Recovery**: Automatic retry dan rollback mechanisms
- **Memory Optimization**: 4-bit quantization untuk efisiensi memory

### 7. **Security & Authentication**
- JWT-based authentication dengan role-based access control
- API key management untuk programmatic access
- Security audit logging dan suspicious activity detection
- Input validation dan sanitization

### 8. **Monitoring & Observability**
- Structured logging dengan performance metrics
- Prometheus metrics collection
- Grafana dashboards untuk visualisasi
- Real-time alerting untuk critical events
- System resource monitoring (CPU, memory, disk, GPU)

## 🏗️ Arsitektur Sistem

### Backend (Python/FastAPI)
- **Framework**: FastAPI untuk high-performance REST API dengan async support
- **Database**: MongoDB untuk penyimpanan data dengan validation
- **Cache**: Redis untuk session management dan caching
- **Arsitektur**: Microservices dengan async/await pattern
- **Features**: Enhanced training engine, automatic checkpointing, real-time monitoring

### Frontend (React + TypeScript)
- **Framework**: React 19 dengan TypeScript dan React Router
- **UI Components**: Radix UI + Tailwind CSS dengan shadcn/ui
- **State Management**: React Hooks dengan TypeScript
- **Real-time**: WebSocket integration untuk live updates
- **Build Tool**: Vite untuk development dan production build

### Infrastructure
- **Containerization**: Docker dengan multi-stage builds
- **Orchestration**: Docker Compose untuk service management
- **GPU Support**: NVIDIA Docker Runtime untuk GPU acceleration
- **Monitoring**: Prometheus, Grafana, dan alerting rules

## 📋 Prasyarat

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended) atau Windows dengan WSL2
- **Memory**: Minimum 16GB RAM (32GB+ recommended untuk large models)
- **Storage**: 100GB+ available disk space
- **GPU**: NVIDIA GPU dengan 8GB+ VRAM (RTX 3080/4070 atau better recommended)

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (untuk development)
- Git
- NVIDIA Docker Runtime (untuk GPU support)

## 🛠️ Instalasi dan Setup

### Quick Start (Recommended)
```bash
# 1. Clone repository
git clone <repository-url>
cd qlora

# 2. Setup environment
cp .env.example .env
# Edit .env dengan konfigurasi Anda

# 3. Validasi environment
python backend/core/environment_validator.py --env-file .env

# 4. Deploy aplikasi
chmod +x deploy.sh
./deploy.sh setup
```

### Access Points
- **Frontend**: http://localhost
- **Backend API**: http://localhost:8000/docs
- **Grafana Monitoring**: http://localhost:3000 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090

### Development Setup
```bash
# Backend setup
cd backend
pip install -r requirements.txt
python server.py

# Frontend setup
cd frontend
npm install
npm run dev
```

## 🔧 Konfigurasi

### Environment Variables (.env)
```env
# Database
DATABASE_URL=mongodb://admin:password@localhost:27017/qlora_db?authSource=admin
MONGO_PASSWORD=your-secure-mongodb-password
REDIS_PASSWORD=your-secure-redis-password

# Security
SECRET_KEY=your-32-character-secret-key-here-change-this-in-production
JWT_SECRET_KEY=your-32-character-jwt-secret-key-here-change-this-in-production

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
MAX_WORKERS=4

# GPU Configuration
GPU_MEMORY_THRESHOLD=0.9
ENABLE_GPU_MONITORING=true

# ML Platform Integration
WANDB_API_KEY=your-wandb-api-key
HUGGINGFACE_TOKEN=your-huggingface-token
```

## 🎯 Mekanisme Kerja

### 1. **Proses Fine-tuning**
1. User memilih model base (Llama, Mistral, dll)
2. Upload dataset training dalam format yang didukung
3. Konfigurasi parameter QLoRA sesuai kebutuhan
4. Sistem otomatis memilih GPU yang optimal
5. Start training job dengan monitoring real-time
6. Sistem otomatis membuat checkpoint setiap epoch
7. Evaluasi model setelah training selesai

### 2. **QLoRA Implementation**
- **Quantization**: Model base dikonversi ke 4-bit precision
- **Low-Rank Adaptation**: Menambah adapter layers dengan rank rendah
- **Memory Efficient**: Mengurangi memory usage hingga 75%
- **Performance**: Mempertahankan kualitas model original

### 3. **Training Pipeline**
```
Dataset Upload → Validation → Model Selection → GPU Allocation →
Parameter Configuration → Training Start → Progress Monitoring →
Checkpoint Creation → Model Evaluation → Result Storage
```

## 📊 API Endpoints

### Models
- `GET /api/models` - List available models
- `GET /api/models/{id}` - Get model details
- `POST /api/models/load` - Load model ke GPU

### Datasets
- `POST /api/datasets/upload` - Upload dataset
- `GET /api/datasets` - List datasets
- `GET /api/datasets/{id}/validate` - Validate dataset
- `DELETE /api/datasets/{id}` - Delete dataset

### Training
- `POST /api/training/start` - Start training job
- `GET /api/training/jobs` - List training jobs
- `GET /api/training/jobs/{id}` - Get job details
- `GET /api/training/jobs/{id}/progress` - Get real-time progress
- `POST /api/training/jobs/{id}/stop` - Stop training
- `DELETE /api/training/jobs/{id}` - Cancel job

### Checkpoints
- `GET /api/checkpoints` - List checkpoints
- `GET /api/checkpoints/{id}/download` - Download checkpoint
- `DELETE /api/checkpoints/{id}` - Delete checkpoint

### Evaluation
- `POST /api/evaluate` - Evaluate model
- `GET /api/evaluations` - List evaluations
- `GET /api/evaluations/{id}/results` - Get evaluation results

### Authentication
- `POST /api/auth/register` - Register user
- `POST /api/auth/login` - Login user
- `POST /api/auth/logout` - Logout user
- `GET /api/auth/profile` - Get user profile

### API Keys
- `POST /api/api-keys` - Create API key
- `GET /api/api-keys` - List API keys
- `DELETE /api/api-keys/{id}` - Revoke API key

## 🚀 Deployment Options

### Basic Deployment
```bash
# Deploy core services saja
./deploy.sh deploy
```

### Full Deployment with Monitoring
```bash
# Deploy dengan monitoring stack (Prometheus, Grafana)
./deploy.sh deploy-monitoring
```

### Full-Stack Deployment
```bash
# Deploy combined backend dan frontend
./deploy.sh deploy-fullstack
```

### Production Deployment
```bash
# Full setup dengan validasi
./deploy.sh setup
```

## ☁️ Cloud Deployment Recommendations

> **Note**: QLoRA training memerlukan **GPU NVIDIA** yang tidak tersedia di Railway atau Render. Berikut platform yang direkomendasikan:

### 🎯 Recommended Cloud Platforms for ML Training

| Platform | GPU Support | Pricing | Best For |
|----------|-------------|---------|----------|
| **[RunPod](https://www.runpod.io)** | RTX 3090, A100, H100 | ~$0.20-0.50/jam | ⭐ Serverless GPU on-demand, API integration |
| **[Lambda Labs](https://lambdalabs.com)** | A100, H100 | ~$1.10/jam A100 | Continuous training workloads |
| **[Vast.ai](https://vast.ai)** | RTX 3090, A6000 | ~$0.15-0.30/jam | 💰 Termurah, peer-to-peer marketplace |
| **[AWS EC2](https://aws.amazon.com/ec2)** | T4, V100, A100 | Spot: 70% off | Enterprise dengan existing infrastructure |
| **[Google Cloud](https://cloud.google.com)** | T4, V100, A100 | Flexible | Vertex AI integration, auto-scaling |
| **[Paperspace](https://www.paperspace.com)** | RTX 5000, A6000 | Free tier avail | Managed notebooks + API |

### 🏗️ Hybrid Architecture (Recommended)

```
┌─────────────────────────────────────────┐
│  Frontend (React) + Backend (FastAPI)   │
│         ↓ Railway / Render              │  ← Web tier
│         ↓ MongoDB Atlas / Redis Cloud   │  ← Managed DB
└─────────────────────────────────────────┘
                   │
                   ↓ Webhook/Queue
┌─────────────────────────────────────────┐
│     Training Worker (RunPod/Vast.ai)    │
│     - Docker dengan NVIDIA runtime      │  ← GPU tier
│     - GPU 8GB+ untuk QLoRA              │
│     - S3/GCS untuk model storage        │
└─────────────────────────────────────────┘
```

### 💰 Cost Estimation (Monthly)

| Component | Platform | Estimated Cost |
|-----------|----------|----------------|
| Web Tier | Railway | $20-50 |
| Training GPU | RunPod (20 jam) | $100-200 |
| Database | MongoDB Atlas (M10) | $60 |
| Storage | AWS S3 (100GB) | $5 |
| **Total** | | **~$185-315/bulan** |

### 🚀 Quick Start Cloud Setup

#### 1. Web Tier (Railway)
```bash
# Deploy frontend + backend
curl -fsSL https://railway.app/install.sh | sh
railway login
railway init
railway up
```

#### 2. Training Workers (RunPod)
```python
# Integrasi dengan RunPod Serverless API
# Lihat: backend/core/training_engine.py

import requests

def start_training_on_runpod(config):
    response = requests.post(
        "https://api.runpod.io/v2/your-endpoint/run",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        json={"input": config}
    )
    return response.json()
```

#### 3. Database (MongoDB Atlas)
```env
# .env
DATABASE_URL=mongodb+srv://user:pass@cluster.mongodb.net/qlora_db
```

### ⚠️ Platform Limitations

| Platform | GPU | Not Suitable Because |
|----------|-----|---------------------|
| **Railway** | ❌ | No GPU support |
| **Render** | ❌ | No GPU support |
| **Vercel** | ❌ | Serverless functions only |
| **Heroku** | ❌ | No GPU, limited memory |

### 📋 System Requirements for Cloud

**Minimum untuk QLoRA Training:**
- **GPU**: NVIDIA dengan 8GB+ VRAM (T4, RTX 3080, A100)
- **Memory**: 16GB+ RAM
- **Storage**: 100GB+ SSD untuk model checkpoints
- **Network**: Stable connection untuk HuggingFace model download

**Recommended:**
- **GPU**: RTX 3090 (24GB) atau A100 (40GB)
- **Memory**: 32GB+ RAM
- **Storage**: 500GB+ NVMe SSD

## 🧪 Testing

### Backend Testing
```bash
cd backend
pytest tests/ -v
```

### Frontend Testing
```bash
cd frontend
npm test
```

### Integration Testing
```bash
# Test API endpoints
cd backend
pytest tests/test_integration.py -v

# Test WebSocket connections
pytest tests/test_websocket.py -v
```

## 📁 Struktur Proyek

```
qlora/
├── backend/
│   ├── core/                    # Core modules
│   │   ├── training_engine.py   # Enhanced training engine
│   │   ├── enhanced_gpu_manager.py    # GPU management
│   │   ├── enhanced_data_processor.py # Data processing
│   │   ├── enhanced_training_callbacks.py # Training callbacks
│   │   ├── enhanced_auth.py     # Authentication
│   │   ├── enhanced_logging.py  # Structured logging
│   │   └── environment_validator.py # Environment validation
│   ├── api/                     # API endpoints
│   ├── models/                  # Data models
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # Docker configuration
│   └── .env.example            # Environment template
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── RealTimeMonitoringDashboard.tsx
│   │   │   └── EnhancedTrainingConfig.tsx
│   │   ├── pages/              # Page components
│   │   ├── hooks/              # Custom hooks
│   │   └── lib/                # Utilities
│   ├── public/                 # Static assets
│   ├── Dockerfile              # Docker configuration
│   └── package.json            # Node dependencies
├── docker/                     # Docker configurations
│   ├── mongodb/init.js         # Database initialization
│   ├── prometheus/             # Prometheus config
│   └── grafana/                # Grafana dashboards
├── tests/                      # Test files
├── deploy.sh                   # Deployment script
├── docker-compose.yml        # Docker services config
├── DEPLOYMENT.md             # Deployment guide
└── README.md                 # Documentation
```

## 🔍 Monitoring & Maintenance

### Health Checks
```bash
# Check service health
./deploy.sh health

# View service status
./deploy.sh status

# Check logs
./deploy.sh logs backend
./deploy.sh logs frontend
```

### Performance Monitoring
- **Grafana Dashboards**: Pre-configured dashboards untuk system dan application metrics
- **Prometheus Metrics**: Real-time performance data collection
- **Custom Alerts**: Setup alerts untuk critical metrics

### Backup & Recovery
```bash
# Database backup
docker-compose exec mongodb mongodump --out /backup --gzip

# Model backup
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/ checkpoints/
```

## 🔍 Troubleshooting

### Masalah Umum

1. **GPU Not Detected**
   - Pastikan NVIDIA Docker Runtime terinstall
   - Cek GPU driver compatibility
   - Verifikasi GPU memory availability

2. **Database Connection Failed**
   - Verifikasi MongoDB credentials
   - Cek network connectivity
   - Review database logs

3. **Out of Memory Errors**
   - Kurangi batch size dalam training configuration
   - Adjust GPU memory threshold
   - Monitor system memory usage

4. **Model Download Issues**
   - Cek Hugging Face token validity
   - Verifikasi internet connectivity
   - Review disk space availability

### Log Analysis
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs backend
docker-compose logs mongodb

# Follow logs in real-time
docker-compose logs -f backend
```

## 🤝 Kontribusi

1. Fork repository
2. Buat branch fitur baru (`git checkout -b feature/amazing-feature`)
3. Commit perubahan (`git commit -m 'Add some amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Buat Pull Request

## 📄 Lisensi

Proyek ini dilisensikan under MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 🌟 Showcase

Platform ini telah berhasil digunakan untuk:
- Fine-tuning model bahasa Indonesia untuk customer service
- Adaptasi model medis untuk diagnosis support
- Training model code generation untuk development tools
- Customization model legal untuk document analysis

---

**⭐ Note**: Aplikasi ini dilengkapi dengan **Enhanced Training Engine** yang mendukung training QLoRA nyata dengan GPU acceleration, monitoring real-time, dan enterprise-grade security features. Sistem otomatis mengaktifkan mode simulasi hanya jika hardware tidak memadai, namun tetap memberikan pengalaman UI/UX yang optimal.

**🚀 Ready for Production**: Platform ini siap untuk deployment production dengan Docker containerization, monitoring stack, dan comprehensive security features.