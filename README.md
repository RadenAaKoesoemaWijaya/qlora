# QLoRA Fine-tuning Platform

Aplikasi web-based untuk fine-tuning Large Language Models (LLM) menggunakan metode QLoRA (Quantized Low-Rank Adaptation). Platform ini menyediakan antarmuka yang user-friendly untuk melakukan fine-tuning model AI secara efisien dengan teknik kuantisasi.

## 🚀 Fitur Utama

### 1. **Model Selection**
- Pilihan model pre-trained: Llama 2, Llama 3, Mistral, Gemma, Mixtral
- Informasi detail tentang ukuran model dan karakteristik
- Support untuk model 7B hingga 47B parameters

### 2. **Dataset Management**
- Upload dataset dalam format JSON, CSV, atau JSONL
- Validasi dataset otomatis
- Manajemen dataset dengan informasi jumlah baris dan ukuran file

### 3. **Training Configuration**
- Konfigurasi parameter QLoRA (rank, alpha, dropout)
- Pengaturan learning rate, batch size, dan epochs
- Custom target modules untuk adaptation
- Support GPU acceleration

### 4. **Training Monitor**
- Real-time monitoring progress training
- Visualisasi loss dan learning rate
- Checkpoint management otomatis
- Estimasi waktu completion

### 5. **Evaluation & Analytics**
- Evaluasi model dengan metrik: accuracy, perplexity, F1-score
- Perbandingan performa model
- History training dan evaluasi

## 🏗️ Arsitektur Sistem

### Backend (Python/FastAPI)
- **Framework**: FastAPI untuk high-performance REST API
- **Database**: MongoDB untuk penyimpanan data
- **Arsitektur**: Microservices dengan async/await pattern
- **Features**: Simulated training engine, automatic checkpointing

### Frontend (React)
- **Framework**: React 19 dengan React Router
- **UI Components**: Radix UI + Tailwind CSS
- **State Management**: React Hooks
- **Build Tool**: CRACO untuk konfigurasi custom

## 📋 Prasyarat

### Backend Requirements
- Python 3.8+
- MongoDB 4.4+
- pip package manager

### Frontend Requirements
- Node.js 16+
- Yarn package manager
- Modern web browser

## 🛠️ Instalasi dan Setup

### 1. Clone Repository
```bash
git clone [repository-url]
cd qlora
```

### 2. Setup Backend
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Setup environment
# Edit file .env sesuai kebutuhan
# Default MongoDB: mongodb://localhost:27017

# Jalankan server
python server.py
```

### 3. Setup Frontend
```bash
cd frontend

# Install dependencies
yarn install

# Setup environment
# Edit file .env untuk backend URL
# Default: REACT_APP_BACKEND_URL=http://localhost:8000

# Jalankan development server
yarn start
```

## 🔧 Konfigurasi

### Backend Configuration (.env)
```env
MONGO_URL="mongodb://localhost:27017"
DB_NAME="test_database"
CORS_ORIGINS="*"
```

### Frontend Configuration (.env)
```env
REACT_APP_BACKEND_URL=http://localhost:8000
WDS_SOCKET_PORT=443
ENABLE_HEALTH_CHECK=false
```

## 🎯 Mekanisme Kerja

### 1. **Proses Fine-tuning**
1. User memilih model base (Llama, Mistral, dll)
2. Upload dataset training dalam format yang didukung
3. Konfigurasi parameter QLoRA sesuai kebutuhan
4. Start training job dengan monitoring real-time
5. Sistem otomatis membuat checkpoint setiap epoch
6. Evaluasi model setelah training selesai

### 2. **QLoRA Implementation**
- **Quantization**: Model base dikonversi ke 4-bit precision
- **Low-Rank Adaptation**: Menambah adapter layers dengan rank rendah
- **Memory Efficient**: Mengurangi memory usage hingga 75%
- **Performance**: Mempertahankan kualitas model original

### 3. **Training Pipeline**
```
Dataset Upload → Validation → Model Selection → 
Parameter Configuration → Training Start → 
Progress Monitoring → Checkpoint Creation → 
Model Evaluation → Result Storage
```

## 📊 API Endpoints

### Models
- `GET /api/models` - List available models
- `GET /api/models/{id}` - Get model details

### Datasets
- `POST /api/datasets/upload` - Upload dataset
- `GET /api/datasets` - List datasets
- `DELETE /api/datasets/{id}` - Delete dataset

### Training
- `POST /api/training/start` - Start training job
- `GET /api/training/jobs` - List training jobs
- `GET /api/training/jobs/{id}` - Get job details
- `POST /api/training/jobs/{id}/stop` - Stop training

### Checkpoints
- `GET /api/checkpoints` - List checkpoints
- `DELETE /api/checkpoints/{id}` - Delete checkpoint

### Evaluation
- `POST /api/evaluate` - Evaluate model
- `GET /api/evaluations` - List evaluations

## 🚀 Menjalankan Aplikasi

### Mode Development
```bash
# Terminal 1 - Backend
cd backend && python server.py

# Terminal 2 - Frontend
cd frontend && yarn start
```

### Mode Production
```bash
# Build frontend
cd frontend && yarn build

# Deploy backend with production server
# Gunakan Uvicorn dengan workers
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🧪 Testing

### Backend Testing
```bash
cd backend
pytest tests/
```

### Frontend Testing
```bash
cd frontend
yarn test
```

## 📁 Struktur Proyek

```
qlora/
├── backend/
│   ├── server.py          # FastAPI server
│   ├── requirements.txt   # Python dependencies
│   └── .env              # Environment variables
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/        # Page components
│   │   ├── hooks/        # Custom hooks
│   │   └── lib/          # Utilities
│   ├── public/           # Static assets
│   └── package.json      # Node dependencies
├── tests/                # Test files
└── README.md            # Documentation
```

## 🔍 Troubleshooting

### Masalah Umum

1. **MongoDB Connection Failed**
   - Pastikan MongoDB service running
   - Cek connection string di .env

2. **CORS Error**
   - Cek CORS_ORIGINS di backend .env
   - Pastikan frontend URL diizinkan

3. **Training Job Stuck**
   - Cek log di terminal backend
   - Restart training job jika perlu

4. **Frontend Build Error**
   - Clear node_modules dan reinstall
   - Cek Node.js version compatibility

## 🤝 Kontribusi

1. Fork repository
2. Buat branch fitur baru
3. Commit perubahan
4. Push ke branch
5. Buat Pull Request

## 📄 Lisensi

[License information here]

## 📞 Support

Untuk pertanyaan dan bantuan:
- Email: [support email]
- Issues: [GitHub issues]
- Documentation: [Wiki/Documentation]

---

**Note**: Aplikasi ini menggunakan simulated training engine untuk demo dan development. Untuk implementasi training asli, integrasi dengan framework seperti Hugging Face Transformers atau PyTorch diperlukan.