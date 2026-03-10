# 📊 QLoRA Platform - Comprehensive Workflow Architecture

Dokumen ini menyajikan alur kerja komprehensif platform QLoRA Fine-tuning untuk keperluan pembuatan skema arsitektur lengkap.

---

## 🏗️ 1. OVERALL SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Web Browser │  │  CLI Tool   │  │  API Client │  │ Mobile App  │          │
│  │  (React)    │  │  (Python)   │  │  (cURL)     │  │  (Future)   │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
└─────────┼────────────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    FastAPI Application                             │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │       │
│  │  │  CORS    │ │  JWT     │ │ Rate     │ │ Request  │            │       │
│  │  │  Middleware│ │  Auth    │ │ Limiting │ │ Logging  │            │       │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │       │
│  └─────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BUSINESS LOGIC LAYER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │
│  │ Training Engine │  │  Data Processor │  │  GPU Manager    │               │
│  │    Factory      │  │                 │  │                 │               │
│  │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │               │
│  │  │ QLoRA     │  │  │  │  JSON     │  │  │  │ Health    │  │               │
│  │  │ DoRA      │  │  │  │  JSONL    │  │  │  │ Monitor   │  │               │
│  │  │ IA3       │  │  │  │  CSV      │  │  │  │ Selection │  │               │
│  │  │ VeRA      │  │  │  │  TXT      │  │  │  │ Allocation│  │               │
│  │  │ LoRA+     │  │  │  │  Parquet  │  │  │  └───────────┘  │               │
│  │  │ AdaLoRA   │  │  │  │  XLSX     │  │  └─────────────────┘               │
│  │  │ OFT       │  │  │  └───────────┘  │                                    │
│  │  └───────────┘  │  └─────────────────┘                                    │
│  └─────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA PERSISTENCE LAYER                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │
│  │   MongoDB       │  │    Redis        │  │   File System   │               │
│  │  (Primary DB)   │  │   (Cache)       │  │  (Datasets/     │               │
│  │                 │  │                 │  │   Models)       │               │
│  │ • training_jobs │  │ • session data  │  │                 │               │
│  │ • datasets      │  │ • API responses │  │ • /datasets     │               │
│  │ • checkpoints   │  │ • rate limits   │  │ • /models       │               │
│  │ • metrics       │  │ • pub/sub       │  │ • /checkpoints  │               │
│  │ • evaluations   │  │                 │  │                 │               │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INFRASTRUCTURE LAYER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │
│  │  Docker         │  │  Prometheus     │  │   Grafana       │               │
│  │  Containers     │  │  Metrics        │  │   Dashboard     │               │
│  │                 │  │                 │  │                 │               │
│  │ • Backend       │  │ • System        │  │ • Monitoring    │               │
│  │ • Frontend      │  │ • Application   │  │ • Alerts        │               │
│  │ • MongoDB       │  │ • Custom        │  │ • Visualization │               │
│  │ • Redis         │  │                 │  │                 │               │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 2. END-TO-END USER WORKFLOW

### 2.1 User Registration & Authentication Flow

```
┌────────┐     ┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│  User  │     │   Frontend  │     │  Backend API    │     │  MongoDB    │
└───┬────┘     └──────┬──────┘     └────────┬────────┘     └──────┬──────┘
    │                 │                     │                     │
    │ 1. Register     │                     │                     │
    │────────────────>│                     │                     │
    │                 │ 2. POST /api/auth/  │                     │
    │                 │    register         │                     │
    │                 │────────────────────>│                     │
    │                 │                     │ 3. Validate input   │
    │                 │                     │    Hash password  │
    │                 │                     │                     │
    │                 │                     │ 4. Insert user      │
    │                 │                     │────────────────────>│
    │                 │                     │                     │
    │                 │ 5. Return JWT       │                     │
    │                 │<────────────────────│                     │
    │                 │                     │                     │
    │ 6. Store token  │                     │                     │
    │<────────────────│                     │                     │
```

### 2.2 Dataset Upload & Validation Flow

```
┌────────┐     ┌─────────────┐     ┌─────────────────┐     ┌─────────────┐     ┌─────────────┐
│  User  │     │   Frontend  │     │  Backend API    │     │  Validation │     │  MongoDB    │
└───┬────┘     └──────┬──────┘     └────────┬────────┘     └──────┬──────┘     └──────┬──────┘
    │                 │                     │                     │                     │
    │ 1. Upload file  │                     │                     │                     │
    │────────────────>│                     │                     │                     │
    │                 │ 2. POST /api/       │                     │                     │
    │                 │    datasets/upload  │                     │                     │
    │                 │────────────────────>│                     │                     │
    │                 │                     │ 3. Security check   │                     │
    │                 │                     │    validate_path()│                     │
    │                 │                     │                     │                     │
    │                 │                     │ 4. File type check  │                     │
    │                 │                     │    (JSON/JSONL    │                     │
    │                 │                     │     CSV/TXT/XLSX  │                     │
    │                 │                     │     Parquet)        │                     │
    │                 │                     │                     │                     │
    │                 │                     │ 5. Async process  │                     │
    │                 │                     │────────────────────>│                     │
    │                 │                     │                     │ 6. Parse content  │
    │                 │                     │                     │    Count rows     │
    │                 │                     │                     │    Validate format│
    │                 │                     │<────────────────────│                     │
    │                 │                     │                     │                     │
    │                 │                     │ 7. Save to FS       │                     │
    │                 │                     │    Save metadata    │                     │
    │                 │                     │───────────────────────────────────────────>│
    │                 │                     │                     │                     │
    │                 │ 8. Return dataset   │                     │                     │
    │                 │    info             │                     │                     │
    │                 │<────────────────────│                     │                     │
    │                 │                     │                     │                     │
    │ 9. Show success │                     │                     │                     │
    │<────────────────│                     │                     │                     │
```

---

## 🎯 3. TRAINING PIPELINE WORKFLOW

### 3.1 Training Job Lifecycle

```
┌───────────┐
│   START   │
└─────┬─────┘
      │
      ▼
┌───────────────┐
│  INITIALIZING │ ←── Validation config, check resources
└───────┬───────┘
      │
      ▼
┌───────────────┐
│   QUEUED      │ ←── Add to job queue
└───────┬───────┘
      │
      ▼
┌───────────────┐
│   TRAINING    │ ←── Active training with progress updates
│   ┌─────────┐ │
│   │ Epoch 1 │ │
│   │ Epoch 2 │ │
│   │ Epoch N │ │
│   └─────────┘ │
└───────┬───────┘
      │
      ├──────────┐
      │          │
      ▼          ▼
┌──────────┐ ┌──────────┐
│ COMPLETED│ │  FAILED  │
└────┬─────┘ └────┬─────┘
     │            │
     ▼            ▼
┌──────────┐ ┌──────────┐
│ Save     │ │ Cleanup  │
│ checkpoint│ │  resources│
│ Cleanup  │ │ Log error │
└──────────┘ └──────────┘
```

### 3.2 Detailed Training Execution Flow

```
┌─────────┐   ┌─────────────┐   ┌─────────────────┐   ┌─────────────┐   ┌─────────────┐
│  User   │   │   Backend   │   │  Training Engine│   │  GPU Manager│   │   MongoDB   │
└───┬─────┘   └──────┬──────┘   └────────┬────────┘   └──────┬──────┘   └──────┬──────┘
    │                │                     │                   │                 │
    │ 1. POST /api/  │                     │                   │                 │
    │    training/   │                     │                   │                 │
    │    start       │                     │                   │                 │
    │───────────────>│                     │                   │                 │
    │                │                     │                   │                 │
    │                │ 2. Validate TrainingConfig                  │                 │
    │                │    (9 validators)   │                   │                 │
    │                │─────────────────────>│                   │                 │
    │                │                     │                   │                 │
    │                │ 3. Create job       │                   │                 │
    │                │    Insert to DB     │                   │                 │
    │                │───────────────────────────────────────────────────────────>│
    │                │                     │                   │                 │
    │                │ 4. Check GPU          │                   │                 │
    │                │    Select optimal   │                   │                 │
    │                │───────────────────────────────────────>│                 │
    │                │                     │                   │                 │
    │                │ 5. Allocate GPU       │                   │                 │
    │                │<───────────────────────────────────────│                 │
    │                │                     │                   │                 │
    │                │ 6. Load dataset       │                   │                 │
    │                │    (async with      │                   │                 │
    │                │     5min timeout)   │                   │                 │
    │                │─────────────────────>│                   │                 │
    │                │                     │                   │                 │
    │                │ 7. Initialize Engine  │                   │                 │
    │                │    (Factory pattern)│                   │                 │
    │                │    • QLoRA/DoRA/etc │                   │                 │
    │                │─────────────────────>│                   │                 │
    │                │                     │                   │                 │
    │                │ 8. Start training     │                   │                 │
    │                │    loop             │                   │                 │
    │                │─────────────────────>│                   │                 │
    │                │                     │                   │                 │
    │                │ 9. Update progress  │                   │                 │
    │                │    (WebSocket +     │                   │                 │
    │                │     DB writes)      │                   │                 │
    │                │───────────────────────────────────────────────────────────>│
    │                │                     │                   │                 │
    │ 10. Real-time  │                     │                   │                 │
    │     updates    │<────────────────────│                   │                 │
    │<───────────────│                     │                   │                 │
    │                │                     │                   │                 │
    │                │ 11. Epoch complete  │                   │                 │
    │                │    Save checkpoint  │                   │                 │
    │                │    (if configured)  │                   │                 │
    │                │───────────────────────────────────────────────────────────>│
    │                │                     │                   │                 │
    │                │ 12. Repeat or       │                   │                 │
    │                │     finalize        │                   │                 │
    │                │                     │                   │                 │
    │                │ 13. Cleanup         │                   │                 │
    │                │    engine.cleanup() │                   │                 │
    │                │    GPU release      │                   │                 │
    │                │───────────────────────────────────────>│                 │
    │                │                     │                   │                 │
    │                │ 14. Update status   │                   │                 │
    │                │    (completed/    │                   │                 │
    │                │     failed)         │                   │                 │
    │                │───────────────────────────────────────────────────────────>│
    │                │                     │                   │                 │
    │ 15. Final     │                     │                   │                 │
    │     response   │<────────────────────│                   │                 │
    │<───────────────│                     │                   │                 │
```

---

## 🛡️ 4. SECURITY LAYER WORKFLOW

### 4.1 Request Security Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INCOMING REQUEST                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. CORS MIDDLEWARE                                                          │
│    • Validate Origin header                                                   │
│    • Check against CORS_ORIGINS whitelist                                   │
│    • Reject if not in allowed origins                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. RATE LIMITING (Redis-backed)                                             │
│    • Check request count per IP/API key                                     │
│    • Reject if exceeds limit (429 Too Many Requests)                        │
│    • Increment counter with expiry                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. JWT AUTHENTICATION (if required)                                         │
│    • Extract Bearer token from Authorization header                       │
│    • Verify signature and expiry                                            │
│    • Decode payload (user_id, role, permissions)                            │
│    • Reject if invalid (401 Unauthorized)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. INPUT VALIDATION (Pydantic)                                              │
│    • TrainingConfig: 9 validators                                             │
│      - lora_rank: 1-1024                                                    │
│      - learning_rate: 1e-6 to 1e-2                                          │
│      - num_epochs: 1-100                                                    │
│      - batch_size: 1-128                                                    │
│      - max_seq_length: 64-8192                                            │
│      - training_method: whitelist                                           │
│      - lora_dropout: 0.0-1.0                                                │
│    • Reject with detailed error message (422 Unprocessable Entity)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. PATH VALIDATION (File Operations)                                        │
│    • validate_dataset_path()                                                │
│    • Resolve to absolute path                                               │
│    • Verify within allowed base directory                                   │
│    • Reject path traversal attempts (400 Bad Request)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. FILENAME SANITIZATION                                                    │
│    • Remove path separators                                                 │
│    • Strip leading dots (hidden files)                                      │
│    • Limit length to 255 chars                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. AUTHORIZATION (Role-based)                                               │
│    • Check user role against required permissions                           │
│    • admin: all operations                                                    │
│    • trainer: training + datasets                                           │
│    • viewer: read-only                                                        │
│    • Reject if insufficient (403 Forbidden)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROCESS REQUEST                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ 5. CACHING WORKFLOW

### 5.1 Cache Read/Write Flow

```
┌─────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Request │   │  Cache      │   │   Redis     │   │  Backend    │
└───┬─────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
    │                │                 │                 │
    │ 1. Request     │                 │                 │
    │    data        │                 │                 │
    │───────────────>│                 │                 │
    │                │                 │                 │
    │                │ 2. Generate     │                 │
    │                │    cache key    │                 │
    │                │    hash(func+args)                │
    │                │                 │                 │
    │                │ 3. Check Redis  │                 │
    │                │────────────────>│                 │
    │                │                 │                 │
    │                │ 4. Cache Hit?   │                 │
    │                │<────────────────│                 │
    │                │                 │                 │
    │                │ ┌─────────────┐ │                 │
    │                │ │ YES: Return │ │                 │
    │                │ │ cached data │ │                 │
    │                │ │─────────────>│                 │
    │ 5. Return      │ └─────────────┘ │                 │
    │    cached      │                 │                 │
    │<───────────────│                 │                 │
    │                │                 │                 │
    │                │ ┌─────────────┐ │                 │
    │                │ │ NO: Check   │ │                 │
    │                │ │ memory cache│ │                 │
    │                │ │─────────────>│                 │
    │                │ └─────────────┘ │                 │
    │                │                 │                 │
    │                │ 6. Memory Hit?  │                 │
    │                │ ┌─────────────┐ │                 │
    │                │ │ YES: Return │ │                 │
    │                │ └─────────────┘ │                 │
    │                │                 │                 │
    │                │ ┌─────────────┐ │                 │
    │                │ │ NO: Call    │ │                 │
    │                │ │ backend     │ │                 │
    │                │ │─────────────>│                 │
    │                │ └─────────────┘ │                 │
    │                │                 │ 7. Execute query│
    │                │                 │────────────────>│
    │                │                 │                 │
    │                │                 │ 8. Return result│
    │                │                 │<────────────────│
    │                │                 │                 │
    │                │ 9. Store in     │                 │
    │                │    both caches  │                 │
    │                │    (Redis + Mem)│                 │
    │                │────────────────>│                 │
    │                │                 │                 │
    │ 10. Return     │                 │                 │
    │     fresh data │<────────────────│                 │
    │<───────────────│                 │                 │
```

---

## 📊 6. MONITORING & OBSERVABILITY WORKFLOW

### 6.1 Metrics Collection Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Training  │  │    API      │  │   Database  │  │    GPU      │          │
│  │   Engine    │  │  Endpoints  │  │   Queries   │  │   Manager   │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
└─────────┼────────────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │                │
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STRUCTURED LOGGING                                     │
│  • Log level (DEBUG, INFO, WARNING, ERROR)                                    │
│  • Timestamp dengan timezone                                                │
│  • Correlation ID untuk request tracking                                    │
│  • Context (user_id, job_id, operation)                                     │
│  • Performance metrics (duration, memory)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ├────────────────────────────────────────┐
          │                                        │
          ▼                                        ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│     Prometheus          │      │      Grafana            │
│     Metrics             │      │      Dashboard          │
│                         │      │                         │
│ • training_jobs_total   │      │ • Real-time charts      │
│ • active_training_jobs  │      │ • Historical trends     │
│ • gpu_utilization       │      │ • Alert panels          │
│ • request_duration      │      │ • Performance metrics   │
│ • error_rate            │      │ • System health         │
└─────────────────────────┘      └─────────────────────────┘
```

### 6.2 Real-time WebSocket Updates

```
┌─────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────┐
│ Client  │     │  WebSocket  │     │  Training   │     │  Redis  │
│ Browser │     │   Server    │     │   Engine    │     │ Pub/Sub │
└───┬─────┘     └──────┬──────┘     └──────┬──────┘     └────┬────┘
    │                  │                   │                 │
    │ 1. Connect       │                   │                 │
    │─────────────────>│                   │                 │
    │                  │                   │                 │
    │ 2. Subscribe to  │                   │                 │
    │    job updates   │                   │                 │
    │─────────────────>│                   │                 │
    │                  │                   │                 │
    │                  │ 3. Listen for     │                 │
    │                  │    training       │                 │
    │                  │    events         │                 │
    │                  │<─────────────────│                 │
    │                  │                   │                 │
    │                  │                   │ 4. Training     │
    │                  │                   │    progress     │
    │                  │                   │    update       │
    │                  │                   │                 │
    │                  │                   │ 5. Publish      │
    │                  │                   │    event        │
    │                  │                   │───────────────>│
    │                  │                   │                 │
    │                  │ 6. Receive        │                 │
    │                  │    event          │                 │
    │                  │<─────────────────────────────────────│
    │                  │                   │                 │
    │ 7. Push update   │                   │                 │
    │<─────────────────│                   │                 │
    │                  │                   │                 │
    │ 8. Update UI     │                   │                 │
    │    (progress bar,│                   │                 │
    │     metrics)     │                   │                 │
```

---

## 🔧 7. ERROR HANDLING & RECOVERY WORKFLOW

### 7.1 Error Recovery Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ERROR DETECTED                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │   1. CLASSIFY ERROR      │
                    │                          │
                    │ • ValidationError        │
                    │ • TimeoutError           │
                    │ • GPUOutOfMemoryError    │
                    │ • DatabaseError          │
                    │ • FileNotFoundError      │
                    │ • TrainingError          │
                    └───────────┬──────────────┘
                                │
               ┌────────────────┼────────────────┐
               │                │                │
               ▼                ▼                ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │ RETRYABLE? │  │ RETRYABLE? │  │ FATAL?     │
        │ (Timeout)  │  │ (GPU OOM)  │  │ (Validation│
        └──────┬─────┘  └──────┬─────┘  │  Error)    │
               │               │        └──────┬─────┘
               │               │               │
        ┌──────┴─────┐  ┌──────┴─────┐  ┌──────┴─────┐
        │ YES        │  │ YES        │  │ NO         │
        │            │  │ (with      │  │            │
        │ Exponential│  │ adjusted   │  │ Immediate  │
        │ backoff    │  │ params)    │  │ failure    │
        │ retry      │  │            │  │            │
        │ (max 3)    │  │ Reduce     │  │            │
        │            │  │ batch size │  │            │
        └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
               │               │               │
               ▼               ▼               ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │ SUCCESS?   │  │ SUCCESS?   │  │ Log error  │
        └──────┬─────┘  └──────┬─────┘  │ Cleanup    │
               │               │        │ Update DB  │
          ┌────┴────┐     ┌────┴────┐  │ Notify user│
          │ YES     │     │ YES     │  └────────────┘
          │         │     │         │
          ▼         │     ▼         │
    ┌─────────┐     │ ┌─────────┐   │
    │ Continue│     │ │ Continue│   │
    │ normally│     │ │ with new│   │
    └─────────┘     │ │ params  │   │
                    │ └─────────┘   │
          ┌─────────┘               └─────────┐
          │ NO                      NO        │
          ▼                                   ▼
    ┌────────────┐                    ┌────────────┐
    │ Mark as    │                    │ Mark as    │
    │ FAILED     │                    │ FAILED     │
    │ Cleanup    │                    │ Cleanup    │
    │ resources  │                    │ resources  │
    │ Log final  │                    │ Log final  │
    │ error      │                    │ error      │
    └────────────┘                    └────────────┘
```

---

## 🗄️ 8. DATA MODEL RELATIONSHIPS

### 8.1 Entity Relationship Diagram (Conceptual)

```
┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│      USER           │       │   TRAINING_JOB      │       │     DATASET         │
├─────────────────────┤       ├─────────────────────┤       ├─────────────────────┤
│ _id                 │       │ _id                 │       │ _id                 │
│ id (UUID)           │       │ id (UUID)           │       │ id (UUID)           │
│ email               │       │ user_id ────────────┼──────>│ user_id             │
│ password_hash       │       │ model_name          │       │ name                │
│ role                │       │ dataset_id ─────────┼──────>│ file_path           │
│ api_key             │       │ status              │       │ file_type           │
│ created_at          │       │ training_method     │       │ size                │
│ last_login          │       │ config (JSON)       │       │ rows                │
└─────────────────────┘       │ progress            │       │ validation_status   │
          │                   │ current_epoch       │       │ created_at          │
          │                   │ total_epochs        │       └─────────────────────┘
          │                   │ current_loss        │
          │                   │ learning_rate       │
          │                   │ started_at          │
          │                   │ completed_at        │
          │                   │ model_path          │
          │                   │ error               │
          │                   │ created_at          │
          │                   └─────────────────────┘
          │                             │
          │                             │
          │                             │ 1:N
          │                             ▼
          │                   ┌─────────────────────┐
          │                   │   TRAINING_METRIC   │
          │                   ├─────────────────────┤
          │                   │ _id                 │
          │                   │ job_id              │
          │                   │ epoch               │
          │                   │ step                │
          │                   │ loss                │
          │                   │ learning_rate       │
          │                   │ timestamp           │
          │                   └─────────────────────┘
          │
          │ 1:N
          ▼
┌─────────────────────┐
│     CHECKPOINT      │
├─────────────────────┤
│ _id                 │
│ id (UUID)           │
│ training_job_id     │
│ epoch               │
│ step                │
│ loss                │
│ model_path          │
│ size_mb             │
│ created_at          │
└─────────────────────┘
          ▲
          │
          │ 1:N
          │
┌─────────────────────┐
│   EVALUATION_RESULT │
├─────────────────────┤
│ _id                 │
│ id (UUID)           │
│ model_id            │
│ checkpoint_id       │
│ accuracy            │
│ perplexity          │
│ f1_score            │
│ bert_score          │
│ rouge_l             │
│ bleu_score          │
│ evaluated_at        │
│ details (JSON)      │
└─────────────────────┘
```

---

## 🐳 9. DEPLOYMENT ARCHITECTURE

### 9.1 Docker Compose Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOCKER NETWORK: qlora-net                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   qlora-frontend    │  │   qlora-backend     │  │   qlora-mongodb     │
│   (Node.js/Nginx)   │  │   (Python/FastAPI)  │  │   (MongoDB 6.0)     │
│                     │  │                     │  │                     │
│ Port: 80            │  │ Port: 8000          │  │ Port: 27017         │
│                     │  │                     │  │                     │
│ • React SPA         │  │ • FastAPI app       │  │ • Training jobs     │
│ • Nginx proxy       │  │ • Uvicorn server    │  │ • Datasets          │
│ • Static assets     │  │ • Python 3.11       │  │ • Checkpoints       │
│                     │  │ • ML Libraries      │  │ • Metrics           │
│ Volumes:            │  │                     │  │ • Evaluations       │
│ - frontend/build    │  │ Environment:        │  │                     │
│                     │  │ - MONGO_URL         │  │ Volumes:            │
│ Depends on:         │  │ - REDIS_URL         │  │ - mongodb_data      │
│ - backend           │  │ - GPU access        │  │                     │
│                     │  │                     │  │                     │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   qlora-redis       │  │   qlora-prometheus  │  │   qlora-grafana     │
│   (Redis 7)         │  │   (Prometheus)      │  │   (Grafana)         │
│                     │  │                     │  │                     │
│ Port: 6379          │  │ Port: 9090          │  │ Port: 3000          │
│                     │  │                     │  │                     │
│ • Session store     │  │ • Metrics collection│  │ • Dashboards        │
│ • Caching           │  │ • Time-series DB    │  │ • Visualization     │
│ • Pub/Sub           │  │ • Alert rules       │  │ • Alerts            │
│ • Rate limiting     │  │                     │  │                     │
│                     │  │ Targets:            │  │ Data source:        │
│ Volumes:            │  │ - backend:8000      │  │ - Prometheus        │
│ - redis_data        │  │ - cadvisor          │  │                     │
│                     │  │ - node-exporter     │  │ Volumes:            │
│                     │  │                     │  │ - grafana_data      │
│                     │  │ Config:             │  │ - dashboards        │
│                     │  │ - prometheus.yml    │  │                     │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

---

## 📋 10. API ENDPOINT FLOW SUMMARY

### 10.1 Request-Response Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API ROUTES (api_router)                           │
│                         Prefix: /api                                        │
└─────────────────────────────────────────────────────────────────────────────┘

AUTHENTICATION:
├── POST /api/auth/register          → Create user account
├── POST /api/auth/login             → Authenticate, return JWT
├── POST /api/auth/refresh           → Refresh access token
└── POST /api/auth/logout            → Invalidate token

MODELS:
└── GET  /api/models                 → List available base models

DATASETS:
├── POST /api/datasets/upload        → Upload dataset file
├── GET  /api/datasets               → List user datasets
└── DELETE /api/datasets/{id}        → Remove dataset

TRAINING:
├── POST /api/training/start         → Start new training job
├── GET  /api/training/jobs          → List all jobs
├── GET  /api/training/jobs/{id}     → Get job details
├── POST /api/training/jobs/{id}/stop → Stop running job
└── GET  /api/training/jobs/{id}/metrics → Get training metrics

CHECKPOINTS:
├── GET  /api/checkpoints            → List checkpoints
└── DELETE /api/checkpoints/{id}     → Delete checkpoint

EVALUATION:
├── POST /api/evaluate               → Run model evaluation
└── GET  /api/evaluations            → List evaluation results

DASHBOARD:
├── GET  /api/dashboard/stats        → Get dashboard statistics
├── GET  /api/training/methods       → List available methods
├── GET  /api/training/methods/{id}/config-schema → Method config
└── GET  /api/training/methods/{id}/recommendations → Method guide
```

---

## 🎯 KEY WORKFLOW CHARACTERISTICS

### Summary of Architecture Decisions:

| Aspect | Implementation | Benefit |
|--------|---------------|---------|
| **Architecture** | Layered (Client → API → Logic → Data) | Separation of concerns, maintainability |
| **Communication** | REST API + WebSocket | Synchronous + Real-time updates |
| **Authentication** | JWT + Role-based | Secure, scalable, stateless |
| **Data Persistence** | MongoDB + Redis | Document storage + Fast caching |
| **Training Engines** | Factory Pattern | Extensible, easy to add new methods |
| **File Processing** | Async I/O (aiofiles) | Non-blocking, better concurrency |
| **Caching Strategy** | Redis + Memory fallback | Distributed with local backup |
| **Error Handling** | Classified retry logic | Graceful degradation |
| **Monitoring** | Prometheus + Grafana | Industry-standard observability |
| **Deployment** | Docker Compose | Portable, reproducible |

---

**Dokumen ini dapat digunakan sebagai referensi untuk:**
- Pembuatan diagram arsitektur (Draw.io, Lucidchart, etc.)
- Dokumentasi teknis untuk developer
- Onboarding tim baru
- Audit dan review sistem
- Presentasi stakeholder
