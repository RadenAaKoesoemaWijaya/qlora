from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
import os
import uuid
import json
import asyncio
import random
import logging
import zipfile
import shutil

try:
    from core.training_engine import QLoRATrainingEngine
    from core.data_processor import DataProcessor
    from transformers import AutoTokenizer
    import torch
    HAS_ML_LIBS = True
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_ML_LIBS = False
    HAS_GPU = False
    logging.getLogger(__name__).warning("ML libraries not found or incomplete. Running in SIMULATION MODE.")

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical LLM Fine-tuning API")
api_router = APIRouter(prefix="/api")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= MODELS =============

class ModelInfo(BaseModel):
    id: str
    name: str
    size: str
    description: str
    type: str
    provider: str
    parameters: str

class DatasetUpload(BaseModel):
    id: str
    name: str
    file_type: str
    size: int
    rows: int
    created_at: str
    status: str
    validation_status: str

class TrainingConfig(BaseModel):
    model_id: str
    dataset_id: str
    # Training method selection
    training_method: str = "qlora"  # qlora, dora, ia3, vera, lora_plus, adalora, oft
    # Method-specific config
    method_config: Dict[str, Any] = Field(default_factory=dict)
    # LoRA/DoRA Configuration
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 2
    max_seq_length: int = 512
    use_gpu: bool = True
    gradient_accumulation_steps: int = 4
    use_wandb: bool = False  # Default to False for easier local testing
    
    @validator('model_id', 'dataset_id', 'training_method')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validasi field tidak boleh kosong."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()
    
    @validator('training_method')
    @classmethod
    def validate_training_method(cls, v: str) -> str:
        """Validasi training method yang didukung."""
        valid_methods = {'qlora', 'dora', 'ia3', 'vera', 'lora_plus', 'adalora', 'oft'}
        v_lower = v.lower().strip()
        if v_lower not in valid_methods:
            raise ValueError(f"Invalid training method: {v}. Must be one of: {', '.join(valid_methods)}")
        return v_lower
    
    @validator('lora_rank')
    @classmethod
    def validate_lora_rank(cls, v: int) -> int:
        """Validasi LoRA rank dalam range yang valid."""
        if not isinstance(v, int):
            raise ValueError("lora_rank must be an integer")
        if v < 1 or v > 1024:
            raise ValueError("lora_rank must be between 1 and 1024")
        return v
    
    @validator('lora_alpha')
    @classmethod
    def validate_lora_alpha(cls, v: int) -> int:
        """Validasi lora_alpha harus positif."""
        if not isinstance(v, int):
            raise ValueError("lora_alpha must be an integer")
        if v < 1:
            raise ValueError("lora_alpha must be at least 1")
        return v
    
    @validator('learning_rate')
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validasi learning rate dalam range yang valid."""
        if not isinstance(v, (int, float)):
            raise ValueError("learning_rate must be a number")
        if v < 1e-6 or v > 1e-2:
            raise ValueError("learning_rate must be between 1e-6 and 1e-2")
        return float(v)
    
    @validator('num_epochs')
    @classmethod
    def validate_num_epochs(cls, v: int) -> int:
        """Validasi number of epochs."""
        if not isinstance(v, int):
            raise ValueError("num_epochs must be an integer")
        if v < 1 or v > 100:
            raise ValueError("num_epochs must be between 1 and 100")
        return v
    
    @validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validasi batch size."""
        if not isinstance(v, int):
            raise ValueError("batch_size must be an integer")
        if v < 1 or v > 128:
            raise ValueError("batch_size must be between 1 and 128")
        return v
    
    @validator('max_seq_length')
    @classmethod
    def validate_max_seq_length(cls, v: int) -> int:
        """Validasi max sequence length."""
        if not isinstance(v, int):
            raise ValueError("max_seq_length must be an integer")
        if v < 64 or v > 8192:
            raise ValueError("max_seq_length must be between 64 and 8192")
        return v
    
    @validator('lora_dropout')
    @classmethod
    def validate_lora_dropout(cls, v: float) -> float:
        """Validasi dropout rate."""
        if not isinstance(v, (int, float)):
            raise ValueError("lora_dropout must be a number")
        if v < 0.0 or v > 1.0:
            raise ValueError("lora_dropout must be between 0.0 and 1.0")
        return float(v)

class TrainingJob(BaseModel):
    id: str
    model_name: str
    dataset_name: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    current_loss: Optional[float]
    learning_rate: float
    started_at: str
    estimated_completion: Optional[str]
    config: Dict[str, Any]

class TrainingMetrics(BaseModel):
    epoch: int
    step: int
    loss: float
    learning_rate: float
    timestamp: str

class Checkpoint(BaseModel):
    id: str
    training_job_id: str
    epoch: int
    step: int
    loss: float
    created_at: str
    size_mb: float
    model_name: str

class EvaluationResult(BaseModel):
    id: str
    model_id: str
    # Legacy/Simple Metrics
    accuracy: float = 0.0
    perplexity: float
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    # Advanced NLP Metrics
    bert_score: float = 0.0
    rouge_l: float = 0.0
    bleu_score: float = 0.0
    evaluated_at: str
    details: Optional[Dict[str, Any]] = None

# ============= ENDPOINTS =============

@api_router.get("/")
async def root():
    return {"message": "Medical LLM Fine-tuning API", "status": "running"}

# Models
@api_router.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    """Get list of available base models for fine-tuning"""
    models = [
        {
            "id": "llama-2-7b",
            "name": "Llama 2 7B",
            "size": "7B",
            "description": "Meta's Llama 2 model, excellent for medical text generation",
            "type": "causal-lm",
            "provider": "Meta",
            "parameters": "7 billion"
        },
        {
            "id": "llama-3-8b",
            "name": "Llama 3 8B",
            "size": "8B",
            "description": "Latest Llama 3 model with improved reasoning",
            "type": "causal-lm",
            "provider": "Meta",
            "parameters": "8 billion"
        },
        {
            "id": "mistral-7b",
            "name": "Mistral 7B v0.3",
            "size": "7B",
            "description": "Mistral's powerful 7B model with sliding window attention",
            "type": "causal-lm",
            "provider": "Mistral AI",
            "parameters": "7 billion"
        },
        {
            "id": "gemma-7b",
            "name": "Gemma 7B",
            "size": "7B",
            "description": "Google's Gemma model optimized for instruction following",
            "type": "causal-lm",
            "provider": "Google",
            "parameters": "7 billion"
        },
        {
            "id": "mixtral-8x7b",
            "name": "Mixtral 8x7B",
            "size": "8x7B",
            "description": "Mixture of experts model with exceptional performance",
            "type": "causal-lm",
            "provider": "Mistral AI",
            "parameters": "47 billion"
        }
    ]
    return models

# Datasets
DATASET_DIR = ROOT_DIR / "datasets"
DATASET_DIR.mkdir(exist_ok=True)

@api_router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    """Upload training dataset (JSON/CSV/JSONL)"""
    try:
        content = await file.read()
        file_size = len(content)
        
        # Determine file type
        filename = file.filename
        file_type = filename.split('.')[-1].upper()
        
        # Save file to disk
        dataset_id = str(uuid.uuid4())
        file_ext = filename.split('.')[-1]
        save_path = DATASET_DIR / f"{dataset_id}.{file_ext}"
        
        with open(save_path, "wb") as f:
            f.write(content)
        
        # Parse content to count rows
        rows = 0
        decoded_content = content.decode()
        if file_type == 'JSON':
            data = json.loads(decoded_content)
            rows = len(data) if isinstance(data, list) else 1
        elif file_type == 'JSONL':
            rows = len(decoded_content.strip().split('\n'))
        elif file_type == 'CSV':
            rows = len(decoded_content.strip().split('\n')) - 1  # Minus header
        
        dataset = {
            "id": dataset_id,
            "name": name or filename,
            "file_type": file_type,
            "size": file_size,
            "rows": rows,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "uploaded",
            "validation_status": "valid",
            "file_path": str(save_path)
        }
        
        await db.datasets.insert_one(dataset)
        
        return dataset
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@api_router.get("/datasets", response_model=List[DatasetUpload])
async def get_datasets():
    """Get all uploaded datasets"""
    datasets = await db.datasets.find({}, {"_id": 0, "content": 0}).to_list(100)
    return datasets

@api_router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    result = await db.datasets.delete_one({"id": dataset_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"message": "Dataset deleted successfully"}

# Training
@api_router.post("/training/start")
async def start_training(config: TrainingConfig):
    """Start a new training job"""
    # Get model and dataset info
    models = await get_available_models()
    model = next((m for m in models if m['id'] == config.model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    dataset = await db.datasets.find_one({"id": config.dataset_id}, {"_id": 0})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "model_name": model['name'],
        "dataset_name": dataset['name'],
        "status": "initializing",
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": config.num_epochs,
        "current_loss": None,
        "learning_rate": config.learning_rate,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "estimated_completion": None,
        "config": config.dict()
    }
    
    await db.training_jobs.insert_one(job)
    
    # Start training in background
    asyncio.create_task(run_training_job(job_id, config))
    
    return job

@api_router.get("/training/jobs", response_model=List[TrainingJob])
async def get_training_jobs():
    """Get all training jobs"""
    jobs = await db.training_jobs.find({}, {"_id": 0}).sort("started_at", -1).to_list(50)
    return jobs

@api_router.get("/training/jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(job_id: str):
    """Get specific training job"""
    job = await db.training_jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job

@api_router.post("/training/jobs/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a training job"""
    result = await db.training_jobs.update_one(
        {"id": job_id},
        {"$set": {"status": "stopped"}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Training job not found")
    return {"message": "Training stopped"}

@api_router.get("/training/jobs/{job_id}/metrics", response_model=List[TrainingMetrics])
async def get_training_metrics(job_id: str):
    """Get training metrics for a job"""
    metrics = await db.training_metrics.find(
        {"job_id": job_id},
        {"_id": 0}
    ).sort("step", 1).to_list(1000)
    return metrics

# Checkpoints
@api_router.get("/checkpoints", response_model=List[Checkpoint])
async def get_checkpoints(job_id: Optional[str] = None):
    """Get all checkpoints"""
    query = {"training_job_id": job_id} if job_id else {}
    checkpoints = await db.checkpoints.find(query, {"_id": 0}).sort("created_at", -1).to_list(100)
    return checkpoints

@api_router.delete("/checkpoints/{checkpoint_id}")
async def delete_checkpoint(checkpoint_id: str):
    """Delete a checkpoint"""
    result = await db.checkpoints.delete_one({"id": checkpoint_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return {"message": "Checkpoint deleted"}

# Evaluation
@api_router.post("/evaluate")
async def evaluate_model(model_id: str, dataset_id: str):
    """Evaluate a fine-tuned model"""
    eval_id = str(uuid.uuid4())
    
    # Check dependencies
    if not HAS_ML_LIBS:
        # Fallback to simulation if libs not present
        result = {
            "id": eval_id,
            "model_id": model_id,
            "accuracy": round(random.uniform(0.85, 0.95), 4),
            "perplexity": round(random.uniform(2.5, 4.5), 4),
            "f1_score": round(random.uniform(0.80, 0.92), 4),
            "precision": round(random.uniform(0.82, 0.93), 4),
            "recall": round(random.uniform(0.78, 0.90), 4),
            "bert_score": round(random.uniform(0.85, 0.90), 4),
            "rouge_l": round(random.uniform(0.40, 0.60), 4),
            "bleu_score": round(random.uniform(20.0, 40.0), 2),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "details": {"note": "Simulated Result (ML libs missing)"}
        }
    else:
        try:
            from core.evaluation_engine import EvaluationEngine
            
            # 1. Get dataset info
            dataset_info = await db.datasets.find_one({"id": dataset_id})
            if not dataset_info:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            file_path = dataset_info.get("file_path")
            
            # 2. Process dataset for evaluation (take subset)
            def load_eval_data():
                processor = DataProcessor(None) # Tokenizer not needed for raw read
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if dataset_info['file_type'] == 'JSON':
                    data = json.loads(content)
                elif dataset_info['file_type'] == 'JSONL':
                    data = [json.loads(line) for line in content.strip().split('\n')]
                else:
                    data = [] # CSV implementation skipped for brevity
                
                # Normalize keys
                normalized = []
                for item in data:
                    norm_item = {}
                    # Try to find instruction/output keys
                    keys = item.keys()
                    inst_key = next((k for k in keys if 'instruction' in k.lower() or 'prompt' in k.lower()), None)
                    out_key = next((k for k in keys if 'output' in k.lower() or 'response' in k.lower()), None)
                    
                    if inst_key and out_key:
                        norm_item['instruction'] = item[inst_key]
                        norm_item['output'] = item[out_key]
                        normalized.append(norm_item)
                return normalized

            eval_data = await asyncio.to_thread(load_eval_data)
            if not eval_data:
                raise HTTPException(status_code=400, detail="Could not parse dataset or find instruction/output columns")

            # 3. Run Evaluation (in thread/process ideally, but here inline async)
            # Determine if we are evaluating a base model or adapter
            # For this prototype, assuming model_id is a base model ID or path
            # In real app, we would look up the training job to find adapter path
            
            # Check if this is a finetuned model (check training jobs)
            job = await db.training_jobs.find_one({"model_path": {"$regex": f".*{model_id}.*"}})
            adapter_path = None
            base_model = model_id
            
            if job:
                # It's a trained model
                adapter_path = job.get("model_path")
                base_model = job.get("config", {}).get("model_id", "meta-llama/Llama-2-7b-hf")
            
            # Initialize Engine
            engine = EvaluationEngine(base_model_id=base_model, adapter_path=adapter_path)
            await engine.load_model()
            
            # Run Eval
            metrics = await asyncio.to_thread(engine.evaluate_dataset, eval_data, limit=20)
            
            result = {
                "id": eval_id,
                "model_id": model_id,
                "perplexity": round(metrics['perplexity'], 4),
                "bert_score": round(metrics['bert_score_f1'], 4),
                "rouge_l": round(metrics['rougeL'], 4),
                "bleu_score": round(metrics['bleu'], 2),
                # Legacy fields mapping
                "accuracy": 0.0, # Not applicable for generation
                "f1_score": round(metrics['bert_score_f1'], 4), # Use BERTScore as proxy
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "details": {
                    "samples": metrics['samples'],
                    "rouge1": metrics['rouge1'],
                    "rouge2": metrics['rouge2']
                }
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    await db.evaluations.insert_one(result)
    return result

@api_router.get("/evaluations", response_model=List[EvaluationResult])
async def get_evaluations():
    """Get all evaluation results"""
    evals = await db.evaluations.find({}, {"_id": 0}).sort("evaluated_at", -1).to_list(50)
    return evals

@api_router.get("/training/jobs/{job_id}/download")
async def download_finetuned_model(job_id: str):
    """Download hasil fine-tuning sebagai ZIP file"""
    job = await db.training_jobs.find_one({"id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Training not yet completed")
    
    model_path = job.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model files not found")
    
    # Buat ZIP file
    zip_path = f"/tmp/{job_id}_model.zip"
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, model_path)
                    zipf.write(file_path, arcname)
        
        # Return file response
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"finetuned_model_{job_id}.zip",
            background=lambda: os.remove(zip_path) if os.path.exists(zip_path) else None
        )
    except Exception as e:
        logger.error(f"Error creating zip file for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create download package: {str(e)}")

@api_router.get("/huggingface/models")
async def search_huggingface_models(
    query: str = "",
    limit: int = 20,
    sort: str = "downloads"
):
    """Cari model dari HuggingFace Hub"""
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        models = api.list_models(
            search=query,
            sort=sort,
            limit=limit,
            filter="text-generation",  # Filter untuk LLM
            library="transformers"
        )
        
        return [
            {
                "id": model.modelId,
                "name": model.modelId.split("/")[-1] if "/" in model.modelId else model.modelId,
                "downloads": model.downloads,
                "likes": model.likes,
                "tags": model.tags,
                "pipeline_tag": model.pipeline_tag
            }
            for model in models
        ]
    except Exception as e:
        logger.error(f"Error fetching HF models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard Stats - Optimized dengan aggregation
@api_router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics dengan optimized aggregation"""
    # Single aggregation query untuk semua job stats (lebih efisien)
    job_stats = await db.training_jobs.aggregate([
        {
            "$group": {
                "_id": None,
                "total": {"$sum": 1},
                "active": {"$sum": {"$cond": [{"$eq": ["$status", "training"]}, 1, 0]}},
                "completed": {"$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}},
                "failed": {"$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}},
                "queued": {"$sum": {"$cond": [{"$eq": ["$status", "initializing"]}, 1, 0]}}
            }
        }
    ]).to_list(1)
    
    # Parallel queries untuk collections lain
    datasets_count, checkpoints_count = await asyncio.gather(
        db.datasets.count_documents({}),
        db.checkpoints.count_documents({})
    )
    
    # Extract stats dari aggregation result
    stats = job_stats[0] if job_stats else {
        "total": 0, "active": 0, "completed": 0, "failed": 0, "queued": 0
    }
    
    return {
        "total_training_jobs": stats.get("total", 0),
        "active_training_jobs": stats.get("active", 0),
        "completed_training_jobs": stats.get("completed", 0),
        "failed_training_jobs": stats.get("failed", 0),
        "queued_training_jobs": stats.get("queued", 0),
        "total_datasets": datasets_count,
        "total_checkpoints": checkpoints_count
    }

# Training Methods API
@api_router.get("/training/methods")
async def get_training_methods():
    """Get available training methods dengan metadata"""
    try:
        from core.training_engine_factory import TrainingEngineFactory
        methods = TrainingEngineFactory.get_available_methods()
        default_method = TrainingEngineFactory.get_default_method()
        
        return {
            "methods": methods,
            "default_method": default_method,
            "recommended_method": "dora" if "dora" in methods else "qlora"
        }
    except Exception as e:
        logger.error(f"Error getting training methods: {e}")
        # Fallback response jika import gagal
        return {
            "methods": {
                "qlora": {
                    "name": "QLoRA",
                    "description": "Quantized Low-Rank Adaptation",
                    "efficiency": "high",
                    "performance": "excellent",
                    "difficulty": "easy",
                    "available": True
                }
            },
            "default_method": "qlora",
            "recommended_method": "qlora"
        }

@api_router.get("/training/methods/{method_id}/config-schema")
async def get_method_config_schema(method_id: str):
    """Get configuration schema untuk specific training method"""
    schemas = {
        "dora": {
            "fields": [
                {
                    "name": "use_dora",
                    "type": "boolean",
                    "default": True,
                    "label": "Enable DoRA",
                    "description": "Aktifkan weight decomposition untuk performa lebih baik"
                },
                {
                    "name": "dora_simple",
                    "type": "boolean",
                    "default": False,
                    "label": "Simplified Mode",
                    "description": "Mode sederhana untuk training lebih cepat"
                }
            ]
        },
        "lora_plus": {
            "fields": [
                {
                    "name": "lora_plus_ratio",
                    "type": "integer",
                    "default": 16,
                    "min": 1,
                    "max": 64,
                    "label": "Learning Rate Ratio (A:B)",
                    "description": "Rasio learning rate antara matriks A dan B"
                }
            ]
        },
        "ia3": {
            "fields": [
                {
                    "name": "ia3_target_modules",
                    "type": "multiselect",
                    "default": ["k_proj", "v_proj", "down_proj"],
                    "options": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    "label": "Target Modules",
                    "description": "Modules untuk element-wise scaling"
                },
                {
                    "name": "ia3_feedforward_modules",
                    "type": "multiselect",
                    "default": ["down_proj"],
                    "options": ["gate_proj", "up_proj", "down_proj"],
                    "label": "Feedforward Modules",
                    "description": "Subset dari target modules untuk feedforward adaptation"
                }
            ]
        },
        "vera": {
            "fields": [
                {
                    "name": "vera_rank",
                    "type": "integer",
                    "default": 256,
                    "min": 64,
                    "max": 1024,
                    "step": 64,
                    "label": "Projection Rank (Frozen)",
                    "description": "Rank untuk frozen random projections"
                },
                {
                    "name": "vera_seed",
                    "type": "integer",
                    "default": 42,
                    "min": 0,
                    "max": 999999,
                    "label": "Random Seed",
                    "description": "Seed untuk reproducible initialization"
                }
            ]
        },
        "adalora": {
            "fields": [
                {
                    "name": "adalora_init_r",
                    "type": "integer",
                    "default": 12,
                    "min": 1,
                    "max": 64,
                    "label": "Initial Rank",
                    "description": "Initial rank sebelum pruning"
                },
                {
                    "name": "adalora_target_r",
                    "type": "integer",
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "label": "Target Budget Rank",
                    "description": "Target rank setelah pruning"
                },
                {
                    "name": "adalora_tinit",
                    "type": "integer",
                    "default": 0,
                    "min": 0,
                    "label": "Warmup Steps (tinit)",
                    "description": "Steps sebelum pruning dimulai"
                },
                {
                    "name": "adalora_tfinal",
                    "type": "integer",
                    "default": 1000,
                    "min": 100,
                    "label": "Pruning End (tfinal)",
                    "description": "Steps saat pruning selesai"
                },
                {
                    "name": "adalora_deltaT",
                    "type": "integer",
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "label": "Pruning Interval (deltaT)",
                    "description": "Interval antar pruning steps"
                },
                {
                    "name": "adalora_beta1",
                    "type": "float",
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Beta1 (Importance Decay)",
                    "description": "Decay factor untuk importance scores"
                }
            ]
        },
        "oft": {
            "fields": [
                {
                    "name": "oft_r",
                    "type": "integer",
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "label": "OFT Rank",
                    "description": "Rank untuk orthogonal transformation"
                },
                {
                    "name": "oft_dropout",
                    "type": "float",
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "label": "Module Dropout",
                    "description": "Dropout probability untuk OFT modules"
                },
                {
                    "name": "oft_init_weights",
                    "type": "boolean",
                    "default": True,
                    "label": "Initialize Weights",
                    "description": "Initialize dengan orthogonal matrices"
                }
            ]
        },
        "qlora": {
            "fields": []  # No method-specific config for standard QLoRA
        }
    }
    
    if method_id not in schemas:
        raise HTTPException(status_code=404, detail=f"Method {method_id} not found")
    
    return schemas[method_id]

@api_router.get("/training/methods/{method_id}/recommendations")
async def get_method_recommendations(method_id: str):
    """Get recommendations untuk penggunaan method tertentu"""
    recommendations = {
        "dora": {
            "recommended_for": [
                "Production fine-tuning",
                "Maximum performance requirements",
                "Stability-critical applications",
                "When using LoRA rank 8-32"
            ],
            "not_recommended_for": [
                "Very low resource constraints (< 8GB VRAM)"
            ],
            "default_rank": 16,
            "default_alpha": 32
        },
        "lora_plus": {
            "recommended_for": [
                "Fast iteration dan prototyping",
                "Large-scale experiments",
                "Quick convergence needs",
                "When training time is critical"
            ],
            "recommended_ratio": 16,
            "speedup": "2x convergence"
        },
        "ia3": {
            "recommended_for": [
                "Fast inference scenarios",
                "Key-value adaptation tasks",
                "When parameter count is critical",
                "Classification tasks"
            ],
            "recommended_target_modules": ["k_proj", "v_proj", "down_proj"]
        },
        "vera": {
            "recommended_for": [
                "Edge deployment",
                "Extreme resource constraints",
                "Multiple adapter storage",
                "IoT and mobile devices"
            ],
            "recommended_rank": 256,
            "parameter_reduction": "99.999%"
        },
        "adalora": {
            "recommended_for": [
                "Complex fine-tuning tasks",
                "Budget optimization needs",
                "When layer importance varies",
                "Multi-task scenarios"
            ],
            "recommended_budget_ratio": 0.3
        },
        "oft": {
            "recommended_for": [
                "Multimodal tasks",
                "Vision-language models",
                "Domain adaptation",
                "When geometric stability matters"
            ]
        },
        "qlora": {
            "recommended_for": [
                "General purpose fine-tuning",
                "First-time users",
                "Proven stability requirements",
                "When other methods fail"
            ]
        }
    }
    
    if method_id not in recommendations:
        raise HTTPException(status_code=404, detail=f"Method {method_id} not found")
    
    return recommendations[method_id]

# ============= TRAINING EXECUTION =============

async def run_training_job(job_id: str, config: TrainingConfig):
    """Main entry point for training job execution"""
    # Check resources and libraries
    can_run_real = HAS_ML_LIBS and HAS_GPU and config.use_gpu
    
    if can_run_real:
        logger.info(f"Starting REAL training for job {job_id}")
        await run_real_training(job_id, config)
    else:
        reason = "No GPU available" if not HAS_GPU else "ML libraries missing"
        if not config.use_gpu: reason = "GPU disabled in config"
        logger.info(f"Starting SIMULATED training for job {job_id} (Reason: {reason})")
        await run_simulated_training(job_id, config)

async def run_real_training(job_id: str, config: TrainingConfig):
    """Execute real training dengan metode yang dipilih"""
    try:
        # Update status
        await db.training_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "initializing"}}
        )

        # 1. Get dataset info dengan security validation
        from core.security import validate_dataset_path
        
        dataset_info = await db.datasets.find_one({"id": config.dataset_id})
        if not dataset_info:
            raise ValueError("Dataset not found")
            
        file_path = dataset_info.get("file_path")
        
        # Security: Validasi path untuk mencegah path traversal
        if not validate_dataset_path(file_path):
            raise HTTPException(status_code=400, detail="Invalid dataset path - possible path traversal attack")
        
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")

        # 2. Map Model ID
        MODEL_MAP = {
            "llama-2-7b": "meta-llama/Llama-2-7b-hf",
            "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
            "mistral-7b": "mistralai/Mistral-7B-v0.3",
            "gemma-7b": "google/gemma-7b",
            "phi-4": "microsoft/phi-4",
            "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
            "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "gemma-2-9b": "google/gemma-2-9b-it",
        }
        # Jika model_id tidak ada di mapping, anggap itu langsung HuggingFace model ID
        real_model_id = MODEL_MAP.get(config.model_id, config.model_id)
        
        logger.info(f"Using model ID: {real_model_id} (mapped from {config.model_id})")

        # 3. Process Dataset (Blocking, run in thread)
        def process_data():
            from core.data_processor import DataProcessor
            tokenizer = AutoTokenizer.from_pretrained(real_model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            processor = DataProcessor(tokenizer)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if dataset_info['file_type'] == 'JSON':
                processed = processor.process_json_dataset(content)
            elif dataset_info['file_type'] == 'JSONL':
                processed = processor.process_jsonl_dataset(content)
            elif dataset_info['file_type'] == 'CSV':
                processed = processor.process_csv_dataset(content)
            else:
                raise ValueError(f"Unsupported file type: {dataset_info['file_type']}")
                
            return processed['dataset']

        # Run data processing in thread dengan timeout
        logger.info("Processing dataset...")
        try:
            train_dataset = await asyncio.wait_for(
                asyncio.to_thread(process_data),
                timeout=300  # 5 minutes timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Dataset processing timeout for job {job_id}")
            await db.training_jobs.update_one(
                {"id": job_id},
                {"$set": {"status": "failed", "error": "Dataset processing timeout (5 minutes exceeded)"}}
            )
            return
        
        # 4. Get training method
        training_method = config.training_method.lower()
        logger.info(f"Using training method: {training_method}")
        
        # Update job dengan method info
        await db.training_jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": "training",
                "training_method": training_method,
                "method_config": config.method_config
            }}
        )
        
        # 5. Initialize Engine menggunakan Factory
        from core.training_engine_factory import TrainingEngineFactory
        
        # Merge base config dengan method-specific config
        merged_config = {
            **config.dict(),
            **config.method_config  # Method-specific config override
        }
        
        engine = None
        try:
            engine = TrainingEngineFactory.create_engine(training_method, merged_config)
            
            # 6. Start Training
            result = await engine.start_training(job_id, real_model_id, train_dataset, db)
            
            # Update final status
            if result["status"] == "completed":
                await db.training_jobs.update_one(
                    {"id": job_id},
                    {"$set": {
                        "status": "completed",
                        "progress": 100.0,
                        "model_path": result.get("model_path"),
                        "completed_at": datetime.now(timezone.utc).isoformat()
                    }}
                )
            else:
                await db.training_jobs.update_one(
                    {"id": job_id},
                    {"$set": {
                        "status": "failed",
                        "error": result.get("error", "Unknown error")
                    }}
                )
        finally:
            # Cleanup engine resources
            if engine is not None:
                try:
                    await engine.cleanup()
                    logger.info(f"Engine cleanup completed for job {job_id}")
                except Exception as cleanup_error:
                    logger.warning(f"Engine cleanup failed for job {job_id}: {cleanup_error}")

    except Exception as e:
        logger.error(f"Real training failed: {e}")
        await db.training_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "failed", "error": str(e)}}
        )

async def run_simulated_training(job_id: str, config: TrainingConfig):
    """Simulate training process with realistic metrics"""
    try:
        # Update to training status
        await db.training_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "training"}}
        )
        
        total_steps = config.num_epochs * 100  # 100 steps per epoch
        step = 0
        initial_loss = 2.5
        
        for epoch in range(config.num_epochs):
            for mini_step in range(100):
                # Check if stopped
                job = await db.training_jobs.find_one({"id": job_id})
                if job['status'] == 'stopped':
                    return
                
                step += 1
                progress = (step / total_steps) * 100
                
                # Simulate decreasing loss with some noise
                loss = initial_loss * (1 - (step / total_steps) * 0.7) + random.uniform(-0.05, 0.05)
                loss = max(0.3, loss)  # Minimum loss
                
                # Update job
                await db.training_jobs.update_one(
                    {"id": job_id},
                    {"$set": {
                        "progress": round(progress, 2),
                        "current_epoch": epoch + 1,
                        "current_loss": round(loss, 4)
                    }}
                )
                
                # Store metrics
                metric = {
                    "job_id": job_id,
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": round(loss, 4),
                    "learning_rate": config.learning_rate,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                await db.training_metrics.insert_one(metric)
                
                # Save checkpoint at end of epoch
                if mini_step == 99:
                    checkpoint = {
                        "id": str(uuid.uuid4()),
                        "training_job_id": job_id,
                        "epoch": epoch + 1,
                        "step": step,
                        "loss": round(loss, 4),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "size_mb": round(random.uniform(450, 550), 2),
                        "model_name": f"checkpoint-epoch-{epoch + 1}"
                    }
                    await db.checkpoints.insert_one(checkpoint)
                
                # Simulate time
                await asyncio.sleep(0.5)  # 0.5 seconds per step
        
        # Mark as completed
        await db.training_jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": "completed",
                "progress": 100.0
            }}
        )
        
    except Exception as e:
        logger.error(f"Training simulation failed: {e}")
        await db.training_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "failed"}}
        )

app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()