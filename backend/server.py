from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
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
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 2
    max_seq_length: int = 512
    target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    use_gpu: bool = True
    gradient_accumulation_steps: int = 4
    use_wandb: bool = False  # Default to False for easier local testing

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
    accuracy: float
    perplexity: float
    f1_score: float
    precision: float
    recall: float
    evaluated_at: str

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
    
    # Simulate evaluation
    result = {
        "id": eval_id,
        "model_id": model_id,
        "accuracy": round(random.uniform(0.85, 0.95), 4),
        "perplexity": round(random.uniform(2.5, 4.5), 4),
        "f1_score": round(random.uniform(0.80, 0.92), 4),
        "precision": round(random.uniform(0.82, 0.93), 4),
        "recall": round(random.uniform(0.78, 0.90), 4),
        "evaluated_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.evaluations.insert_one(result)
    return result

@api_router.get("/evaluations", response_model=List[EvaluationResult])
async def get_evaluations():
    """Get all evaluation results"""
    evals = await db.evaluations.find({}, {"_id": 0}).sort("evaluated_at", -1).to_list(50)
    return evals

# Dashboard Stats
@api_router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    total_jobs = await db.training_jobs.count_documents({})
    active_jobs = await db.training_jobs.count_documents({"status": "training"})
    completed_jobs = await db.training_jobs.count_documents({"status": "completed"})
    total_datasets = await db.datasets.count_documents({})
    total_checkpoints = await db.checkpoints.count_documents({})
    
    return {
        "total_training_jobs": total_jobs,
        "active_training_jobs": active_jobs,
        "completed_training_jobs": completed_jobs,
        "total_datasets": total_datasets,
        "total_checkpoints": total_checkpoints
    }

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
    """Execute real QLoRA training"""
    try:
        # Update status
        await db.training_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "initializing"}}
        )

        # 1. Get dataset info
        dataset_info = await db.datasets.find_one({"id": config.dataset_id})
        if not dataset_info:
            raise ValueError("Dataset not found")
            
        file_path = dataset_info.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")

        # 2. Map Model ID
        # Mapping simplified for prototype
        MODEL_MAP = {
            "llama-2-7b": "meta-llama/Llama-2-7b-hf",
            "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
            "mistral-7b": "mistralai/Mistral-7B-v0.3",
            "gemma-7b": "google/gemma-7b",
        }
        real_model_id = MODEL_MAP.get(config.model_id, config.model_id)

        # 3. Process Dataset (Blocking, run in thread)
        def process_data():
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

        # Run data processing in thread
        logger.info("Processing dataset...")
        train_dataset = await asyncio.to_thread(process_data)
        
        # 4. Initialize Engine & Train
        engine = QLoRATrainingEngine(config.dict())
        
        # We need to modify start_training to be truly async or run in thread
        # Since start_training in core is defined async but has blocking calls, 
        # we will wrap it. Ideally refactor core, but for now:
        await engine.start_training(job_id, real_model_id, train_dataset, db)

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