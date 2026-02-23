import json
import logging
import time
import psutil
import torch
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

class TrainingPhase(Enum):
    """Training phases."""
    INITIALIZING = "initializing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"

class MetricType(Enum):
    """Types of metrics."""
    LOSS = "loss"
    LEARNING_RATE = "learning_rate"
    GRADIENT_NORM = "gradient_norm"
    GPU_MEMORY = "gpu_memory"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    TRAINING_SPEED = "training_speed"
    ETA = "eta"

@dataclass
class TrainingMetric:
    """Training metric data point."""
    job_id: str
    metric_type: str
    value: float
    step: int
    epoch: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TrainingSnapshot:
    """Training snapshot for real-time monitoring."""
    job_id: str
    phase: TrainingPhase
    current_step: int
    total_steps: int
    current_epoch: float
    total_epochs: int
    loss: Optional[float]
    learning_rate: Optional[float]
    gpu_memory_used: Optional[float]
    gpu_memory_total: Optional[float]
    cpu_usage: Optional[float]
    memory_usage: Optional[float]
    training_speed: Optional[float]  # steps per second
    eta: Optional[str]  # estimated time to completion
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class MetricsBuffer:
    """Thread-safe metrics buffer for batch processing."""
    
    def __init__(self, max_size: int = 1000, flush_interval: float = 5.0):
        self.buffer = deque(maxlen=max_size)
        self.flush_interval = flush_interval
        self.last_flush = datetime.now()
        self._lock = asyncio.Lock()
    
    async def add_metric(self, metric: TrainingMetric):
        """Add metric to buffer."""
        async with self._lock:
            self.buffer.append(metric)
    
    async def get_metrics(self) -> List[TrainingMetric]:
        """Get all metrics and clear buffer."""
        async with self._lock:
            metrics = list(self.buffer)
            self.buffer.clear()
            return metrics
    
    def should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        return (datetime.now() - self.last_flush).total_seconds() >= self.flush_interval
    
    def update_flush_time(self):
        """Update last flush time."""
        self.last_flush = datetime.now()

class RealTimeMetricsCollector:
    """Collects real-time training metrics."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.metrics_buffer = MetricsBuffer()
        self.training_snapshots = deque(maxlen=100)  # Keep last 100 snapshots
        self.start_time = datetime.now()
        self.step_times = deque(maxlen=100)  # Keep last 100 step times
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics (CPU, memory, GPU)."""
        try:
            # CPU and memory metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            metrics = {
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3)
            }
            
            # GPU metrics if available
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                metrics.update({
                    "gpu_memory_used_gb": gpu_memory_used,
                    "gpu_memory_total_gb": gpu_memory_total,
                    "gpu_memory_percent": (gpu_memory_used / gpu_memory_total) * 100,
                    "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return {}
    
    def calculate_training_speed(self, current_step: int) -> float:
        """Calculate training speed in steps per second."""
        current_time = datetime.now()
        
        if not self.step_times:
            return 0.0
        
        # Calculate average step time
        total_time = (current_time - self.start_time).total_seconds()
        steps_per_second = current_step / total_time if total_time > 0 else 0.0
        
        return steps_per_second
    
    def calculate_eta(self, current_step: int, total_steps: int, 
                     steps_per_second: float) -> str:
        """Calculate estimated time to completion."""
        if steps_per_second <= 0:
            return "Unknown"
        
        remaining_steps = total_steps - current_step
        if remaining_steps <= 0:
            return "Completed"
        
        remaining_seconds = remaining_steps / steps_per_second
        remaining_time = timedelta(seconds=remaining_seconds)
        
        # Format as HH:MM:SS
        hours, remainder = divmod(remaining_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    async def create_training_snapshot(self, state: TrainerState, 
                                       phase: TrainingPhase) -> TrainingSnapshot:
        """Create training snapshot."""
        # Collect system metrics
        system_metrics = await self.collect_system_metrics()
        
        # Calculate training speed
        training_speed = self.calculate_training_speed(state.global_step)
        
        # Calculate ETA
        eta = self.calculate_eta(
            state.global_step, 
            state.max_steps, 
            training_speed
        )
        
        # Get current loss
        current_loss = None
        if state.log_history and len(state.log_history) > 0:
            latest_logs = state.log_history[-1]
            if "loss" in latest_logs:
                current_loss = latest_logs["loss"]
        
        snapshot = TrainingSnapshot(
            job_id=self.job_id,
            phase=phase,
            current_step=state.global_step,
            total_steps=state.max_steps,
            current_epoch=state.epoch,
            total_epochs=state.num_train_epochs,
            loss=current_loss,
            learning_rate=state.log_history[-1].get("learning_rate", 0) if state.log_history else None,
            gpu_memory_used=system_metrics.get("gpu_memory_used_gb"),
            gpu_memory_total=system_metrics.get("gpu_memory_total_gb"),
            cpu_usage=system_metrics.get("cpu_usage"),
            memory_usage=system_metrics.get("memory_usage"),
            training_speed=training_speed,
            eta=eta,
            timestamp=datetime.now(),
            metadata=system_metrics
        )
        
        # Store snapshot
        self.training_snapshots.append(snapshot)
        
        return snapshot
    
    async def store_metric(self, metric: TrainingMetric):
        """Store metric in buffer."""
        await self.metrics_buffer.add_metric(metric)
    
    def get_latest_snapshot(self) -> Optional[TrainingSnapshot]:
        """Get latest training snapshot."""
        return self.training_snapshots[-1] if self.training_snapshots else None
    
    def get_snapshots(self, limit: int = 100) -> List[TrainingSnapshot]:
        """Get training snapshots."""
        return list(self.training_snapshots)[-limit:]
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

class EnhancedQLoRATrainingCallback(TrainerCallback):
    """
    Enhanced QLoRA training callback dengan comprehensive monitoring dan real-time updates.
    """
    
    def __init__(self, job_id: str, db=None, websocket_manager=None, 
                 update_interval: int = 50, enable_real_time_monitoring: bool = True):
        self.job_id = job_id
        self.db = db
        self.websocket_manager = websocket_manager
        self.update_interval = update_interval
        self.enable_real_time_monitoring = enable_real_time_monitoring
        
        # Initialize metrics collector
        self.metrics_collector = RealTimeMetricsCollector(job_id) if enable_real_time_monitoring else None
        
        # Training state
        self.start_time = datetime.now()
        self.last_update_step = 0
        self.training_phase = TrainingPhase.INITIALIZING
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = 5
        
        # Performance tracking
        self.loss_history = deque(maxlen=100)
        self.learning_rate_history = deque(maxlen=100)
        self.gradient_norm_history = deque(maxlen=100)
        
        # Async executor for database operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"Enhanced training callback initialized for job: {job_id}")
    
    async def _update_database_async(self, update_data: Dict[str, Any]):
        """Update database asynchronously."""
        if not self.db:
            return
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._update_database_sync,
                update_data
            )
        except Exception as e:
            logger.error(f"Error updating database asynchronously: {str(e)}")
    
    def _update_database_sync(self, update_data: Dict[str, Any]):
        """Update database synchronously."""
        try:
            self.db.training_jobs.update_one(
                {"id": self.job_id},
                {"$set": update_data}
            )
        except Exception as e:
            logger.error(f"Error updating database synchronously: {str(e)}")
    
    async def _broadcast_update_async(self, data: Dict[str, Any]):
        """Broadcast update via WebSocket asynchronously."""
        if not self.websocket_manager:
            return
        
        try:
            await self.websocket_manager.broadcast_training_update(
                self.job_id, data
            )
        except Exception as e:
            logger.error(f"Error broadcasting WebSocket update: {str(e)}")
    
    def _store_training_metrics(self, metrics: List[TrainingMetric]):
        """Store training metrics in database."""
        if not self.db or not metrics:
            return
        
        try:
            # Convert metrics to dictionaries
            metric_dicts = [asdict(metric) for metric in metrics]
            
            # Add to database in batch
            if metric_dicts:
                self.db.training_metrics.insert_many(metric_dicts)
                
        except Exception as e:
            logger.error(f"Error storing training metrics: {str(e)}")
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                      control: TrainerControl, **kwargs):
        """Called when training begins."""
        self.training_phase = TrainingPhase.TRAINING
        self.start_time = datetime.now()
        
        logger.info(f"Enhanced training started for job: {self.job_id}")
        logger.info(f"Total steps: {state.max_steps}, Epochs: {state.num_train_epochs}")
        
        # Update database
        if self.db:
            try:
                training_config = {
                    "learning_rate": args.learning_rate,
                    "batch_size": args.per_device_train_batch_size,
                    "max_steps": state.max_steps,
                    "num_epochs": state.num_train_epochs,
                    "warmup_steps": args.warmup_steps,
                    "logging_steps": args.logging_steps,
                    "save_steps": args.save_steps,
                    "eval_steps": args.eval_steps
                }
                
                update_data = {
                    "status": "training",
                    "started_at": datetime.now().isoformat(),
                    "total_steps": state.max_steps,
                    "total_epochs": state.num_train_epochs,
                    "current_step": 0,
                    "current_epoch": 0,
                    "progress": 0.0,
                    "training_config": training_config,
                    "phase": self.training_phase.value
                }
                
                # Async database update
                asyncio.create_task(self._update_database_async(update_data))
                
            except Exception as e:
                logger.error(f"Error updating database on_train_begin: {str(e)}")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, 
                   control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        
        # Collect metrics every update_interval steps
        if state.global_step % self.update_interval == 0 or state.global_step == state.max_steps:
            
            # Create training snapshot
            if self.metrics_collector:
                try:
                    # Run async operations
                    loop = asyncio.get_event_loop()
                    
                    # Create snapshot
                    snapshot = loop.run_until_complete(
                        self.metrics_collector.create_training_snapshot(state, self.training_phase)
                    )
                    
                    # Prepare update data
                    update_data = {
                        "current_step": state.global_step,
                        "current_epoch": state.epoch,
                        "progress": round((state.global_step / state.max_steps * 100), 2),
                        "loss": snapshot.loss,
                        "learning_rate": snapshot.learning_rate,
                        "gpu_memory_used_gb": snapshot.gpu_memory_used,
                        "gpu_memory_total_gb": snapshot.gpu_memory_total,
                        "cpu_usage": snapshot.cpu_usage,
                        "memory_usage": snapshot.memory_usage,
                        "training_speed": snapshot.training_speed,
                        "eta": snapshot.eta,
                        "phase": self.training_phase.value,
                        "last_updated": datetime.now().isoformat()
                    }
                    
                    # Update database asynchronously
                    asyncio.create_task(self._update_database_async(update_data))
                    
                    # Broadcast update via WebSocket
                    broadcast_data = {
                        "type": "training_update",
                        "job_id": self.job_id,
                        "snapshot": asdict(snapshot),
                        "metrics": {
                            "loss": snapshot.loss,
                            "learning_rate": snapshot.learning_rate,
                            "progress": update_data["progress"],
                            "gpu_memory_percent": ((snapshot.gpu_memory_used / snapshot.gpu_memory_total * 100) 
                                                 if snapshot.gpu_memory_used and snapshot.gpu_memory_total else None)
                        }
                    }
                    
                    asyncio.create_task(self._broadcast_update_async(broadcast_data))
                    
                    # Store metrics in buffer
                    if snapshot.loss is not None:
                        metric = TrainingMetric(
                            job_id=self.job_id,
                            metric_type=MetricType.LOSS.value,
                            value=snapshot.loss,
                            step=state.global_step,
                            epoch=state.epoch,
                            timestamp=datetime.now()
                        )
                        asyncio.create_task(self.metrics_collector.store_metric(metric))
                    
                    # Log progress
                    logger.info(
                        f"Job {self.job_id}: Step {state.global_step}/{state.max_steps} "
                        f"({update_data['progress']:.1f}%) - Loss: {snapshot.loss:.4f} "
                        f"- Speed: {snapshot.training_speed:.2f} steps/s - ETA: {snapshot.eta}"
                    )
                    
                    self.last_update_step = state.global_step
                    
                except Exception as e:
                    logger.error(f"Error in enhanced step monitoring: {str(e)}")
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        """Called at the end of each epoch."""
        self.training_phase = TrainingPhase.EVALUATING
        
        logger.info(f"Job {self.job_id}: Epoch {state.epoch} completed")
        
        # Update database
        if self.db:
            try:
                update_data = {
                    "current_epoch": state.epoch,
                    "current_step": state.global_step,
                    "phase": self.training_phase.value,
                    "epoch_completed_at": datetime.now().isoformat()
                }
                
                asyncio.create_task(self._update_database_async(update_data))
                
                # Broadcast epoch completion
                broadcast_data = {
                    "type": "epoch_completed",
                    "job_id": self.job_id,
                    "epoch": state.epoch,
                    "step": state.global_step
                }
                
                asyncio.create_task(self._broadcast_update_async(broadcast_data))
                
            except Exception as e:
                logger.error(f"Error updating database on_epoch_end: {str(e)}")
    
    def on_save(self, args: TrainingArguments, state: TrainerState, 
               control: TrainerControl, **kwargs):
        """Called when model checkpoint is saved."""
        self.training_phase = TrainingPhase.SAVING
        
        logger.info(f"Job {self.job_id}: Checkpoint saved at step {state.global_step}")
        
        # Create checkpoint record in database
        if self.db:
            try:
                current_loss = None
                if state.log_history and len(state.log_history) > 0:
                    # Find the last loss value
                    for log_entry in reversed(state.log_history):
                        if "loss" in log_entry:
                            current_loss = log_entry["loss"]
                            break
                
                checkpoint = {
                    "id": f"checkpoint_{self.job_id}_{state.global_step}",
                    "training_job_id": self.job_id,
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "loss": current_loss,
                    "created_at": datetime.now().isoformat(),
                    "model_name": f"checkpoint-{state.global_step}",
                    "path": f"./checkpoints/{self.job_id}/checkpoint-{state.global_step}",
                    "size_mb": self._get_checkpoint_size(state.global_step)
                }
                
                # Async database operation
                loop = asyncio.get_event_loop()
                loop.run_in_executor(
                    self.executor,
                    self.db.checkpoints.insert_one,
                    checkpoint
                )
                
                # Update job with latest checkpoint
                update_data = {
                    "latest_checkpoint_step": state.global_step,
                    "latest_checkpoint_at": datetime.now().isoformat(),
                    "phase": self.training_phase.value
                }
                
                asyncio.create_task(self._update_database_async(update_data))
                
                # Broadcast checkpoint save
                broadcast_data = {
                    "type": "checkpoint_saved",
                    "job_id": self.job_id,
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "loss": current_loss
                }
                
                asyncio.create_task(self._broadcast_update_async(broadcast_data))
                
            except Exception as e:
                logger.error(f"Error creating checkpoint record: {str(e)}")
    
    def _get_checkpoint_size(self, step: int) -> Optional[float]:
        """Get checkpoint size in MB."""
        try:
            import os
            checkpoint_path = f"./checkpoints/{self.job_id}/checkpoint-{step}"
            if os.path.exists(checkpoint_path):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(checkpoint_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)
                return total_size / (1024 * 1024)  # Convert to MB
        except:
            pass
        return None
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        """Called when training ends."""
        self.training_phase = TrainingPhase.COMPLETED
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() / 3600  # hours
        
        logger.info(f"Enhanced training completed for job: {self.job_id}")
        logger.info(f"Total duration: {duration:.2f} hours")
        logger.info(f"Total steps: {state.global_step}")
        logger.info(f"Average speed: {state.global_step / (duration * 3600):.2f} steps/second")
        
        # Get final metrics
        final_loss = None
        final_learning_rate = None
        if state.log_history and len(state.log_history) > 0:
            # Find the last loss and learning rate values
            for log_entry in reversed(state.log_history):
                if final_loss is None and "loss" in log_entry:
                    final_loss = log_entry["loss"]
                if final_learning_rate is None and "learning_rate" in log_entry:
                    final_learning_rate = log_entry["learning_rate"]
                if final_loss is not None and final_learning_rate is not None:
                    break
        
        # Update database
        if self.db:
            try:
                # Get final snapshot
                if self.metrics_collector:
                    loop = asyncio.get_event_loop()
                    final_snapshot = loop.run_until_complete(
                        self.metrics_collector.create_training_snapshot(state, self.training_phase)
                    )
                    
                    # Store final metrics
                    metrics_to_store = []
                    if final_loss is not None:
                        metrics_to_store.append(TrainingMetric(
                            job_id=self.job_id,
                            metric_type=MetricType.LOSS.value,
                            value=final_loss,
                            step=state.global_step,
                            epoch=state.epoch,
                            timestamp=datetime.now()
                        ))
                    
                    if metrics_to_store:
                        self._store_training_metrics(metrics_to_store)
                
                update_data = {
                    "status": "completed",
                    "progress": 100.0,
                    "current_step": state.global_step,
                    "final_loss": final_loss,
                    "final_learning_rate": final_learning_rate,
                    "completed_at": end_time.isoformat(),
                    "duration_hours": round(duration, 2),
                    "total_epochs_completed": state.epoch,
                    "phase": self.training_phase.value,
                    "training_summary": {
                        "total_steps": state.global_step,
                        "total_epochs": state.epoch,
                        "duration_hours": round(duration, 2),
                        "average_speed_steps_per_second": round(state.global_step / (duration * 3600), 2),
                        "final_loss": final_loss,
                        "best_loss": self.best_loss if self.best_loss != float('inf') else None
                    }
                }
                
                # Final database update
                asyncio.create_task(self._update_database_async(update_data))
                
                # Final broadcast
                broadcast_data = {
                    "type": "training_completed",
                    "job_id": self.job_id,
                    "summary": update_data["training_summary"]
                }
                
                asyncio.create_task(self._broadcast_update_async(broadcast_data))
                
            except Exception as e:
                logger.error(f"Error updating database on_train_end: {str(e)}")
        
        # Cleanup
        if self.metrics_collector:
            self.metrics_collector.cleanup()
        
        self.executor.shutdown(wait=True)
    
    def on_exception(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, exception: Exception, **kwargs):
        """Handle training exceptions."""
        self.training_phase = TrainingPhase.FAILED
        
        logger.error(f"Training exception for job {self.job_id}: {str(exception)}")
        logger.error(f"Exception type: {type(exception).__name__}")
        
        # Update database with error status
        if self.db:
            try:
                update_data = {
                    "status": "failed",
                    "error": str(exception),
                    "error_type": type(exception).__name__,
                    "failed_at": datetime.now().isoformat(),
                    "current_step": state.global_step if state else 0,
                    "current_epoch": state.epoch if state else 0,
                    "phase": self.training_phase.value,
                    "error_context": {
                        "step": state.global_step if state else 0,
                        "epoch": state.epoch if state else 0,
                        "progress": (state.global_step / state.max_steps * 100) if state and state.max_steps > 0 else 0
                    }
                }
                
                asyncio.create_task(self._update_database_async(update_data))
                
                # Broadcast error
                broadcast_data = {
                    "type": "training_failed",
                    "job_id": self.job_id,
                    "error": str(exception),
                    "error_type": type(exception).__name__,
                    "step": state.global_step if state else 0
                }
                
                asyncio.create_task(self._broadcast_update_async(broadcast_data))
                
            except Exception as e:
                logger.error(f"Error updating database with failure status: {str(e)}")
        
        # Return control to stop training
        control.should_training_stop = True
        return control
    
    def on_log(self, args: TrainingArguments, state: TrainerState, 
              control: TrainerControl, logs: Dict[str, Any], **kwargs):
        """Called when logs are produced."""
        # Track loss history for early stopping detection
        if "loss" in logs:
            current_loss = logs["loss"]
            self.loss_history.append(current_loss)
            
            # Update best loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Check for early stopping (optional - can be enabled)
            # if self.patience_counter >= self.early_stopping_patience:
            #     logger.warning(f"Early stopping triggered after {self.patience_counter} steps without improvement")
            #     control.should_training_stop = True
        
        # Track learning rate history
        if "learning_rate" in logs:
            self.learning_rate_history.append(logs["learning_rate"])
        
        # Track gradient norm if available
        if "total_flos" in logs:
            # This is a placeholder - actual gradient norm tracking would require
            # custom gradient clipping logic
            pass
        
        # Store additional metrics
        if self.metrics_collector:
            try:
                # Create metrics for different log values
                current_time = datetime.now()
                
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        metric = TrainingMetric(
                            job_id=self.job_id,
                            metric_type=key,
                            value=float(value),
                            step=state.global_step,
                            epoch=state.epoch,
                            timestamp=current_time
                        )
                        asyncio.create_task(self.metrics_collector.store_metric(metric))
                        
            except Exception as e:
                logger.error(f"Error storing additional metrics: {str(e)}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        latest_snapshot = self.metrics_collector.get_latest_snapshot() if self.metrics_collector else None
        
        return {
            "job_id": self.job_id,
            "phase": self.training_phase.value,
            "best_loss": self.best_loss if self.best_loss != float('inf') else None,
            "loss_history": list(self.loss_history),
            "learning_rate_history": list(self.learning_rate_history),
            "latest_snapshot": asdict(latest_snapshot) if latest_snapshot else None,
            "start_time": self.start_time.isoformat(),
            "training_duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600
        }

class AdvancedErrorHandlingCallback(TrainerCallback):
    """Advanced error handling callback dengan recovery mechanisms."""
    
    def __init__(self, job_id: str, db=None, max_retries: int = 3, 
                 retry_delay: float = 30.0):
        self.job_id = job_id
        self.db = db
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_count = 0
        self.last_error_time = None
        self.error_history = deque(maxlen=10)
        
        # Error classification
        self.recoverable_errors = [
            "CUDA out of memory",
            "RuntimeError",
            "OSError",
            "ConnectionError"
        ]
        
        self.non_recoverable_errors = [
            "ValueError",
            "KeyError",
            "AttributeError",
            "TypeError"
        ]
    
    def classify_error(self, exception: Exception) -> str:
        """Classify error type for appropriate handling."""
        error_message = str(exception)
        error_type = type(exception).__name__
        
        # Check for recoverable errors
        for recoverable_pattern in self.recoverable_errors:
            if recoverable_pattern in error_message or recoverable_pattern in error_type:
                return "recoverable"
        
        # Check for non-recoverable errors
        for non_recoverable_pattern in self.non_recoverable_errors:
            if non_recoverable_pattern in error_message or non_recoverable_pattern in error_type:
                return "non_recoverable"
        
        # Default classification
        return "unknown"
    
    def should_retry_training(self, exception: Exception) -> bool:
        """Determine if training should be retried."""
        error_classification = self.classify_error(exception)
        current_time = datetime.now()
        
        # Check retry count
        if self.retry_count >= self.max_retries:
            logger.warning(f"Max retries ({self.max_retries}) reached. Giving up.")
            return False
        
        # Check time since last error
        if self.last_error_time:
            time_since_last_error = (current_time - self.last_error_time).total_seconds()
            if time_since_last_error < self.retry_delay:
                logger.warning(f"Too soon to retry (wait {self.retry_delay - time_since_last_error:.0f}s more)")
                return False
        
        # Check error classification
        if error_classification == "non_recoverable":
            logger.warning("Non-recoverable error detected. Giving up.")
            return False
        
        return True
    
    def on_exception(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, exception: Exception, **kwargs):
        """Handle training exceptions with advanced recovery logic."""
        current_time = datetime.now()
        
        logger.error(f"Advanced error handling for job {self.job_id}: {str(exception)}")
        logger.error(f"Error classification: {self.classify_error(exception)}")
        logger.error(f"Retry count: {self.retry_count}/{self.max_retries}")
        
        # Store error in history
        self.error_history.append({
            "timestamp": current_time.isoformat(),
            "error": str(exception),
            "error_type": type(exception).__name__,
            "step": state.global_step if state else 0,
            "classification": self.classify_error(exception)
        })
        
        # Update error tracking
        self.last_error_time = current_time
        self.retry_count += 1
        
        # Determine if training should be retried
        if self.should_retry_training(exception):
            logger.info(f"Attempting retry #{self.retry_count} after {self.retry_delay}s delay")
            
            # Set retry delay
            time.sleep(self.retry_delay)
            
            # Reset control to continue training
            control.should_training_stop = False
            
            # Log retry attempt
            if self.db:
                try:
                    retry_data = {
                        "retry_attempt": self.retry_count,
                        "last_error": str(exception),
                        "retry_at": current_time.isoformat(),
                        "next_retry_at": (current_time + timedelta(seconds=self.retry_delay)).isoformat()
                    }
                    
                    self.db.training_jobs.update_one(
                        {"id": self.job_id},
                        {
                            "$set": {
                                "retry_status": retry_data,
                                "error_history": list(self.error_history)
                            }
                        }
                    )
                except Exception as e:
                    logger.error(f"Error updating retry status: {str(e)}")
            
        else:
            logger.error("Training will not be retried. Marking as failed.")
            
            # Update database with final failure status
            if self.db:
                try:
                    self.db.training_jobs.update_one(
                        {"id": self.job_id},
                        {
                            "$set": {
                                "status": "failed",
                                "error": str(exception),
                                "error_type": type(exception).__name__,
                                "failed_at": current_time.isoformat(),
                                "current_step": state.global_step if state else 0,
                                "error_history": list(self.error_history),
                                "final_retry_count": self.retry_count,
                                "error_classification": self.classify_error(exception)
                            }
                        }
                    )
                except Exception as e:
                    logger.error(f"Error updating database with failure status: {str(e)}")
            
            # Stop training
            control.should_training_stop = True
        
        return control

# Utility functions
def setup_enhanced_training_callbacks(
    job_id: str, 
    db=None, 
    websocket_manager=None,
    enable_real_time_monitoring: bool = True,
    update_interval: int = 50
) -> List[TrainerCallback]:
    """
    Setup enhanced training callbacks.
    
    Args:
        job_id: Training job ID
        db: Database instance
        websocket_manager: WebSocket manager for real-time updates
        enable_real_time_monitoring: Enable real-time monitoring
        update_interval: Update interval in steps
        
    Returns:
        List of callback instances
    """
    return [
        EnhancedQLoRATrainingCallback(
            job_id=job_id,
            db=db,
            websocket_manager=websocket_manager,
            update_interval=update_interval,
            enable_real_time_monitoring=enable_real_time_monitoring
        ),
        AdvancedErrorHandlingCallback(
            job_id=job_id,
            db=db,
            max_retries=3,
            retry_delay=30.0
        )
    ]

def get_training_metrics_summary(job_id: str, db=None) -> Dict[str, Any]:
    """
    Get training metrics summary for a job.
    
    Args:
        job_id: Training job ID
        db: Database instance
        
    Returns:
        Metrics summary
    """
    if not db:
        return {}
    
    try:
        # Get latest metrics
        loss_metrics = list(db.training_metrics.find(
            {"job_id": job_id, "metric_type": "loss"}
        ).sort("step", -1).limit(100))
        
        # Calculate summary statistics
        if loss_metrics:
            loss_values = [m["value"] for m in loss_metrics]
            
            return {
                "total_metrics": len(loss_metrics),
                "latest_loss": loss_metrics[0]["value"] if loss_metrics else None,
                "best_loss": min(loss_values) if loss_values else None,
                "loss_trend": "decreasing" if len(loss_values) > 1 and loss_values[0] < loss_values[-1] else "increasing",
                "average_loss": sum(loss_values) / len(loss_values) if loss_values else None,
                "loss_std": np.std(loss_values) if len(loss_values) > 1 else 0
            }
        
        return {"total_metrics": 0}
        
    except Exception as e:
        logger.error(f"Error getting training metrics summary: {str(e)}")
        return {"error": str(e)}