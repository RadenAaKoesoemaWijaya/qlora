import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

class QLoRATrainingCallback(TrainerCallback):
    """
    Custom callback untuk monitoring QLoRA training progress dan update database.
    """
    
    def __init__(self, job_id: str, db=None):
        self.job_id = job_id
        self.db = db
        self.start_time = datetime.now()
        self.step_count = 0
        self.epoch_count = 0
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                      control: TrainerControl, **kwargs):
        """Called when training begins."""
        logger.info(f"Training started for job: {self.job_id}")
        
        # Update database
        if self.db:
            try:
                self.db.training_jobs.update_one(
                    {"id": self.job_id},
                    {
                        "$set": {
                            "status": "training",
                            "started_at": datetime.now().isoformat(),
                            "total_steps": state.max_steps,
                            "current_step": 0,
                            "current_epoch": 0,
                            "progress": 0.0
                        }
                    }
                )
            except Exception as e:
                logger.error(f"Error updating database on_train_begin: {str(e)}")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, 
                   control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        self.step_count = state.global_step
        
        # Calculate progress
        progress = (state.global_step / state.max_steps * 100) if state.max_steps > 0 else 0
        
        # Log every 100 steps or at the end
        if state.global_step % 100 == 0 or state.global_step == state.max_steps:
            logger.info(f"Job {self.job_id}: Step {state.global_step}/{state.max_steps} ({progress:.1f}%)")
        
        # Update database every 50 steps
        if state.global_step % 50 == 0 and self.db:
            try:
                # Get current loss
                current_loss = None
                if state.log_history and len(state.log_history) > 0:
                    latest_logs = state.log_history[-1]
                    if "loss" in latest_logs:
                        current_loss = latest_logs["loss"]
                
                self.db.training_jobs.update_one(
                    {"id": self.job_id},
                    {
                        "$set": {
                            "current_step": state.global_step,
                            "progress": round(progress, 2),
                            "current_loss": current_loss,
                            "learning_rate": state.log_history[-1].get("learning_rate", 0) if state.log_history else 0
                        }
                    }
                )
                
                # Store training metrics
                if current_loss is not None:
                    metric = {
                        "job_id": self.job_id,
                        "step": state.global_step,
                        "epoch": state.epoch,
                        "loss": current_loss,
                        "learning_rate": state.log_history[-1].get("learning_rate", 0) if state.log_history else 0,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.db.training_metrics.insert_one(metric)
                    
            except Exception as e:
                logger.error(f"Error updating database on_step_end: {str(e)}")
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        """Called at the end of each epoch."""
        self.epoch_count = state.epoch
        
        logger.info(f"Job {self.job_id}: Epoch {state.epoch} completed")
        
        # Update database
        if self.db:
            try:
                self.db.training_jobs.update_one(
                    {"id": self.job_id},
                    {
                        "$set": {
                            "current_epoch": state.epoch,
                            "current_step": state.global_step
                        }
                    }
                )
            except Exception as e:
                logger.error(f"Error updating database on_epoch_end: {str(e)}")
    
    def on_save(self, args: TrainingArguments, state: TrainerState, 
               control: TrainerControl, **kwargs):
        """Called when model checkpoint is saved."""
        logger.info(f"Job {self.job_id}: Checkpoint saved at step {state.global_step}")
        
        # Create checkpoint record in database
        if self.db:
            try:
                current_loss = None
                if state.log_history and len(state.log_history) > 0:
                    latest_logs = state.log_history[-1]
                    if "loss" in latest_logs:
                        current_loss = latest_logs["loss"]
                
                checkpoint = {
                    "id": f"checkpoint_{self.job_id}_{state.global_step}",
                    "training_job_id": self.job_id,
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "loss": current_loss,
                    "created_at": datetime.now().isoformat(),
                    "model_name": f"checkpoint-{state.global_step}",
                    "path": f"./checkpoints/{self.job_id}/checkpoint-{state.global_step}"
                }
                
                self.db.checkpoints.insert_one(checkpoint)
                
            except Exception as e:
                logger.error(f"Error creating checkpoint record: {str(e)}")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        """Called when training ends."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() / 3600  # hours
        
        logger.info(f"Training completed for job: {self.job_id}")
        logger.info(f"Total duration: {duration:.2f} hours")
        logger.info(f"Total steps: {state.global_step}")
        
        # Update database
        if self.db:
            try:
                # Get final loss
                final_loss = None
                if state.log_history and len(state.log_history) > 0:
                    # Find the last loss value
                    for log_entry in reversed(state.log_history):
                        if "loss" in log_entry:
                            final_loss = log_entry["loss"]
                            break
                
                self.db.training_jobs.update_one(
                    {"id": self.job_id},
                    {
                        "$set": {
                            "status": "completed",
                            "progress": 100.0,
                            "current_step": state.global_step,
                            "final_loss": final_loss,
                            "completed_at": end_time.isoformat(),
                            "duration_hours": round(duration, 2)
                        }
                    }
                )
                
            except Exception as e:
                logger.error(f"Error updating database on_train_end: {str(e)}")
    
    def on_log(self, args: TrainingArguments, state: TrainerState, 
              control: TrainerControl, logs: Dict[str, Any], **kwargs):
        """Called when logs are produced."""
        # You can add custom logging logic here
        if "loss" in logs:
            logger.debug(f"Job {self.job_id} - Step {state.global_step}: loss={logs['loss']:.4f}")

class ErrorHandlingCallback(TrainerCallback):
    """Callback untuk handling errors during training."""
    
    def __init__(self, job_id: str, db=None):
        self.job_id = job_id
        self.db = db
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                      control: TrainerControl, **kwargs):
        """Setup error handling."""
        # Add custom error handling setup here
        pass
    
    def on_exception(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, exception: Exception, **kwargs):
        """Handle training exceptions."""
        logger.error(f"Training exception for job {self.job_id}: {str(exception)}")
        
        # Update database with error status
        if self.db:
            try:
                self.db.training_jobs.update_one(
                    {"id": self.job_id},
                    {
                        "$set": {
                            "status": "failed",
                            "error": str(exception),
                            "error_type": type(exception).__name__,
                            "failed_at": datetime.now().isoformat(),
                            "current_step": state.global_step if state else 0
                        }
                    }
                )
            except Exception as e:
                logger.error(f"Error updating database with failure status: {str(e)}")
        
        # Return control to stop training
        control.should_training_stop = True
        return control

# Utility function untuk setup callbacks
def setup_training_callbacks(job_id: str, db=None) -> list:
    """
    Setup all training callbacks.
    
    Args:
        job_id: Training job ID
        db: Database instance
        
    Returns:
        List of callback instances
    """
    return [
        QLoRATrainingCallback(job_id, db),
        ErrorHandlingCallback(job_id, db)
    ]