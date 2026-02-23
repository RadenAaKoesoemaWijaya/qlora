import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import wandb
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

from .enhanced_logging_system import EnhancedLoggingSystem, LogCategory, LogContext

class QLoRATrainingEngine:
    """
    Real QLoRA training engine untuk fine-tuning model LLM dengan quantization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_job_id = None
        
        # Setup structured logging
        self.structured_logger = EnhancedLoggingSystem(
            component_name="QLoRATrainingEngine",
            log_level="INFO",
            enable_file_logging=True,
            enable_console_logging=True,
            log_file_path="logs/training_engine.log"
        )
        self.structured_logger.setup_logging()
        
    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """Setup 4-bit quantization configuration."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8
        )
    
    def setup_lora_config(self) -> LoraConfig:
        """Setup LoRA configuration."""
        return LoraConfig(
            r=self.config.get("lora_rank", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            target_modules=self.config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
    
    async def load_model_and_tokenizer(self, model_id: str) -> tuple:
        """
        Load model dan tokenizer dengan quantization.
        
        Args:
            model_id: Hugging Face model ID (misalnya: "meta-llama/Llama-2-7b-hf")
            
        Returns:
            Tuple (model, tokenizer)
        """
        try:
            # Create log context
            context = LogContext(
                job_id=self.training_job_id,
                component="QLoRATrainingEngine",
                operation="load_model_and_tokenizer"
            )
            
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"Loading model: {model_id}",
                context=context
            )
            
            # Setup quantization
            bnb_config = self.setup_quantization_config()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # Add padding token jika belum ada
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model dengan quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"  # Optimasi memory
            )
            
            # Enable gradient checkpointing untuk efisiensi memory
            self.model.gradient_checkpointing_enable()
            
            # Prepare model untuk k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Add LoRA adapters
            lora_config = self.setup_lora_config()
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable_params, all_param = self.model.get_nb_trainable_parameters()
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"Trainable parameters: {trainable_params:,} / {all_param:,} ({100 * trainable_params / all_param:.2f}%)",
                context=context
            )
            
            return self.model, self.tokenizer
            
        except Exception as e:
            self.structured_logger.log(
                level="ERROR",
                category=LogCategory.TRAINING,
                message=f"Error loading model {model_id}: {str(e)}",
                context=context,
                exc_info=True
            )
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def setup_training_arguments(self, job_id: str) -> TrainingArguments:
        """Setup training arguments untuk QLoRA training."""
        output_dir = f"./checkpoints/{job_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 2),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            learning_rate=self.config.get("learning_rate", 2e-4),
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            evaluation_strategy="no",  # Bisa diubah jika ada validation set
            gradient_checkpointing=True,
            fp16=False,  # Sudah menggunakan 4-bit quantization
            bf16=torch.cuda.is_bf16_supported(),
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="wandb" if self.config.get("use_wandb", True) else None,
            run_name=f"qlora_{job_id}",
            load_best_model_at_end=False,  # Tidak perlu untuk sekarang
            optim="paged_adamw_32bit",  # Optimizer untuk quantized training
            max_grad_norm=0.3,
            group_by_length=True,  # Efisiensi training
            length_column_name="length",
        )
    
    def create_trainer(self, 
                      model, 
                      tokenizer, 
                      train_dataset: Dataset,
                      job_id: str) -> Trainer:
        """Create trainer instance untuk QLoRA training."""
        
        training_args = self.setup_training_arguments(job_id)
        
        # Data collator untuk language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Kita tidak menggunakan masked language modeling
            pad_to_multiple_of=8  # Optimasi untuk tensor cores
        )
        
        # Custom callback untuk monitoring
        from .training_callback import QLoRATrainingCallback
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[QLoRATrainingCallback(job_id, self.config.get("db"))]
        )
        
        return trainer
    
    async def start_training(self, 
                           job_id: str,
                           model_id: str,
                           train_dataset: Dataset,
                           db=None) -> Dict[str, Any]:
        """
        Start QLoRA training dengan model dan dataset yang sudah diproses.
        
        Args:
            job_id: Unique training job ID
            model_id: Hugging Face model ID
            train_dataset: Processed training dataset
            db: Database instance untuk update progress
            
        Returns:
            Dictionary dengan training status dan informasi
        """
        try:
            self.training_job_id = job_id
            
            # Create log context for training
            context = LogContext(
                job_id=job_id,
                component="QLoRATrainingEngine",
                operation="start_training"
            )
            
            # Log training start with configuration
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"Starting QLoRA training for job: {job_id}",
                context=context,
                extra_data={"training_config": self.config, "model_id": model_id}
            )
            
            # Initialize Weights & Biases jika diaktifkan
            if self.config.get("use_wandb", True):
                wandb.init(
                    project=self.config.get("wandb_project", "qlora-finetuning"),
                    name=f"training_{job_id}",
                    config=self.config,
                    tags=["qlora", model_id.split("/")[-1]]
                )
            
            # Load model dan tokenizer
            model, tokenizer = await self.load_model_and_tokenizer(model_id)
            
            # Create trainer
            trainer = self.create_trainer(model, tokenizer, train_dataset, job_id)
            
            # Start training
            trainer.train()
            
            # Save final model
            final_model_path = f"./models/{job_id}/final"
            Path(final_model_path).mkdir(parents=True, exist_ok=True)
            
            # Save LoRA adapters
            model.save_pretrained(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            
            # Save training configuration
            config_path = Path(final_model_path) / "training_config.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"Training completed successfully. Model saved to: {final_model_path}",
                context=context
            )
            
            # Finalize W&B
            if self.config.get("use_wandb", True):
                wandb.finish()
            
            return {
                "status": "completed",
                "job_id": job_id,
                "model_path": final_model_path,
                "message": "Training completed successfully"
            }
            
        except Exception as e:
            self.structured_logger.log(
                level="ERROR",
                category=LogCategory.TRAINING,
                message=f"Training failed for job {job_id}: {str(e)}",
                context=context,
                exc_info=True
            )
            
            # Finalize W&B dengan status failed
            if self.config.get("use_wandb", True):
                wandb.finish(exit_code=1)
            
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(e),
                "message": "Training failed"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get informasi model yang sedang digunakan."""
        if self.model is None:
            return {"status": "no_model_loaded"}
        
        return {
            "status": "model_loaded",
            "device": str(self.device),
            "model_type": type(self.model).__name__,
            "config": self.config,
            "trainable_parameters": self.model.get_nb_trainable_parameters() if hasattr(self.model, 'get_nb_trainable_parameters') else None
        }

# Utility function untuk load model yang sudah difinetune
def load_qlora_model(model_path: str, base_model_id: str) -> tuple:
    """
    Load model yang sudah difinetune dengan QLoRA.
    
    Args:
        model_path: Path ke LoRA adapters
        base_model_id: Hugging Face model ID untuk base model
        
    Returns:
        Tuple (model, tokenizer)
    """
    try:
        # Setup quantization untuk inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load base model dengan quantization
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapters
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
        return model, tokenizer
        
    except Exception as e:
        # Create log context for utility function
        context = LogContext(
            component="QLoRATrainingEngine",
            operation="load_qlora_model"
        )
        
        # Note: For utility function, we need to create a logger instance
        temp_logger = EnhancedLoggingSystem(
            component_name="QLoRATrainingEngine",
            log_level="ERROR",
            enable_console_logging=True
        )
        temp_logger.setup_logging()
        
        temp_logger.log(
            level="ERROR",
            category=LogCategory.TRAINING,
            message=f"Error loading QLoRA model: {str(e)}",
            context=context,
            exc_info=True
        )
        raise RuntimeError(f"Failed to load QLoRA model: {str(e)}")