"""
VeRA (Vector-based Random Matrix Adaptation) Training Engine.

VeRA menggunakan random frozen projections dan hanya meng-train scaling vectors:
- Matriks A dan B diinisialisasi random dan dibekukan (frozen)
- Hanya trainable vectors b dan d (scaling factors)
- Parameter count: ~10,000x lebih sedikit dari LoRA

Ideal untuk edge deployment dan extreme resource constraints.

Paper: "VeRA: Vector-based Random Matrix Adaptation" (2024)
"""

from typing import Dict, Any, Tuple, Optional
import torch
from peft import VeraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset

from .base_engine import BaseTrainingEngine
from .enhanced_logging_system import EnhancedLoggingSystem, LogContext, LogCategory


class VeRATrainingEngine(BaseTrainingEngine):
    """
    Training engine untuk VeRA (Vector-based Random Matrix Adaptation).
    
    VeRA Architecture:
    - Frozen random matrices A and B (initialized once, never updated)
    - Trainable scaling vectors b and d
    - Update: W = W0 + Λ_b B A Λ_d
    - dimana Λ_b dan Λ_d adalah diagonal matrices dari learned vectors
    
    Benefits:
    - Extreme parameter efficiency (~1/1000 of LoRA)
    - Great for multiple adapters (shared frozen matrices)
    - Minimal storage for each adapter
    - Good for edge deployment
    
    Constraints:
    - Requires PEFT >= 0.10.0
    - Slightly lower performance than LoRA on some tasks
    - Larger rank recommended (256-1024) karena matrices frozen
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Setup structured logging
        self.structured_logger = EnhancedLoggingSystem(
            component_name="VeRATrainingEngine",
            log_level="INFO",
            enable_file_logging=True,
            enable_console_logging=True,
            log_file_path="logs/vera_training_engine.log"
        )
        self.structured_logger.setup_logging()
        
        # VeRA specific config
        self.vera_rank = config.get("vera_rank", 256)  # Lebih besar karena frozen
        self.vera_seed = config.get("vera_seed", 42)  # Untuk reproducibility
    
    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """Setup quantization untuk VeRA."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8
        )
    
    def setup_peft_config(self) -> VeraConfig:
        """
        Setup VeRA configuration.
        
        Perbedaan dengan LoRA:
        - Rank bisa lebih besar (256-1024) karena frozen
        - Seed untuk reproducible random initialization
        - Hanya scaling vectors yang trainable
        """
        target_modules = self.config.get(
            "target_modules",
            ["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        # Validate rank (VeRA bisa menggunakan rank besar)
        if self.vera_rank < 64:
            self.structured_logger.log(
                level="WARNING",
                category=LogCategory.TRAINING,
                message=f"VeRA rank {self.vera_rank} < 64, increasing to 64 for better performance",
                context=LogContext(component="VeRATrainingEngine", operation="setup_peft_config")
            )
            self.vera_rank = 64
        
        vera_config = VeraConfig(
            r=self.vera_rank,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            vera_dropout=self.config.get("vera_dropout", 0.0),
            projection_prng_key=self.vera_seed,  # Seed untuk reproducibility
        )
        
        context = LogContext(
            component="VeRATrainingEngine",
            operation="setup_peft_config"
        )
        
        self.structured_logger.log(
            level="INFO",
            category=LogCategory.TRAINING,
            message=f"VeRA configuration: rank={self.vera_rank}, seed={self.vera_seed}, {len(target_modules)} target modules",
            context=context,
            extra_data={
                "vera_rank": self.vera_rank,
                "vera_seed": self.vera_seed,
                "target_modules": target_modules,
                "peft_type": "VERA"
            }
        )
        
        return vera_config
    
    async def load_model_and_tokenizer(self, model_id: str) -> Tuple[Any, Any]:
        """Load model dengan VeRA."""
        context = LogContext(
            job_id=self.training_job_id,
            component="VeRATrainingEngine",
            operation="load_model_and_tokenizer"
        )
        
        try:
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"Loading model dengan VeRA (rank={self.vera_rank}): {model_id}",
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
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            
            # Enable gradient checkpointing
            self.model.gradient_checkpointing_enable()
            
            # Apply VeRA
            vera_config = self.setup_peft_config()
            self.model = get_peft_model(self.model, vera_config)
            
            # Log trainable parameters
            if hasattr(self.model, 'get_nb_trainable_parameters'):
                trainable, all_params = self.model.get_nb_trainable_parameters()
                self.structured_logger.log(
                    level="INFO",
                    category=LogCategory.TRAINING,
                    message=f"VeRA model loaded: {trainable:,} trainable params ({100 * trainable / all_params:.6f}%) - Ultra efficient!",
                    context=context,
                    extra_data={
                        "trainable_parameters": trainable,
                        "total_parameters": all_params,
                        "trainable_percentage": 100 * trainable / all_params,
                        "vera_rank": self.vera_rank,
                    }
                )
            
            return self.model, self.tokenizer
            
        except Exception as e:
            self.structured_logger.log(
                level="ERROR",
                category=LogCategory.TRAINING,
                message=f"Error loading model {model_id} dengan VeRA: {str(e)}",
                context=context,
                exc_info=True
            )
            raise RuntimeError(f"Failed to load model dengan VeRA: {str(e)}")
    
    def setup_training_arguments(self, job_id: str) -> TrainingArguments:
        """Setup training arguments untuk VeRA."""
        output_dir = f"./checkpoints/{job_id}"
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 2),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            learning_rate=self.config.get("learning_rate", 1e-2),  # VeRA bisa menggunakan LR tinggi
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            evaluation_strategy="no",
            gradient_checkpointing=True,
            fp16=False,
            bf16=torch.cuda.is_bf16_supported(),
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="wandb" if self.config.get("use_wandb", True) else None,
            run_name=f"vera_{job_id}",
            load_best_model_at_end=False,
            optim="paged_adamw_32bit",
            max_grad_norm=0.3,
        )
    
    def create_trainer(self, model, tokenizer, train_dataset: Dataset, job_id: str) -> Trainer:
        """Create trainer untuk VeRA."""
        training_args = self.setup_training_arguments(job_id)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
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
    
    async def start_training(self, job_id: str, model_id: str, train_dataset: Dataset, db=None) -> Dict[str, Any]:
        """Start VeRA training."""
        try:
            self.training_job_id = job_id
            
            context = LogContext(
                job_id=job_id,
                component="VeRATrainingEngine",
                operation="start_training"
            )
            
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"Starting VeRA training (rank={self.vera_rank}) for job: {job_id}",
                context=context,
                extra_data={
                    "training_config": self.config,
                    "model_id": model_id,
                    "vera_rank": self.vera_rank,
                    "vera_seed": self.vera_seed,
                }
            )
            
            # Initialize wandb
            import wandb
            if self.config.get("use_wandb", True):
                wandb.init(
                    project=self.config.get("wandb_project", "vera-finetuning"),
                    name=f"training_{job_id}",
                    config=self.config,
                    tags=["vera", f"rank_{self.vera_rank}", model_id.split("/")[-1]]
                )
            
            # Load model
            model, tokenizer = await self.load_model_and_tokenizer(model_id)
            
            # Create trainer
            trainer = self.create_trainer(model, tokenizer, train_dataset, job_id)
            
            # Train
            trainer.train()
            
            # Save
            final_model_path = f"./models/{job_id}/final"
            from pathlib import Path
            Path(final_model_path).mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            
            # Save config
            import json
            config_path = Path(final_model_path) / "training_config.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"VeRA training completed. Model saved to: {final_model_path}",
                context=context
            )
            
            if self.config.get("use_wandb", True):
                wandb.finish()
            
            return {
                "status": "completed",
                "job_id": job_id,
                "model_path": final_model_path,
                "method": "VeRA",
                "vera_rank": self.vera_rank,
                "message": "VeRA training completed successfully - Ultra parameter efficient!"
            }
            
        except Exception as e:
            self.structured_logger.log(
                level="ERROR",
                category=LogCategory.TRAINING,
                message=f"VeRA training failed for job {job_id}: {str(e)}",
                context=LogContext(job_id=job_id, component="VeRATrainingEngine", operation="start_training"),
                exc_info=True
            )
            
            import wandb
            if self.config.get("use_wandb", True):
                wandb.finish(exit_code=1)
            
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(e),
                "method": "VeRA",
                "message": "VeRA training failed"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get informasi model VeRA."""
        if self.model is None:
            return {
                "status": "no_model_loaded",
                "method": "VeRA",
                "vera_rank": self.vera_rank,
            }
        
        info = {
            "status": "model_loaded",
            "method": "VeRA",
            "device": str(self.device),
            "model_type": type(self.model).__name__,
            "vera_rank": self.vera_rank,
            "vera_seed": self.vera_seed,
            "config": self.config,
            "notes": "Ultra parameter-efficient: only scaling vectors are trainable"
        }
        
        if hasattr(self.model, 'get_nb_trainable_parameters'):
            trainable, all_params = self.model.get_nb_trainable_parameters()
            info["trainable_parameters"] = trainable
            info["total_parameters"] = all_params
            info["trainable_percentage"] = 100 * trainable / all_params
        
        return info
    
    def get_trainable_parameters(self) -> Optional[Tuple[int, int]]:
        """Get jumlah trainable parameters."""
        if self.model and hasattr(self.model, 'get_nb_trainable_parameters'):
            return self.model.get_nb_trainable_parameters()
        return None
    
    @staticmethod
    def get_method_description() -> Dict[str, Any]:
        """Get static description."""
        return {
            "name": "VeRA",
            "full_name": "Vector-based Random Matrix Adaptation",
            "paper": "VeRA: Vector-based Random Matrix Adaptation (2024)",
            "key_benefits": [
                "~10,000x fewer parameters than LoRA",
                "Ideal for edge deployment",
                "Minimal storage per adapter",
                "Good for multiple adapters"
            ],
            "use_cases": [
                "Edge deployment",
                "Multiple adapter storage",
                "Extreme resource constraints",
                "IoT devices"
            ],
            "recommended_rank": 256,
            "rank_range": "64-1024",
            "peft_version": ">=0.10.0"
        }
