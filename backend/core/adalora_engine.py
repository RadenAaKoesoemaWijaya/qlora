"""
AdaLoRA (Adaptive Budget Allocation) Training Engine.

AdaLoRA mengalokasikan budget parameter secara dinamis selama training:
- Menggunakan SVD-based adaptation (singular value decomposition)
- Budget rank dialokasikan adaptif berdasarkan importance scores
- Menggabungkan pruning dan growing selama training
- Budget awareness untuk kontrol total parameter

Paper: "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning" (ICLR 2023)
"""

from typing import Dict, Any, Tuple, Optional
import torch
from peft import AdaLoraConfig, get_peft_model, TaskType
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


class AdaLoRATrainingEngine(BaseTrainingEngine):
    """
    Training engine untuk AdaLoRA dengan adaptive budget allocation.
    
    Key Features:
    - SVD-based adaptation: W = W0 + P Λ Q^T
    - P dan Q adalah orthogonal matrices
    - Λ adalah diagonal singular values matrix
    - Budget dialokasikan secara adaptif ke layer yang lebih penting
    
    Hyperparameters:
    - target_r: target budget rank (final rank setelah pruning)
    - init_r: initial rank
    - tinit: warmup steps sebelum mulai pruning
    - tfinal: steps saat pruning selesai
    - deltaT: interval antar pruning steps
    - beta1, beta2: exponential decay factors untuk importance scores
    
    Benefits:
    - Dynamic budget allocation ke layer yang paling penting
    - SOTA performance pada banyak benchmarks
    - Budget-aware: kontrol total parameter yang di-train
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Setup logging
        self.structured_logger = EnhancedLoggingSystem(
            component_name="AdaLoRATrainingEngine",
            log_level="INFO",
            enable_file_logging=True,
            enable_console_logging=True,
            log_file_path="logs/adalora_training_engine.log"
        )
        self.structured_logger.setup_logging()
        
        # AdaLoRA specific config dengan defaults yang reasonable
        self.init_r = config.get("adalora_init_r", 12)  # Initial rank
        self.target_r = config.get("adalora_target_r", 4)  # Target budget rank
        self.total_step = config.get("num_epochs", 3) * 1000  # Estimate total steps
        self.tinit = config.get("adalora_tinit", 0)  # Warmup steps
        self.tfinal = config.get("adalora_tfinal", self.total_step)  # Pruning end
        self.deltaT = config.get("adalora_deltaT", 10)  # Pruning interval
        self.beta1 = config.get("adalora_beta1", 0.85)  # Decay factor 1
        self.beta2 = config.get("adalora_beta2", 0.85)  # Decay factor 2
        self.orth_reg_weight = config.get("adalora_orth_reg_weight", 0.5)  # Orthogonality reg
        
        # Validate config
        self._validate_config()
    
    def _validate_config(self):
        """Validasi dan adjust AdaLoRA config."""
        # Target rank harus <= initial rank
        if self.target_r > self.init_r:
            self.structured_logger.log(
                level="WARNING",
                category=LogCategory.TRAINING,
                message=f"target_r ({self.target_r}) > init_r ({self.init_r}), setting target_r = init_r",
                context=LogContext(component="AdaLoRATrainingEngine", operation="_validate_config")
            )
            self.target_r = self.init_r
        
        # tfinal harus > tinit
        if self.tfinal <= self.tinit:
            self.tfinal = self.tinit + 100
    
    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """Setup quantization untuk AdaLoRA."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8
        )
    
    def setup_peft_config(self) -> AdaLoraConfig:
        """
        Setup AdaLoRA configuration dengan adaptive budget.
        
        Perbedaan dengan LoRA biasa:
        - Menggunakan SVD: W = W0 + P Λ Q^T
        - P dan Q trainable, Λ mengandung singular values
        - Budget pruning/growing selama training
        """
        target_modules = self.config.get(
            "target_modules",
            ["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        adalora_config = AdaLoraConfig(
            peft_type="ADALORA",
            r=self.init_r,
            target_modules=target_modules,
            lora_alpha=self.config.get("lora_alpha", 32),
            target_r=self.target_r,
            init_r=self.init_r,
            tinit=self.tinit,
            tfinal=self.tfinal,
            deltaT=self.deltaT,
            beta1=self.beta1,
            beta2=self.beta2,
            dropout=self.config.get("lora_dropout", 0.05),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            # Orthogonality regularization untuk menjaga P dan Q orthogonal
            # Note: Ini di-handle oleh AdaLoRA algorithm
        )
        
        context = LogContext(
            component="AdaLoRATrainingEngine",
            operation="setup_peft_config"
        )
        
        self.structured_logger.log(
            level="INFO",
            category=LogCategory.TRAINING,
            message=f"AdaLoRA config: init_r={self.init_r}, target_r={self.target_r}, "
                   f"tinit={self.tinit}, tfinal={self.tfinal}, deltaT={self.deltaT}",
            context=context,
            extra_data={
                "init_r": self.init_r,
                "target_r": self.target_r,
                "tinit": self.tinit,
                "tfinal": self.tfinal,
                "deltaT": self.deltaT,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "target_modules": target_modules,
                "peft_type": "ADALORA"
            }
        )
        
        return adalora_config
    
    async def load_model_and_tokenizer(self, model_id: str) -> Tuple[Any, Any]:
        """Load model dengan AdaLoRA."""
        context = LogContext(
            job_id=self.training_job_id,
            component="AdaLoRATrainingEngine",
            operation="load_model_and_tokenizer"
        )
        
        try:
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"Loading model dengan AdaLoRA (init_r={self.init_r}, target_r={self.target_r}): {model_id}",
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
            
            # Apply AdaLoRA
            adalora_config = self.setup_peft_config()
            self.model = get_peft_model(self.model, adalora_config)
            
            # Log trainable parameters
            if hasattr(self.model, 'get_nb_trainable_parameters'):
                trainable, all_params = self.model.get_nb_trainable_parameters()
                self.structured_logger.log(
                    level="INFO",
                    category=LogCategory.TRAINING,
                    message=f"AdaLoRA model loaded: {trainable:,} trainable params ({100 * trainable / all_params:.4f}%)",
                    context=context,
                    extra_data={
                        "trainable_parameters": trainable,
                        "total_parameters": all_params,
                        "trainable_percentage": 100 * trainable / all_params,
                        "init_r": self.init_r,
                        "target_r": self.target_r,
                    }
                )
            
            return self.model, self.tokenizer
            
        except Exception as e:
            self.structured_logger.log(
                level="ERROR",
                category=LogCategory.TRAINING,
                message=f"Error loading model {model_id} dengan AdaLoRA: {str(e)}",
                context=context,
                exc_info=True
            )
            raise RuntimeError(f"Failed to load model dengan AdaLoRA: {str(e)}")
    
    def setup_training_arguments(self, job_id: str) -> TrainingArguments:
        """Setup training arguments dengan consideration untuk AdaLoRA pruning schedule."""
        output_dir = f"./checkpoints/{job_id}"
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Hitung steps untuk tfinal berdasarkan dataset size
        num_epochs = self.config.get("num_epochs", 3)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.config.get("batch_size", 2),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            learning_rate=self.config.get("learning_rate", 2e-4),
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
            run_name=f"adalora_{job_id}",
            load_best_model_at_end=False,
            optim="paged_adamw_32bit",
            max_grad_norm=0.3,
        )
        
        return training_args
    
    def create_trainer(self, model, tokenizer, train_dataset: Dataset, job_id: str) -> Trainer:
        """Create trainer untuk AdaLoRA dengan custom callback untuk monitoring budget."""
        training_args = self.setup_training_arguments(job_id)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Custom callback untuk AdaLoRA budget monitoring
        from .training_callback import QLoRATrainingCallback
        from transformers import TrainerCallback
        
        class AdaLoRABudgetCallback(TrainerCallback):
            """Callback untuk monitor AdaLoRA budget allocation."""
            
            def __init__(self, logger):
                self.logger = logger
                self.step_count = 0
            
            def on_step_end(self, args, state, control, **kwargs):
                self.step_count = state.global_step
                # Log budget info setiap beberapa steps
                if self.step_count % 100 == 0:
                    self.logger.log(
                        level="INFO",
                        category=LogCategory.TRAINING,
                        message=f"AdaLoRA step {self.step_count}: Pruning schedule active",
                        context=LogContext(component="AdaLoRATrainingEngine", operation="training_step")
                    )
        
        budget_callback = AdaLoRABudgetCallback(self.structured_logger)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[
                QLoRATrainingCallback(job_id, self.config.get("db")),
                budget_callback
            ]
        )
        
        return trainer
    
    async def start_training(self, job_id: str, model_id: str, train_dataset: Dataset, db=None) -> Dict[str, Any]:
        """Start AdaLoRA training dengan adaptive budget."""
        try:
            self.training_job_id = job_id
            
            context = LogContext(
                job_id=job_id,
                component="AdaLoRATrainingEngine",
                operation="start_training"
            )
            
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"Starting AdaLoRA training with adaptive budget: init_r={self.init_r} -> target_r={self.target_r}",
                context=context,
                extra_data={
                    "training_config": self.config,
                    "model_id": model_id,
                    "init_r": self.init_r,
                    "target_r": self.target_r,
                    "pruning_schedule": f"{self.tinit} to {self.tfinal}, every {self.deltaT} steps"
                }
            )
            
            # Initialize wandb
            import wandb
            if self.config.get("use_wandb", True):
                wandb.init(
                    project=self.config.get("wandb_project", "adalora-finetuning"),
                    name=f"training_{job_id}",
                    config=self.config,
                    tags=["adalora", f"r{self.init_r}_to_{self.target_r}", model_id.split("/")[-1]]
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
                message=f"AdaLoRA training completed. Model saved to: {final_model_path}",
                context=context
            )
            
            if self.config.get("use_wandb", True):
                wandb.finish()
            
            return {
                "status": "completed",
                "job_id": job_id,
                "model_path": final_model_path,
                "method": "AdaLoRA",
                "init_r": self.init_r,
                "target_r": self.target_r,
                "message": "AdaLoRA training completed successfully with adaptive budget allocation"
            }
            
        except Exception as e:
            self.structured_logger.log(
                level="ERROR",
                category=LogCategory.TRAINING,
                message=f"AdaLoRA training failed for job {job_id}: {str(e)}",
                context=LogContext(job_id=job_id, component="AdaLoRATrainingEngine", operation="start_training"),
                exc_info=True
            )
            
            import wandb
            if self.config.get("use_wandb", True):
                wandb.finish(exit_code=1)
            
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(e),
                "method": "AdaLoRA",
                "message": "AdaLoRA training failed"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get informasi model AdaLoRA dengan budget info."""
        if self.model is None:
            return {
                "status": "no_model_loaded",
                "method": "AdaLoRA",
                "init_r": self.init_r,
                "target_r": self.target_r,
            }
        
        info = {
            "status": "model_loaded",
            "method": "AdaLoRA",
            "device": str(self.device),
            "model_type": type(self.model).__name__,
            "init_r": self.init_r,
            "target_r": self.target_r,
            "budget_reduction": f"{(1 - self.target_r/self.init_r)*100:.1f}%",
            "config": self.config,
            "notes": "SVD-based adaptive budget allocation"
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
            "name": "AdaLoRA",
            "full_name": "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning",
            "paper": "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning (ICLR 2023)",
            "key_benefits": [
                "Dynamic budget allocation",
                "Allocates more parameters to important layers",
                "Budget-aware parameter control",
                "SOTA performance"
            ],
            "use_cases": [
                "Complex fine-tuning tasks",
                "Budget optimization",
                "When layer importance varies",
                "Multi-task scenarios"
            ],
            "hyperparameters": {
                "init_r": "Initial rank (e.g., 12)",
                "target_r": "Target rank after pruning (e.g., 4)",
                "tinit": "Warmup steps before pruning",
                "tfinal": "Step when pruning completes",
                "deltaT": "Pruning interval in steps",
                "beta1": "Importance score decay factor"
            },
            "peft_version": ">=0.5.0"
        }
