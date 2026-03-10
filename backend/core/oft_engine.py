"""
OFT (Orthogonal Fine-Tuning) Training Engine.

OFT menggunakan orthogonal transformation untuk fine-tuning:
- Menggunakan bentuk multiplicative: W = W0 ⊙ (orthogonal_matrix)
- Menjaga struktur geometris dari weight space
- Multiplicative updates yang lebih stabil
- Excellent untuk domain shift dan multi-task learning

Paper: "Controlling Text-to-Image Diffusion by Orthogonal Finetuning" (CVPR 2024)
"""

from typing import Dict, Any, Tuple, Optional
import torch
from peft import OFTConfig, get_peft_model, TaskType
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


class OFTTrainingEngine(BaseTrainingEngine):
    """
    Training engine untuk OFT (Orthogonal Fine-Tuning).
    
    Key Features:
    - Multiplicative updates: W = W0 ⊙ (I + ΔW) atau W = W0 ⊙ R
    - Orthogonal constraint: R^T R = I
    - Menjaga geometric structure dari pretrained weights
    - Multi-task friendly (orthogonal transformations tidak interfere)
    
    Benefits:
    - Stable training dengan multiplicative updates
    - Better untuk multimodal dan vision-language tasks
    - Geometric interpretability
    
    Hyperparameters:
    - r: rank untuk low-rank component (jika menggunakan OFT dengan low-rank)
    - module_dropout: dropout probability untuk OFT modules
    - init_weights: whether to initialize dengan orthogonal matrix
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Setup logging
        self.structured_logger = EnhancedLoggingSystem(
            component_name="OFTTrainingEngine",
            log_level="INFO",
            enable_file_logging=True,
            enable_console_logging=True,
            log_file_path="logs/oft_training_engine.log"
        )
        self.structured_logger.setup_logging()
    
    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """Setup quantization untuk OFT."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8
        )
    
    def setup_peft_config(self) -> OFTConfig:
        """
        Setup OFT configuration.
        
        OFT menggunakan orthogonal transformations pada weight matrices.
        """
        target_modules = self.config.get(
            "target_modules",
            ["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        # OFT specific parameters
        oft_r = self.config.get("oft_r", 8)
        oft_dropout = self.config.get("oft_dropout", 0.0)
        init_weights = self.config.get("oft_init_weights", True)
        
        oft_config = OFTConfig(
            r=oft_r,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            module_dropout=oft_dropout,
            init_weights=init_weights,
        )
        
        context = LogContext(
            component="OFTTrainingEngine",
            operation="setup_peft_config"
        )
        
        self.structured_logger.log(
            level="INFO",
            category=LogCategory.TRAINING,
            message=f"OFT configuration: r={oft_r}, dropout={oft_dropout}, init_weights={init_weights}",
            context=context,
            extra_data={
                "oft_r": oft_r,
                "oft_dropout": oft_dropout,
                "init_weights": init_weights,
                "target_modules": target_modules,
                "peft_type": "OFT"
            }
        )
        
        return oft_config
    
    async def load_model_and_tokenizer(self, model_id: str) -> Tuple[Any, Any]:
        """Load model dengan OFT."""
        context = LogContext(
            job_id=self.training_job_id,
            component="OFTTrainingEngine",
            operation="load_model_and_tokenizer"
        )
        
        try:
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"Loading model dengan OFT: {model_id}",
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
            
            # Apply OFT
            oft_config = self.setup_peft_config()
            self.model = get_peft_model(self.model, oft_config)
            
            # Log trainable parameters
            if hasattr(self.model, 'get_nb_trainable_parameters'):
                trainable, all_params = self.model.get_nb_trainable_parameters()
                self.structured_logger.log(
                    level="INFO",
                    category=LogCategory.TRAINING,
                    message=f"OFT model loaded: {trainable:,} trainable params ({100 * trainable / all_params:.4f}%)",
                    context=context,
                    extra_data={
                        "trainable_parameters": trainable,
                        "total_parameters": all_params,
                        "trainable_percentage": 100 * trainable / all_params,
                    }
                )
            
            return self.model, self.tokenizer
            
        except Exception as e:
            self.structured_logger.log(
                level="ERROR",
                category=LogCategory.TRAINING,
                message=f"Error loading model {model_id} dengan OFT: {str(e)}",
                context=context,
                exc_info=True
            )
            raise RuntimeError(f"Failed to load model dengan OFT: {str(e)}")
    
    def setup_training_arguments(self, job_id: str) -> TrainingArguments:
        """Setup training arguments untuk OFT."""
        output_dir = f"./checkpoints/{job_id}"
        from pathlib import Path
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
            evaluation_strategy="no",
            gradient_checkpointing=True,
            fp16=False,
            bf16=torch.cuda.is_bf16_supported(),
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="wandb" if self.config.get("use_wandb", True) else None,
            run_name=f"oft_{job_id}",
            load_best_model_at_end=False,
            optim="paged_adamw_32bit",
            max_grad_norm=0.3,
        )
    
    def create_trainer(self, model, tokenizer, train_dataset: Dataset, job_id: str) -> Trainer:
        """Create trainer untuk OFT."""
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
        """Start OFT training."""
        try:
            self.training_job_id = job_id
            
            context = LogContext(
                job_id=job_id,
                component="OFTTrainingEngine",
                operation="start_training"
            )
            
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"Starting OFT training with orthogonal transformations for job: {job_id}",
                context=context,
                extra_data={
                    "training_config": self.config,
                    "model_id": model_id,
                }
            )
            
            # Initialize wandb
            import wandb
            if self.config.get("use_wandb", True):
                wandb.init(
                    project=self.config.get("wandb_project", "oft-finetuning"),
                    name=f"training_{job_id}",
                    config=self.config,
                    tags=["oft", model_id.split("/")[-1]]
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
                message=f"OFT training completed. Model saved to: {final_model_path}",
                context=context
            )
            
            if self.config.get("use_wandb", True):
                wandb.finish()
            
            return {
                "status": "completed",
                "job_id": job_id,
                "model_path": final_model_path,
                "method": "OFT",
                "message": "OFT training completed successfully with orthogonal transformations"
            }
            
        except Exception as e:
            self.structured_logger.log(
                level="ERROR",
                category=LogCategory.TRAINING,
                message=f"OFT training failed for job {job_id}: {str(e)}",
                context=LogContext(job_id=job_id, component="OFTTrainingEngine", operation="start_training"),
                exc_info=True
            )
            
            import wandb
            if self.config.get("use_wandb", True):
                wandb.finish(exit_code=1)
            
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(e),
                "method": "OFT",
                "message": "OFT training failed"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get informasi model OFT."""
        if self.model is None:
            return {
                "status": "no_model_loaded",
                "method": "OFT",
            }
        
        info = {
            "status": "model_loaded",
            "method": "OFT",
            "device": str(self.device),
            "model_type": type(self.model).__name__,
            "config": self.config,
            "notes": "Orthogonal multiplicative updates"
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
            "name": "OFT",
            "full_name": "Orthogonal Fine-Tuning",
            "paper": "Controlling Text-to-Image Diffusion by Orthogonal Finetuning (CVPR 2024)",
            "key_benefits": [
                "Multiplicative updates (more stable)",
                "Geometric structure preservation",
                "Good for multimodal tasks",
                "Orthogonal constraint for stability"
            ],
            "use_cases": [
                "Multimodal fine-tuning",
                "Vision-language models",
                "Multi-task scenarios",
                "Domain adaptation"
            ],
            "peft_version": ">=0.7.0"
        }
