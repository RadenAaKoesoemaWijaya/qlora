"""
IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) Training Engine.

IA³ menambahkan learned scaling vectors (element-wise multiplication) pada:
- Key activations (k_proj)
- Value activations (v_proj)  
- Feedforward activations (down_proj)

Jumlah parameter lebih sedikit dari LoRA tapi performa lebih baik pada banyak task.
Inference lebih cepat karena hanya perkalian element-wise.

Paper: "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning"
"""

from typing import Dict, Any, Tuple, Optional
import torch
from peft import IA3Config, get_peft_model, TaskType
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


class IA3TrainingEngine(BaseTrainingEngine):
    """
    Training engine untuk IA³ (Infused Adapter).
    
    IA³ menggunakan element-wise scaling vectors:
    - W_k = W_k0 ⊙ ( learned_vector_k )  -- key projection scaling
    - W_v = W_v0 ⊙ ( learned_vector_v )  -- value projection scaling  
    - W_ff = W_ff0 ⊙ ( learned_vector_ff )  -- feedforward scaling
    
    Key Differences dari LoRA:
    - No low-rank decomposition
    - Element-wise multiplication (hadamard product)
    - Fewer parameters than LoRA
    - Faster inference (no additional matrix multiplication)
    
    Recommended target_modules:
    - ["k_proj", "v_proj"] untuk attention-focused tasks
    - ["k_proj", "v_proj", "down_proj"] untuk general tasks
    - ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] untuk comprehensive
    
    feedforward_modules:
    - Harus overlap dengan target_modules yang merupakan feedforward
    - Biasanya ["down_proj"] atau ["gate_proj", "up_proj", "down_proj"]
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Setup structured logging
        self.structured_logger = EnhancedLoggingSystem(
            component_name="IA3TrainingEngine",
            log_level="INFO",
            enable_file_logging=True,
            enable_console_logging=True,
            log_file_path="logs/ia3_training_engine.log"
        )
        self.structured_logger.setup_logging()
    
    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """Setup 4-bit quantization configuration untuk IA³."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8
        )
    
    def setup_peft_config(self) -> IA3Config:
        """
        Setup IA³ configuration.
        
        IA³ menggunakan element-wise scaling vectors, bukan low-rank matrices.
        """
        # Target modules untuk scaling
        target_modules = self.config.get(
            "target_modules", 
            ["k_proj", "v_proj", "down_proj"]  # Default: key, value, dan feedforward
        )
        
        # Feedforward modules (subset dari target_modules yang merupakan feedforward)
        feedforward_modules = self.config.get(
            "feedforward_modules",
            ["down_proj"]  # Default: only down_proj
        )
        
        # Validate: feedforward_modules harus subset dari target_modules
        invalid_modules = set(feedforward_modules) - set(target_modules)
        if invalid_modules:
            self.structured_logger.log(
                level="WARNING",
                category=LogCategory.TRAINING,
                message=f"feedforward_modules {invalid_modules} not in target_modules, removing",
                context=LogContext(component="IA3TrainingEngine", operation="setup_peft_config")
            )
            feedforward_modules = [m for m in feedforward_modules if m in target_modules]
        
        ia3_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            feedforward_modules=feedforward_modules,
            inference_mode=False,
        )
        
        # Log config
        context = LogContext(
            component="IA3TrainingEngine",
            operation="setup_peft_config"
        )
        
        self.structured_logger.log(
            level="INFO",
            category=LogCategory.TRAINING,
            message=f"IA³ configuration: {len(target_modules)} target modules, {len(feedforward_modules)} feedforward modules",
            context=context,
            extra_data={
                "target_modules": target_modules,
                "feedforward_modules": feedforward_modules,
                "peft_type": "IA3"
            }
        )
        
        return ia3_config
    
    async def load_model_and_tokenizer(self, model_id: str) -> Tuple[Any, Any]:
        """
        Load model dan tokenizer dengan IA³ configuration.
        """
        context = LogContext(
            job_id=self.training_job_id,
            component="IA3TrainingEngine",
            operation="load_model_and_tokenizer"
        )
        
        try:
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"Loading model dengan IA³: {model_id}",
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
            
            # Load model dengan quantization
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
            
            # Apply IA³
            ia3_config = self.setup_peft_config()
            self.model = get_peft_model(self.model, ia3_config)
            
            # Log trainable parameters
            if hasattr(self.model, 'get_nb_trainable_parameters'):
                trainable, all_params = self.model.get_nb_trainable_parameters()
                self.structured_logger.log(
                    level="INFO",
                    category=LogCategory.TRAINING,
                    message=f"IA³ model loaded: {trainable:,} trainable params ({100 * trainable / all_params:.4f}%)",
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
                message=f"Error loading model {model_id} dengan IA³: {str(e)}",
                context=context,
                exc_info=True
            )
            raise RuntimeError(f"Failed to load model dengan IA³: {str(e)}")
    
    def setup_training_arguments(self, job_id: str) -> TrainingArguments:
        """Setup training arguments untuk IA³."""
        output_dir = f"./checkpoints/{job_id}"
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 2),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            learning_rate=self.config.get("learning_rate", 1e-3),  # IA³ bisa menggunakan LR lebih tinggi
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
            run_name=f"ia3_{job_id}",
            load_best_model_at_end=False,
            optim="paged_adamw_32bit",
            max_grad_norm=0.3,
            group_by_length=True,
        )
    
    def create_trainer(self, model, tokenizer, train_dataset: Dataset, job_id: str) -> Trainer:
        """Create trainer instance untuk IA³."""
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
        """Start IA³ training."""
        try:
            self.training_job_id = job_id
            
            context = LogContext(
                job_id=job_id,
                component="IA3TrainingEngine",
                operation="start_training"
            )
            
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"Starting IA³ training for job: {job_id}",
                context=context,
                extra_data={"training_config": self.config, "model_id": model_id}
            )
            
            # Initialize wandb
            import wandb
            if self.config.get("use_wandb", True):
                wandb.init(
                    project=self.config.get("wandb_project", "ia3-finetuning"),
                    name=f"training_{job_id}",
                    config=self.config,
                    tags=["ia3", model_id.split("/")[-1]]
                )
            
            # Load model dan tokenizer
            model, tokenizer = await self.load_model_and_tokenizer(model_id)
            
            # Create trainer
            trainer = self.create_trainer(model, tokenizer, train_dataset, job_id)
            
            # Start training
            trainer.train()
            
            # Save final model
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
                message=f"IA³ training completed. Model saved to: {final_model_path}",
                context=context
            )
            
            if self.config.get("use_wandb", True):
                wandb.finish()
            
            return {
                "status": "completed",
                "job_id": job_id,
                "model_path": final_model_path,
                "method": "IA3",
                "message": "IA³ training completed successfully"
            }
            
        except Exception as e:
            self.structured_logger.log(
                level="ERROR",
                category=LogCategory.TRAINING,
                message=f"IA³ training failed for job {job_id}: {str(e)}",
                context=LogContext(job_id=job_id, component="IA3TrainingEngine", operation="start_training"),
                exc_info=True
            )
            
            import wandb
            if self.config.get("use_wandb", True):
                wandb.finish(exit_code=1)
            
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(e),
                "method": "IA3",
                "message": "IA³ training failed"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get informasi model IA³."""
        if self.model is None:
            return {"status": "no_model_loaded", "method": "IA3"}
        
        info = {
            "status": "model_loaded",
            "method": "IA3",
            "device": str(self.device),
            "model_type": type(self.model).__name__,
            "config": self.config,
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
        """Get static description untuk method ini."""
        return {
            "name": "IA³",
            "full_name": "Infused Adapter by Inhibiting and Amplifying Inner Activations",
            "paper": "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning",
            "key_benefits": [
                "Fewer parameters than LoRA",
                "Faster inference (element-wise multiplication)",
                "Better performance on many tasks",
                "No low-rank decomposition needed"
            ],
            "use_cases": [
                "Key-value adaptation",
                "Feedforward adaptation",
                "Fast inference scenarios",
                "Memory-constrained environments"
            ],
            "recommended_target_modules": ["k_proj", "v_proj", "down_proj"],
            "peft_version": ">=0.6.0"
        }
