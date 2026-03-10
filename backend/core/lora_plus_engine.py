"""
LoRA+ (LoRA with Layer-wise Learning Rates) Training Engine.

LoRA+ menggunakan learning rates berbeda untuk matriks A dan B:
- Matriks A: learning_rate * ratio (biasanya 16x)
- Matriks B: learning_rate (standar)

Hasilnya adalah convergence 2x lebih cepat dengan kualitas yang sama.

Paper: "LoRA+: Efficient Low Rank Adaptation of Large Models" (2024)
"""

from typing import Dict, Any, Tuple, Optional, List
import torch
from torch.optim import AdamW
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, TaskType
from datasets import Dataset

from .training_engine import QLoRATrainingEngine
from .enhanced_logging_system import LogContext, LogCategory


class LoRAPlusTrainingEngine(QLoRATrainingEngine):
    """
    Training engine untuk LoRA+ dengan layer-wise learning rates.
    
    Key Insight:
    Dalam LoRA, W = W0 + BA, dimana B (output dim) dan A (low-rank).
    LoRA+ menggunakan LR yang lebih tinggi untuk A untuk mempercepat convergence.
    
    Recommended Settings:
    - ratio = 16 (default) atau 32
    - lr_A = learning_rate * ratio
    - lr_B = learning_rate
    
    Benefits:
    - 2x faster convergence (setara 2x speedup dalam epochs)
    - Same final performance as standard LoRA
    - No additional memory overhead
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Update component name untuk logging
        self.structured_logger.component_name = "LoRAPlusTrainingEngine"
        
        # LoRA+ specific config
        self.lora_plus_ratio = config.get("lora_plus_ratio", 16)
        self.validate_ratio()
    
    def validate_ratio(self):
        """Validasi learning rate ratio."""
        if self.lora_plus_ratio < 1:
            self.structured_logger.log(
                level="WARNING",
                category=LogCategory.TRAINING,
                message=f"LoRA+ ratio {self.lora_plus_ratio} < 1, setting to 1 (standard LoRA)",
                context=LogContext(component="LoRAPlusTrainingEngine", operation="validate_ratio")
            )
            self.lora_plus_ratio = 1
        elif self.lora_plus_ratio > 256:
            self.structured_logger.log(
                level="WARNING",
                category=LogCategory.TRAINING,
                message=f"LoRA+ ratio {self.lora_plus_ratio} > 256 may cause instability, capping at 256",
                context=LogContext(component="LoRAPlusTrainingEngine", operation="validate_ratio")
            )
            self.lora_plus_ratio = 256
    
    def create_custom_optimizer(self, model) -> AdamW:
        """
        Create optimizer dengan parameter groups berbeda untuk A dan B.
        
        Ini adalah inti dari LoRA+ - menggunakan learning rates yang berbeda
        untuk matriks A dan B.
        """
        base_lr = self.config.get("learning_rate", 2e-4)
        lr_A = base_lr * self.lora_plus_ratio
        lr_B = base_lr
        
        # Group parameters berdasarkan nama
        lora_a_params = []
        lora_b_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Identify parameter type dari nama
            if "lora_A" in name or "lora_a" in name.lower():
                lora_a_params.append(param)
            elif "lora_B" in name or "lora_b" in name.lower():
                lora_b_params.append(param)
            else:
                other_params.append(param)
        
        # Buat parameter groups
        optimizer_groups = []
        
        if lora_a_params:
            optimizer_groups.append({
                "params": lora_a_params,
                "lr": lr_A,
                "name": "lora_A",
                "weight_decay": 0.01,
            })
        
        if lora_b_params:
            optimizer_groups.append({
                "params": lora_b_params,
                "lr": lr_B,
                "name": "lora_B",
                "weight_decay": 0.01,
            })
        
        if other_params:
            optimizer_groups.append({
                "params": other_params,
                "lr": base_lr,
                "name": "other",
                "weight_decay": 0.01,
            })
        
        # Log parameter distribution
        context = LogContext(
            component="LoRAPlusTrainingEngine",
            operation="create_custom_optimizer"
        )
        
        self.structured_logger.log(
            level="INFO",
            category=LogCategory.TRAINING,
            message=f"LoRA+ optimizer: A params={len(lora_a_params)} (lr={lr_A:.2e}), B params={len(lora_b_params)} (lr={lr_B:.2e}), ratio={self.lora_plus_ratio}x",
            context=context,
            extra_data={
                "lora_plus_ratio": self.lora_plus_ratio,
                "lr_A": lr_A,
                "lr_B": lr_B,
                "base_lr": base_lr,
                "num_a_params": len(lora_a_params),
                "num_b_params": len(lora_b_params),
                "num_other_params": len(other_params),
            }
        )
        
        # Create optimizer
        optimizer = AdamW(
            optimizer_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        return optimizer
    
    def setup_training_arguments(self, job_id: str) -> TrainingArguments:
        """
        Setup training arguments dengan custom optimizer untuk LoRA+.
        
        Note: Kita tidak bisa langsung pass optimizer ke TrainingArguments,
        jadi kita perlu override create_trainer untuk menggunakan custom optimizer.
        """
        # Get base arguments dari parent
        training_args = super().setup_training_arguments(job_id)
        
        # Note: LoRA+ menggunakan custom optimizer yang akan di-apply di create_trainer
        # TrainingArguments tidak support custom optimizer directly,
        # jadi kita perlu create_trainer override
        
        return training_args
    
    def create_trainer(self, model, tokenizer, train_dataset: Dataset, job_id: str) -> Trainer:
        """
        Create trainer dengan custom optimizer untuk LoRA+.
        """
        training_args = self.setup_training_arguments(job_id)
        
        # Import data collator
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Import callback
        from .training_callback import QLoRATrainingCallback
        
        # Create custom optimizer
        optimizer = self.create_custom_optimizer(model)
        
        # Create trainer dengan custom optimizer
        # Note: Kita perlu meng-override trainer untuk menggunakan optimizer custom
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[QLoRATrainingCallback(job_id, self.config.get("db"))],
            optimizers=(optimizer, None),  # (optimizer, lr_scheduler) - scheduler akan dibuat oleh Trainer
        )
        
        return trainer
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get informasi model dengan LoRA+ specific details."""
        info = super().get_model_info()
        
        base_lr = self.config.get("learning_rate", 2e-4)
        
        info.update({
            "method": "LoRA+",
            "lora_plus_ratio": self.lora_plus_ratio,
            "lr_A": base_lr * self.lora_plus_ratio,
            "lr_B": base_lr,
            "expected_speedup": "2x convergence",
            "notes": f"LoRA+ uses {self.lora_plus_ratio}x higher learning rate for LoRA A matrices"
        })
        
        return info
    
    async def start_training(self, job_id: str, model_id: str, train_dataset: Dataset, db=None) -> Dict[str, Any]:
        """
        Start training dengan LoRA+ logging.
        """
        context = LogContext(
            job_id=job_id,
            component="LoRAPlusTrainingEngine",
            operation="start_training"
        )
        
        base_lr = self.config.get("learning_rate", 2e-4)
        
        self.structured_logger.log(
            level="INFO",
            category=LogCategory.TRAINING,
            message=f"Starting LoRA+ training with ratio {self.lora_plus_ratio}x (lr_A={base_lr * self.lora_plus_ratio:.2e}, lr_B={base_lr:.2e})",
            context=context,
            extra_data={
                "method": "LoRA+",
                "lora_plus_ratio": self.lora_plus_ratio,
                "learning_rate_A": base_lr * self.lora_plus_ratio,
                "learning_rate_B": base_lr,
            }
        )
        
        # Call parent implementation
        return await super().start_training(job_id, model_id, train_dataset, db)
    
    @staticmethod
    def get_method_description() -> Dict[str, Any]:
        """Get static description untuk method ini."""
        return {
            "name": "LoRA+",
            "full_name": "LoRA with Layer-wise Learning Rates",
            "paper": "LoRA+: Efficient Low Rank Adaptation of Large Models (2024)",
            "key_benefits": [
                "2x faster convergence",
                "Same final performance as LoRA",
                "No additional memory overhead",
                "Easy to implement (just optimizer change)"
            ],
            "use_cases": [
                "Fast iteration",
                "Large-scale experiments",
                "Quick prototyping",
                "Limited training budget"
            ],
            "recommended_ratio": 16,
            "ratio_range": "4-32 (higher = faster but less stable)"
        }
