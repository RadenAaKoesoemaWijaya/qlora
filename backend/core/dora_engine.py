"""
DoRA (Weight-Decomposed Low-Rank Adaptation) Training Engine.

DoRA adalah metode SOTA yang memisahkan update bobot menjadi magnitude dan direction,
menghasilkan stabilitas training lebih baik dan performa mengungguli LoRA/QLoRA.

Paper: "DoRA: Weight-Decomposed Low-Rank Adaptation" (ICML 2024)
"""

from typing import Dict, Any, Tuple, Optional
import torch
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import existing QLoRA engine dan extend
from .training_engine import QLoRATrainingEngine
from .enhanced_logging_system import LogContext, LogCategory


class DoRATrainingEngine(QLoRATrainingEngine):
    """
    Training engine untuk DoRA (Weight-Decomposed Low-Rank Adaptation).
    
    DoRA memisahkan pretrained weight W0 menjadi magnitude vector (m) dan 
    directional matrix (V), kemudian menerapkan low-rank adaptation hanya pada V.
    
    Keuntungan:
    - Stabilitas training lebih baik
    - Convergence lebih cepat
    - Performa mengungguli LoRA standard
    - Kompatibel dengan quantization (4-bit)
    
    PEFT Config Changes:
    - use_dora=True di LoraConfig
    - use_rslora=False (tidak compatible dengan DoRA)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Update component name untuk logging
        self.structured_logger.component_name = "DoRATrainingEngine"
    
    def setup_lora_config(self) -> LoraConfig:
        """
        Setup LoRA configuration dengan DoRA enabled.
        
        Override dari parent class untuk enable use_dora=True.
        """
        # Validasi: DoRA tidak compatible dengan rslora
        use_rslora = self.config.get("use_rslora", False)
        if use_rslora:
            self.structured_logger.log(
                level="WARNING",
                category=LogCategory.TRAINING,
                message="rsLoRA tidak compatible dengan DoRA, menonaktifkan rsLoRA",
                context=LogContext(component="DoRATrainingEngine", operation="setup_lora_config")
            )
            use_rslora = False
        
        # DoRA simplified mode (opsional untuk speedup)
        use_dora_simple = self.config.get("dora_simple", False)
        
        lora_config = LoraConfig(
            r=self.config.get("lora_rank", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            target_modules=self.config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            # DoRA specific
            use_dora=True,
            use_rslora=use_rslora,
            # Optional: simplified DoRA untuk speedup
            # (beberapa implementasi PEFT support ini)
        )
        
        # Log DoRA activation
        context = LogContext(
            component="DoRATrainingEngine",
            operation="setup_lora_config"
        )
        
        self.structured_logger.log(
            level="INFO",
            category=LogCategory.TRAINING,
            message=f"DoRA configuration created: rank={lora_config.r}, alpha={lora_config.lora_alpha}, use_dora={lora_config.use_dora}",
            context=context,
            extra_data={
                "lora_rank": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "use_dora": lora_config.use_dora,
                "target_modules": lora_config.target_modules,
                "dora_simple_mode": use_dora_simple,
            }
        )
        
        return lora_config
    
    async def load_model_and_tokenizer(self, model_id: str) -> Tuple[Any, Any]:
        """
        Load model dengan DoRA configuration.
        
        Hampir identik dengan QLoRA, tapi dengan logging yang mengindikasikan DoRA.
        """
        context = LogContext(
            job_id=self.training_job_id,
            component="DoRATrainingEngine",
            operation="load_model_and_tokenizer"
        )
        
        self.structured_logger.log(
            level="INFO",
            category=LogCategory.TRAINING,
            message=f"Loading model dengan DoRA: {model_id}",
            context=context
        )
        
        # Gunakan implementasi parent
        model, tokenizer = await super().load_model_and_tokenizer(model_id)
        
        # Additional DoRA-specific logging
        if self.model and hasattr(self.model, 'get_nb_trainable_parameters'):
            trainable, all_params = self.model.get_nb_trainable_parameters()
            
            self.structured_logger.log(
                level="INFO",
                category=LogCategory.TRAINING,
                message=f"DoRA model loaded: {trainable:,} trainable params from {all_params:,} total ({100 * trainable / all_params:.4f}%)",
                context=context,
                extra_data={
                    "trainable_parameters": trainable,
                    "total_parameters": all_params,
                    "trainable_percentage": 100 * trainable / all_params,
                    "method": "DoRA"
                }
            )
        
        return model, tokenizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get informasi model dengan DoRA-specific details."""
        info = super().get_model_info()
        
        # Add DoRA specific info
        info.update({
            "method": "DoRA",
            "use_dora": True,
            "dora_simple_mode": self.config.get("dora_simple", False),
            "notes": "DoRA separates weight updates into magnitude and direction components"
        })
        
        return info
    
    @staticmethod
    def get_method_description() -> Dict[str, Any]:
        """Get static description untuk method ini."""
        return {
            "name": "DoRA",
            "full_name": "Weight-Decomposed Low-Rank Adaptation",
            "paper": "DoRA: Weight-Decomposed Low-Rank Adaptation (ICML 2024)",
            "key_benefits": [
                "State-of-the-art performance",
                "Better training stability than LoRA",
                "Faster convergence",
                "Compatible with 4-bit quantization"
            ],
            "use_cases": [
                "Production fine-tuning",
                "Stability-critical applications",
                "When maximum performance is needed"
            ],
            "config_requirements": {
                "use_dora": True,
                "peft_version": ">=0.8.0"
            }
        }
