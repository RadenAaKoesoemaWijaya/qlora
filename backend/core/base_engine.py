"""
Base Training Engine Interface untuk semua metode fine-tuning.
Mendefinisikan kontrak yang harus diimplementasikan oleh setiap training engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
from datasets import Dataset


class BaseTrainingEngine(ABC):
    """
    Abstract base class untuk semua training engines.
    
    Semua metode fine-tuning (QLoRA, DoRA, IA3, dll) harus mengimplementasikan
    interface ini untuk memastikan kompatibilitas dengan platform.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_job_id = None
        self.structured_logger = None
    
    @abstractmethod
    async def load_model_and_tokenizer(self, model_id: str) -> Tuple[Any, Any]:
        """
        Load model dan tokenizer dengan konfigurasi method-specific.
        
        Args:
            model_id: Hugging Face model ID atau path lokal
            
        Returns:
            Tuple (model, tokenizer)
        """
        pass
    
    @abstractmethod
    def setup_peft_config(self) -> Any:
        """
        Setup PEFT configuration (LoRA, IA3, OFT, dll).
        
        Returns:
            PEFT config object (LoraConfig, IA3Config, OFTConfig, dll)
        """
        pass
    
    @abstractmethod
    def setup_training_arguments(self, job_id: str) -> Any:
        """
        Setup training arguments.
        
        Args:
            job_id: Unique training job ID
            
        Returns:
            TrainingArguments object
        """
        pass
    
    @abstractmethod
    def create_trainer(self, model: Any, tokenizer: Any, train_dataset: Dataset, job_id: str) -> Any:
        """
        Create trainer instance.
        
        Args:
            model: Model yang sudah di-load
            tokenizer: Tokenizer instance
            train_dataset: Training dataset
            job_id: Unique job ID
            
        Returns:
            Trainer instance
        """
        pass
    
    @abstractmethod
    async def start_training(self, 
                           job_id: str,
                           model_id: str,
                           train_dataset: Dataset,
                           db=None) -> Dict[str, Any]:
        """
        Start training process.
        
        Args:
            job_id: Unique training job ID
            model_id: Hugging Face model ID
            train_dataset: Processed training dataset
            db: Database instance untuk update progress (optional)
            
        Returns:
            Dictionary dengan training status dan informasi
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get informasi model yang sedang digunakan.
        
        Returns:
            Dictionary dengan informasi model
        """
        pass
    
    @abstractmethod
    def get_trainable_parameters(self) -> Optional[Tuple[int, int]]:
        """
        Get jumlah trainable parameters.
        
        Returns:
            Tuple (trainable_params, all_params) atau None jika model belum load
        """
        pass
    
    def setup_quantization_config(self):
        """
        Setup 4-bit quantization configuration (default implementation).
        Can be overridden by subclasses if needed.
        """
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8
        )
    
    async def cleanup(self):
        """
        Cleanup resources setelah training selesai.
        Mencakup: GPU memory cleanup, model deletion, garbage collection.
        """
        import gc
        
        # Delete model dan trainer references
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.trainer is not None:
            del self.trainer
            self.trainer = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache jika available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def __del__(self):
        """Destructor untuk cleanup saat object dihapus."""
        try:
            import asyncio
            if self.model is not None or self.trainer is not None:
                # Attempt async cleanup jika event loop running
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.cleanup())
                except RuntimeError:
                    # No event loop, do sync cleanup
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        except Exception:
            pass  # Ignore cleanup errors dalam destructor
