"""
Training Engine Factory untuk membuat dan mengelola berbagai metode fine-tuning.
Centralized factory pattern untuk instantiate training engines.
"""

from typing import Dict, Type, Any, Optional
import logging

# Import base class
from .base_engine import BaseTrainingEngine

# Import existing and new engines
# Note: QLoRATrainingEngine adalah default yang sudah ada
# Engine lain akan di-import setelah dibuat

logger = logging.getLogger(__name__)


class TrainingEngineFactory:
    """
    Factory class untuk membuat training engine berdasarkan metode yang dipilih.
    
    Usage:
        engine = TrainingEngineFactory.create_engine('dora', config)
        methods = TrainingEngineFactory.get_available_methods()
    """
    
    # Registry untuk mapping method name ke engine class
    _engines: Dict[str, Type[BaseTrainingEngine]] = {}
    _initialized = False
    
    @classmethod
    def _initialize_registry(cls):
        """Lazy initialization untuk menghindari circular imports."""
        if cls._initialized:
            return
        
        try:
            # Import semua engines
            from .training_engine import QLoRATrainingEngine
            from .dora_engine import DoRATrainingEngine
            from .lora_plus_engine import LoRAPlusTrainingEngine
            from .ia3_engine import IA3TrainingEngine
            from .vera_engine import VeRATrainingEngine
            from .adalora_engine import AdaLoRATrainingEngine
            from .oft_engine import OFTTrainingEngine
            
            # Register engines
            cls._engines = {
                'qlora': QLoRATrainingEngine,
                'dora': DoRATrainingEngine,
                'lora_plus': LoRAPlusTrainingEngine,
                'ia3': IA3TrainingEngine,
                'vera': VeRATrainingEngine,
                'adalora': AdaLoRATrainingEngine,
                'oft': OFTTrainingEngine,
            }
            
            cls._initialized = True
            logger.info(f"TrainingEngineFactory initialized with {len(cls._engines)} methods")
            
        except ImportError as e:
            logger.warning(f"Some engines not available: {e}")
            # Register minimum engines yang pasti tersedia
            from .training_engine import QLoRATrainingEngine
            cls._engines = {
                'qlora': QLoRATrainingEngine,
            }
            cls._initialized = True
    
    @classmethod
    def create_engine(cls, method: str, config: Dict[str, Any]) -> BaseTrainingEngine:
        """
        Create training engine instance berdasarkan method.
        
        Args:
            method: Nama metode (qlora, dora, ia3, dll)
            config: Konfigurasi training
            
        Returns:
            Instance dari BaseTrainingEngine
            
        Raises:
            ValueError: Jika method tidak dikenal
        """
        cls._initialize_registry()
        
        method = method.lower().strip()
        
        if method not in cls._engines:
            available = list(cls._engines.keys())
            raise ValueError(
                f"Unknown training method: '{method}'. "
                f"Available methods: {available}"
            )
        
        engine_class = cls._engines[method]
        engine = engine_class(config)
        
        logger.info(f"Created {engine_class.__name__} for method '{method}'")
        return engine
    
    @classmethod
    def get_available_methods(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata tentang semua metode yang tersedia.
        
        Returns:
            Dictionary dengan metadata untuk setiap method
        """
        cls._initialize_registry()
        
        methods_info = {
            'qlora': {
                'name': 'QLoRA',
                'description': 'Quantized Low-Rank Adaptation - Metode standard dengan 4-bit quantization',
                'efficiency': 'high',
                'performance': 'excellent',
                'difficulty': 'easy',
                'badges': ['Proven', 'Stable', 'Recommended for Beginners'],
                'available': 'qlora' in cls._engines,
                'supports_quantization': True,
                'paper': 'QLoRA: Efficient Finetuning of Quantized LLMs (2023)',
                'recommended_for': ['General fine-tuning', 'Memory-constrained environments', 'First-time users'],
                'parameter_reduction': '99.9%',
                'memory_reduction': '75%',
            },
            'dora': {
                'name': 'DoRA',
                'description': 'Weight-Decomposed Low-Rank Adaptation - SOTA performance dengan stabilitas lebih baik',
                'efficiency': 'high',
                'performance': 'state_of_the_art',
                'difficulty': 'easy',
                'badges': ['SOTA', 'Recommended', 'Easy Setup'],
                'available': 'dora' in cls._engines,
                'supports_quantization': True,
                'paper': 'DoRA: Weight-Decomposed Low-Rank Adaptation (ICML 2024)',
                'recommended_for': ['Best performance', 'Production deployment', 'Stability-critical applications'],
                'parameter_reduction': '99.9%',
                'memory_reduction': '75%',
            },
            'ia3': {
                'name': 'IA³',
                'description': 'Infused Adapter - Element-wise scaling dengan parameter lebih sedikit dari LoRA',
                'efficiency': 'very_high',
                'performance': 'excellent',
                'difficulty': 'easy',
                'badges': ['Fast Inference', 'Low Memory', 'Simple'],
                'available': 'ia3' in cls._engines,
                'supports_quantization': True,
                'paper': 'Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning',
                'recommended_for': ['Fast inference', 'Element-wise adaptation', 'Key-value adaptation'],
                'parameter_reduction': '99.95%',
                'memory_reduction': '80%',
            },
            'vera': {
                'name': 'VeRA',
                'description': 'Vector-based Random Matrix Adaptation - Ultra-efficient dengan 1/1000 parameter LoRA',
                'efficiency': 'extreme',
                'performance': 'good',
                'difficulty': 'easy',
                'badges': ['Ultra Low-Param', 'Edge Ready', 'Extreme Efficiency'],
                'available': 'vera' in cls._engines,
                'supports_quantization': True,
                'paper': 'VeRA: Vector-based Random Matrix Adaptation (2024)',
                'recommended_for': ['Edge deployment', 'Extreme resource constraints', 'Multiple adapters'],
                'parameter_reduction': '99.999%',
                'memory_reduction': '95%',
            },
            'lora_plus': {
                'name': 'LoRA+',
                'description': 'LoRA dengan Layer-wise Learning Rates - 2x convergence speed',
                'efficiency': 'high',
                'performance': 'excellent',
                'difficulty': 'easy',
                'badges': ['Fast Convergence', 'Same Quality', 'Easy Upgrade'],
                'available': 'lora_plus' in cls._engines,
                'supports_quantization': True,
                'paper': 'LoRA+: Efficient Low Rank Adaptation of Large Models (2024)',
                'recommended_for': ['Fast training', 'Quick iteration', 'Large-scale experiments'],
                'parameter_reduction': '99.9%',
                'memory_reduction': '75%',
                'speedup': '2x convergence',
            },
            'adalora': {
                'name': 'AdaLoRA',
                'description': 'Adaptive Budget Allocation - Parameter dialokasikan secara dinamis selama training',
                'efficiency': 'high',
                'performance': 'state_of_the_art',
                'difficulty': 'medium',
                'badges': ['Adaptive', 'Advanced', 'Budget-aware'],
                'available': 'adalora' in cls._engines,
                'supports_quantization': True,
                'paper': 'AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning (ICLR 2023)',
                'recommended_for': ['Complex tasks', 'Budget optimization', 'Dynamic allocation'],
                'parameter_reduction': '99.9%',
                'memory_reduction': '75%',
            },
            'oft': {
                'name': 'OFT',
                'description': 'Orthogonal Fine-Tuning - Multiplicative updates dengan orthogonal constraint',
                'efficiency': 'high',
                'performance': 'excellent',
                'difficulty': 'medium',
                'badges': ['Orthogonal', 'Geometric', 'Stable'],
                'available': 'oft' in cls._engines,
                'supports_quantization': True,
                'paper': 'Controlling Text-to-Image Diffusion by Orthogonal Finetuning (CVPR 2024)',
                'recommended_for': ['Multimodal tasks', 'Vision-language models', 'Geometric stability'],
                'parameter_reduction': '99.9%',
                'memory_reduction': '75%',
            },
        }
        
        # Filter hanya yang available
        return {k: v for k, v in methods_info.items() if v['available']}
    
    @classmethod
    def is_method_available(cls, method: str) -> bool:
        """Check apakah method tersedia."""
        cls._initialize_registry()
        return method.lower().strip() in cls._engines
    
    @classmethod
    def get_method_info(cls, method: str) -> Optional[Dict[str, Any]]:
        """Get info untuk specific method."""
        methods = cls.get_available_methods()
        return methods.get(method.lower().strip())
    
    @classmethod
    def get_default_method(cls) -> str:
        """Get default recommended method."""
        # Prioritas: DoRA > QLoRA > IA3
        if cls.is_method_available('dora'):
            return 'dora'
        elif cls.is_method_available('qlora'):
            return 'qlora'
        else:
            return list(cls._engines.keys())[0] if cls._engines else 'qlora'
    
    @classmethod
    def register_engine(cls, method: str, engine_class: Type[BaseTrainingEngine]):
        """
        Register custom engine (untuk extensibility).
        
        Args:
            method: Nama method (lowercase)
            engine_class: Class yang inherit dari BaseTrainingEngine
        """
        if not issubclass(engine_class, BaseTrainingEngine):
            raise ValueError(f"Engine class must inherit from BaseTrainingEngine")
        
        cls._engines[method.lower()] = engine_class
        logger.info(f"Registered custom engine: {method}")


# Helper functions untuk convenience
def create_training_engine(method: str, config: Dict[str, Any]) -> BaseTrainingEngine:
    """Convenience function untuk create engine."""
    return TrainingEngineFactory.create_engine(method, config)


def get_available_training_methods() -> Dict[str, Dict[str, Any]]:
    """Convenience function untuk get methods."""
    return TrainingEngineFactory.get_available_methods()
