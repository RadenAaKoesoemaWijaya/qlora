"""
Security utilities untuk validasi input dan proteksi terhadap common attacks.
"""

from pathlib import Path
from typing import Optional
import re


def validate_dataset_path(file_path: str, base_dir: str = "./datasets") -> bool:
    """
    Validasi dataset path untuk mencegah path traversal attack.
    
    Args:
        file_path: Path yang akan divalidasi
        base_dir: Direktori base yang diizinkan
        
    Returns:
        True jika path valid dan dalam allowed directory
        
    Example:
        >>> validate_dataset_path("datasets/train.json")
        True
        >>> validate_dataset_path("../../../etc/passwd")
        False
    """
    try:
        # Resolve paths to absolute
        full_path = Path(file_path).resolve()
        base_path = Path(base_dir).resolve()
        
        # Check if path is within allowed directory
        return str(full_path).startswith(str(base_path))
    except Exception:
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename untuk mencegah malicious characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename yang aman
    """
    # Remove path separators and null bytes
    filename = filename.replace('/', '_').replace('\\', '_').replace('\x00', '')
    
    # Remove leading dots (hidden files)
    filename = filename.lstrip('.')
    
    # Limit length
    if len(filename) > 255:
        name, ext = Path(filename).stem, Path(filename).suffix
        filename = name[:255 - len(ext)] + ext
    
    return filename


def validate_model_id(model_id: str) -> bool:
    """
    Validasi model ID format (HuggingFace format).
    
    Args:
        model_id: Model identifier
        
    Returns:
        True jika format valid
    """
    # HuggingFace format: username/model-name or org/model-name
    pattern = r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, model_id))


def validate_api_key(key: str) -> bool:
    """
    Validasi API key format.
    
    Args:
        key: API key string
        
    Returns:
        True jika format valid (minimal 32 chars, alphanumeric)
    """
    if not key or len(key) < 32:
        return False
    
    # Check for printable characters only
    return all(c.isprintable() and not c.isspace() for c in key)


def validate_training_config(config: dict) -> tuple[bool, Optional[str]]:
    """
    Validasi training configuration parameters.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Tuple (is_valid, error_message)
    """
    # Validate lora_rank
    rank = config.get('lora_rank', 16)
    if not isinstance(rank, int) or not 1 <= rank <= 1024:
        return False, f"lora_rank must be integer 1-1024, got {rank}"
    
    # Validate learning_rate
    lr = config.get('learning_rate', 2e-4)
    if not isinstance(lr, (int, float)) or not 1e-6 <= lr <= 1e-2:
        return False, f"learning_rate must be 1e-6 to 1e-2, got {lr}"
    
    # Validate num_epochs
    epochs = config.get('num_epochs', 3)
    if not isinstance(epochs, int) or not 1 <= epochs <= 100:
        return False, f"num_epochs must be integer 1-100, got {epochs}"
    
    # Validate batch_size
    batch = config.get('batch_size', 2)
    if not isinstance(batch, int) or not 1 <= batch <= 128:
        return False, f"batch_size must be integer 1-128, got {batch}"
    
    return True, None
