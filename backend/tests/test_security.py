"""
Security tests untuk QLoRA Fine-tuning Platform.
Tests untuk path traversal, input validation, dan security hardening.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.security import (
    validate_dataset_path,
    sanitize_filename,
    validate_model_id,
    validate_api_key,
    validate_training_config
)


class TestPathValidation:
    """Test suite untuk path traversal protection."""
    
    def test_valid_dataset_path(self):
        """Valid paths dalam datasets directory."""
        assert validate_dataset_path("datasets/train.json") == True
        assert validate_dataset_path("./datasets/data/train.json") == True
        assert validate_dataset_path("datasets/nested/deep/file.csv") == True
    
    def test_path_traversal_attack_unix(self):
        """Unix-style path traversal attempts."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\..\..\windows\system32",
            "datasets/../../etc/passwd",
            "./datasets/../../../etc/shadow",
            "datasets/train.json/../../../etc/passwd"
        ]
        for path in malicious_paths:
            assert validate_dataset_path(path) == False, f"Should reject: {path}"
    
    def test_path_traversal_null_bytes(self):
        """Null byte injection attempts."""
        assert validate_dataset_path("datasets/train.json\x00.txt") == False
        assert validate_dataset_path("\x00datasets/train.json") == False
    
    def test_absolute_path_outside_base(self):
        """Absolute paths diluar base directory."""
        assert validate_dataset_path("/etc/passwd") == False
        assert validate_dataset_path("/usr/bin/cat") == False
        assert validate_dataset_path("C:\\Windows\\System32") == False
    
    def test_empty_and_none_paths(self):
        """Empty dan None path handling."""
        assert validate_dataset_path("") == False
        assert validate_dataset_path(None) == False
        assert validate_dataset_path("   ") == False


class TestFilenameSanitization:
    """Test suite untuk filename sanitization."""
    
    def test_basic_sanitization(self):
        """Basic filename sanitization."""
        assert sanitize_filename("test.json") == "test.json"
        assert sanitize_filename("my-file.txt") == "my-file.txt"
    
    def test_path_separator_removal(self):
        """Remove path separators dari filename."""
        assert sanitize_filename("path/to/file.txt") == "path_to_file.txt"
        assert sanitize_filename("path\\to\\file.txt") == "path_to_file.txt"
    
    def test_hidden_file_protection(self):
        """Remove leading dots (hidden files)."""
        assert sanitize_filename(".htaccess") == "htaccess"
        assert sanitize_filename("..hidden") == ".hidden"
        assert sanitize_filename("...triple") == "..triple"
    
    def test_length_limit(self):
        """Filename length limiting."""
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 255


class TestModelIdValidation:
    """Test suite untuk model ID validation."""
    
    def test_valid_huggingface_model_ids(self):
        """Valid HuggingFace model ID formats."""
        assert validate_model_id("meta-llama/Llama-2-7b") == True
        assert validate_model_id("mistralai/Mistral-7B-v0.1") == True
        assert validate_model_id("org/model-name") == True
        assert validate_model_id("user/model_v2") == True
    
    def test_invalid_model_ids(self):
        """Invalid model ID formats."""
        assert validate_model_id("invalid") == False  # No slash
        assert validate_model_id("/model") == False  # No username
        assert validate_model_id("user/") == False  # No model name
        assert validate_model_id("user/model/extra") == False  # Too many slashes
        assert validate_model_id("") == False  # Empty


class TestAPIKeyValidation:
    """Test suite untuk API key validation."""
    
    def test_valid_api_keys(self):
        """Valid API key formats."""
        assert validate_api_key("a" * 32) == True
        assert validate_api_key("abc123" * 6) == True  # 36 chars
    
    def test_invalid_api_keys(self):
        """Invalid API key formats."""
        assert validate_api_key("short") == False  # Too short
        assert validate_api_key("a" * 31) == False  # One char short
        assert validate_api_key("") == False  # Empty
        assert validate_api_key(None) == False  # None
        assert validate_api_key("key with spaces") == False  # Spaces
        assert validate_api_key("key\twith\ttabs") == False  # Tabs
    
    def test_api_key_printable_only(self):
        """API keys must be printable characters only."""
        assert validate_api_key("a" * 31 + "\x00") == False  # Null byte
        assert validate_api_key("a" * 31 + "\x01") == False  # Control char


class TestTrainingConfigValidation:
    """Test suite untuk training config validation."""
    
    def test_valid_config(self):
        """Valid training configuration."""
        config = {
            "lora_rank": 16,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "batch_size": 2
        }
        is_valid, error = validate_training_config(config)
        assert is_valid == True
        assert error is None
    
    def test_invalid_lora_rank(self):
        """Invalid LoRA rank values."""
        test_cases = [
            ({"lora_rank": 0}, "too small"),
            ({"lora_rank": 1025}, "too large"),
            ({"lora_rank": -1}, "negative"),
            ({"lora_rank": "16"}, "string"),
        ]
        base_config = {"learning_rate": 2e-4, "num_epochs": 3, "batch_size": 2}
        
        for invalid_config, description in test_cases:
            config = {**base_config, **invalid_config}
            is_valid, error = validate_training_config(config)
            assert is_valid == False, f"Should reject {description}"
            assert "lora_rank" in error.lower()
    
    def test_invalid_learning_rate(self):
        """Invalid learning rate values."""
        test_cases = [
            ({"learning_rate": 0}, "zero"),
            ({"learning_rate": 1e-7}, "too small"),
            ({"learning_rate": 1e-1}, "too large"),
            ({"learning_rate": "2e-4"}, "string"),
        ]
        base_config = {"lora_rank": 16, "num_epochs": 3, "batch_size": 2}
        
        for invalid_config, description in test_cases:
            config = {**base_config, **invalid_config}
            is_valid, error = validate_training_config(config)
            assert is_valid == False, f"Should reject {description}"
            assert "learning_rate" in error.lower()
    
    def test_invalid_num_epochs(self):
        """Invalid num_epochs values."""
        test_cases = [
            ({"num_epochs": 0}, "zero"),
            ({"num_epochs": 101}, "too large"),
            ({"num_epochs": -1}, "negative"),
        ]
        base_config = {"lora_rank": 16, "learning_rate": 2e-4, "batch_size": 2}
        
        for invalid_config, description in test_cases:
            config = {**base_config, **invalid_config}
            is_valid, error = validate_training_config(config)
            assert is_valid == False, f"Should reject {description}"
            assert "num_epochs" in error.lower()
    
    def test_invalid_batch_size(self):
        """Invalid batch_size values."""
        test_cases = [
            ({"batch_size": 0}, "zero"),
            ({"batch_size": 129}, "too large"),
            ({"batch_size": -1}, "negative"),
        ]
        base_config = {"lora_rank": 16, "learning_rate": 2e-4, "num_epochs": 3}
        
        for invalid_config, description in test_cases:
            config = {**base_config, **invalid_config}
            is_valid, error = validate_training_config(config)
            assert is_valid == False, f"Should reject {description}"
            assert "batch_size" in error.lower()


class TestEdgeCases:
    """Edge case dan boundary tests."""
    
    def test_boundary_values_lora_rank(self):
        """Boundary values untuk LoRA rank."""
        config = {"lora_rank": 1, "learning_rate": 2e-4, "num_epochs": 3, "batch_size": 2}
        assert validate_training_config(config)[0] == True  # Minimum
        
        config["lora_rank"] = 1024
        assert validate_training_config(config)[0] == True  # Maximum
    
    def test_boundary_values_learning_rate(self):
        """Boundary values untuk learning rate."""
        config = {"lora_rank": 16, "num_epochs": 3, "batch_size": 2}
        
        config["learning_rate"] = 1e-6  # Minimum
        assert validate_training_config(config)[0] == True
        
        config["learning_rate"] = 1e-2  # Maximum
        assert validate_training_config(config)[0] == True
    
    def test_missing_fields(self):
        """Missing optional fields should still validate."""
        config = {
            "lora_rank": 16,
            "learning_rate": 2e-4,
            # Missing num_epochs and batch_size - use defaults
        }
        is_valid, error = validate_training_config(config)
        assert is_valid == True, f"Should validate with defaults: {error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
