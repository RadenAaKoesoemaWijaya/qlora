#!/usr/bin/env python3
"""
Environment Configuration Validation untuk QLoRA Application
Validasi konfigurasi environment untuk deployment production.
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class ConfigType(Enum):
    """Configuration types."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    DEPENDENT = "dependent"


@dataclass
class ConfigValidation:
    """Configuration validation rule."""
    key: str
    config_type: ConfigType
    validation_func: callable
    description: str
    dependent_on: Optional[str] = None
    default_value: Any = None


class EnvironmentValidator:
    """Environment configuration validator."""
    
    def __init__(self):
        self.validations = self._setup_validations()
        self.errors = []
        self.warnings = []
    
    def _setup_validations(self) -> List[ConfigValidation]:
        """Setup configuration validation rules."""
        return [
            # Database Configuration
            ConfigValidation(
                key="DATABASE_URL",
                config_type=ConfigType.REQUIRED,
                validation_func=self._validate_database_url,
                description="MongoDB connection string"
            ),
            ConfigValidation(
                key="MONGO_USERNAME",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_string,
                description="MongoDB username",
                default_value="admin"
            ),
            ConfigValidation(
                key="MONGO_PASSWORD",
                config_type=ConfigType.REQUIRED,
                validation_func=self._validate_string,
                description="MongoDB password"
            ),
            ConfigValidation(
                key="MONGO_DATABASE",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_string,
                description="MongoDB database name",
                default_value="qlora_db"
            ),
            
            # Redis Configuration
            ConfigValidation(
                key="REDIS_URL",
                config_type=ConfigType.REQUIRED,
                validation_func=self._validate_redis_url,
                description="Redis connection string"
            ),
            ConfigValidation(
                key="REDIS_PASSWORD",
                config_type=ConfigType.REQUIRED,
                validation_func=self._validate_string,
                description="Redis password"
            ),
            
            # Security Configuration
            ConfigValidation(
                key="SECRET_KEY",
                config_type=ConfigType.REQUIRED,
                validation_func=self._validate_secret_key,
                description="Application secret key"
            ),
            ConfigValidation(
                key="JWT_SECRET_KEY",
                config_type=ConfigType.REQUIRED,
                validation_func=self._validate_secret_key,
                description="JWT secret key"
            ),
            ConfigValidation(
                key="JWT_ALGORITHM",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_jwt_algorithm,
                description="JWT algorithm",
                default_value="HS256"
            ),
            ConfigValidation(
                key="ACCESS_TOKEN_EXPIRE_MINUTES",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_positive_integer,
                description="Access token expiration time (minutes)",
                default_value=30
            ),
            
            # Environment Configuration
            ConfigValidation(
                key="ENVIRONMENT",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_environment,
                description="Application environment",
                default_value="development"
            ),
            ConfigValidation(
                key="LOG_LEVEL",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_log_level,
                description="Logging level",
                default_value="INFO"
            ),
            ConfigValidation(
                key="ENABLE_FILE_LOGGING",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_boolean,
                description="Enable file logging",
                default_value="true"
            ),
            ConfigValidation(
                key="LOG_FILE_PATH",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_file_path,
                description="Log file path",
                default_value="logs/app.log"
            ),
            
            # GPU Configuration
            ConfigValidation(
                key="GPU_MEMORY_THRESHOLD",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_percentage,
                description="GPU memory usage threshold",
                default_value=0.9
            ),
            ConfigValidation(
                key="ENABLE_GPU_MONITORING",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_boolean,
                description="Enable GPU monitoring",
                default_value="true"
            ),
            
            # Performance Configuration
            ConfigValidation(
                key="MAX_WORKERS",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_positive_integer,
                description="Maximum worker processes",
                default_value=4
            ),
            ConfigValidation(
                key="ENABLE_PERFORMANCE_MONITORING",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_boolean,
                description="Enable performance monitoring",
                default_value="true"
            ),
            
            # Security Configuration
            ConfigValidation(
                key="ENABLE_SECURITY_AUDIT",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_boolean,
                description="Enable security audit logging",
                default_value="true"
            ),
            
            # ML Platform Configuration
            ConfigValidation(
                key="WANDB_API_KEY",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_api_key,
                description="Weights & Biases API key"
            ),
            ConfigValidation(
                key="HUGGINGFACE_TOKEN",
                config_type=ConfigType.OPTIONAL,
                validation_func=self._validate_api_key,
                description="Hugging Face API token"
            ),
        ]
    
    def validate(self, env_vars: Dict[str, str]) -> Dict[str, Any]:
        """Validate environment configuration."""
        self.errors = []
        self.warnings = []
        
        validated_config = {}
        
        for validation in self.validations:
            value = env_vars.get(validation.key, validation.default_value)
            
            # Check dependent configuration
            if validation.config_type == ConfigType.DEPENDENT and validation.dependent_on:
                if validation.dependent_on not in env_vars:
                    continue
            
            # Validate the value
            is_valid, message = validation.validation_func(value, validation)
            
            if not is_valid:
                if validation.config_type == ConfigType.REQUIRED:
                    self.errors.append(f"{validation.key}: {message}")
                else:
                    self.warnings.append(f"{validation.key}: {message}")
            else:
                validated_config[validation.key] = value
        
        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "config": validated_config
        }
    
    def _validate_string(self, value: Any, validation: ConfigValidation) -> tuple:
        """Validate string configuration."""
        if not isinstance(value, str) or not value.strip():
            return False, "Must be a non-empty string"
        return True, "Valid"
    
    def _validate_positive_integer(self, value: Any, validation: ConfigValidation) -> tuple:
        """Validate positive integer configuration."""
        try:
            int_value = int(value)
            if int_value <= 0:
                return False, "Must be a positive integer"
            return True, "Valid"
        except (ValueError, TypeError):
            return False, "Must be a valid integer"
    
    def _validate_percentage(self, value: Any, validation: ConfigValidation) -> tuple:
        """Validate percentage configuration."""
        try:
            float_value = float(value)
            if not 0 <= float_value <= 1:
                return False, "Must be between 0 and 1"
            return True, "Valid"
        except (ValueError, TypeError):
            return False, "Must be a valid number"
    
    def _validate_boolean(self, value: Any, validation: ConfigValidation) -> tuple:
        """Validate boolean configuration."""
        if isinstance(value, bool):
            return True, "Valid"
        
        if isinstance(value, str):
            if value.lower() in ("true", "false", "1", "0", "yes", "no"):
                return True, "Valid"
        
        return False, "Must be a boolean value (true/false, 1/0, yes/no)"
    
    def _validate_file_path(self, value: Any, validation: ConfigValidation) -> tuple:
        """Validate file path configuration."""
        if not isinstance(value, str):
            return False, "Must be a string"
        
        try:
            path = Path(value)
            if path.is_absolute():
                return False, "Must be a relative path"
            return True, "Valid"
        except Exception:
            return False, "Must be a valid file path"
    
    def _validate_database_url(self, value: Any, validation: ConfigValidation) -> tuple:
        """Validate database URL configuration."""
        if not isinstance(value, str):
            return False, "Must be a string"
        
        # Basic MongoDB URL validation
        pattern = r'^mongodb(\+srv)?://[^:]+:[^@]+@[^/]+/[^?]+(\?.*)?$'
        if not re.match(pattern, value):
            return False, "Must be a valid MongoDB connection string"
        
        return True, "Valid"
    
    def _validate_redis_url(self, value: Any, validation: ConfigValidation) -> tuple:
        """Validate Redis URL configuration."""
        if not isinstance(value, str):
            return False, "Must be a string"
        
        # Basic Redis URL validation
        pattern = r'^redis://[^:]+:[^@]+@[^/]+(/\d+)?$'
        if not re.match(pattern, value):
            return False, "Must be a valid Redis connection string"
        
        return True, "Valid"
    
    def _validate_secret_key(self, value: Any, validation: ConfigValidation) -> tuple:
        """Validate secret key configuration."""
        if not isinstance(value, str):
            return False, "Must be a string"
        
        if len(value) < 32:
            return False, "Must be at least 32 characters long"
        
        if value == "your-secret-key-change-in-production" or value == "your-jwt-secret-key-change-in-production":
            return False, "Must be changed from default value"
        
        return True, "Valid"
    
    def _validate_jwt_algorithm(self, value: Any, validation: ConfigValidation) -> tuple:
        """Validate JWT algorithm configuration."""
        valid_algorithms = ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]
        
        if not isinstance(value, str) or value not in valid_algorithms:
            return False, f"Must be one of: {', '.join(valid_algorithms)}"
        
        return True, "Valid"
    
    def _validate_environment(self, value: Any, validation: ConfigValidation) -> tuple:
        """Validate environment configuration."""
        valid_environments = ["development", "staging", "production", "testing"]
        
        if not isinstance(value, str) or value.lower() not in valid_environments:
            return False, f"Must be one of: {', '.join(valid_environments)}"
        
        return True, "Valid"
    
    def _validate_log_level(self, value: Any, validation: ConfigValidation) -> tuple:
        """Validate log level configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        if not isinstance(value, str) or value.upper() not in valid_levels:
            return False, f"Must be one of: {', '.join(valid_levels)}"
        
        return True, "Valid"
    
    def _validate_api_key(self, value: Any, validation: ConfigValidation) -> tuple:
        """Validate API key configuration."""
        if value is None:
            return True, "Valid (optional)"
        
        if not isinstance(value, str):
            return False, "Must be a string"
        
        if len(value) < 10:
            return False, "Must be at least 10 characters long"
        
        return True, "Valid"


def validate_environment_config(env_file_path: str = ".env") -> Dict[str, Any]:
    """
    Validate environment configuration from file or environment variables.
    
    Args:
        env_file_path: Path to environment file
        
    Returns:
        Validation results
    """
    # Load environment variables from file if it exists
    env_vars = {}
    
    if os.path.exists(env_file_path):
        print(f"Loading environment variables from {env_file_path}")
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    else:
        print(f"Environment file {env_file_path} not found, using system environment variables")
        env_vars = dict(os.environ)
    
    # Validate configuration
    validator = EnvironmentValidator()
    result = validator.validate(env_vars)
    
    # Print results
    print("\n" + "="*60)
    print("ENVIRONMENT CONFIGURATION VALIDATION RESULTS")
    print("="*60)
    
    if result["is_valid"]:
        print("✅ Configuration is VALID")
    else:
        print("❌ Configuration has ERRORS")
    
    if result["errors"]:
        print("\n🚨 ERRORS:")
        for error in result["errors"]:
            print(f"  - {error}")
    
    if result["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in result["warnings"]:
            print(f"  - {warning}")
    
    print(f"\n📋 VALIDATED CONFIGURATION ({len(result['config'])} variables):")
    for key, value in sorted(result["config"].items()):
        # Mask sensitive values
        if "password" in key.lower() or "secret" in key.lower() or "key" in key.lower():
            masked_value = "*" * min(len(str(value)), 20)
            print(f"  {key}: {masked_value}")
        else:
            print(f"  {key}: {value}")
    
    print("="*60)
    
    return result


def generate_env_template():
    """Generate environment template file."""
    template_content = """# QLoRA Application Environment Configuration
# Copy this file to .env and update the values

# Database Configuration
DATABASE_URL=mongodb://admin:password@localhost:27017/qlora_db?authSource=admin
MONGO_USERNAME=admin
MONGO_PASSWORD=your-mongodb-password
MONGO_DATABASE=qlora_db

# Redis Configuration  
REDIS_URL=redis://:redis_password@localhost:6379/0
REDIS_PASSWORD=your-redis-password

# Security Configuration
SECRET_KEY=your-32-character-secret-key-here-change-this-in-production
JWT_SECRET_KEY=your-32-character-jwt-secret-key-here-change-this-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
ENABLE_FILE_LOGGING=true
LOG_FILE_PATH=logs/app.log

# GPU Configuration
GPU_MEMORY_THRESHOLD=0.9
ENABLE_GPU_MONITORING=true

# Performance Configuration
MAX_WORKERS=4
ENABLE_PERFORMANCE_MONITORING=true

# Security Configuration
ENABLE_SECURITY_AUDIT=true

# ML Platform Configuration (Optional)
WANDB_API_KEY=your-wandb-api-key
HUGGINGFACE_TOKEN=your-huggingface-token
"""
    
    with open(".env.template", "w") as f:
        f.write(template_content)
    
    print("✅ Environment template generated: .env.template")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QLoRA Environment Configuration Validator")
    parser.add_argument("--env-file", default=".env", help="Path to environment file")
    parser.add_argument("--generate-template", action="store_true", help="Generate environment template")
    
    args = parser.parse_args()
    
    if args.generate_template:
        generate_env_template()
    else:
        result = validate_environment_config(args.env_file)
        
        if not result["is_valid"]:
            sys.exit(1)