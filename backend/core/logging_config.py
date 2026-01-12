import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
from pythonjsonlogger import jsonlogger
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Create logs directory
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

class CustomJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter untuk structured logging."""
    
    def add_fields(self, log_record, record, message_dict):
        super(CustomJSONFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add custom fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add extra fields if available
        if hasattr(record, 'job_id'):
            log_record['job_id'] = record.job_id
        
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        
        if hasattr(record, 'gpu_id'):
            log_record['gpu_id'] = record.gpu_id
        
        if hasattr(record, 'model_name'):
            log_record['model_name'] = record.model_name

def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_format: str = "json",
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> Dict[str, logging.Logger]:
    """
    Setup comprehensive logging system untuk QLoRA training platform.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        log_format: Log format (json or text)
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Dictionary of loggers for different components
    """
    
    # Validate log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create formatters
    if log_format == "json":
        formatter = CustomJSONFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Setup loggers
    loggers = {}
    
    # Main application logger
    app_logger = logging.getLogger("qlora_app")
    app_logger.setLevel(numeric_level)
    loggers["app"] = app_logger
    
    # Training logger
    training_logger = logging.getLogger("qlora_training")
    training_logger.setLevel(numeric_level)
    loggers["training"] = training_logger
    
    # API logger
    api_logger = logging.getLogger("qlora_api")
    api_logger.setLevel(numeric_level)
    loggers["api"] = api_logger
    
    # GPU logger
    gpu_logger = logging.getLogger("qlora_gpu")
    gpu_logger.setLevel(numeric_level)
    loggers["gpu"] = gpu_logger
    
    # Database logger
    db_logger = logging.getLogger("qlora_db")
    db_logger.setLevel(numeric_level)
    loggers["db"] = db_logger
    
    # Security logger
    security_logger = logging.getLogger("qlora_security")
    security_logger.setLevel(numeric_level)
    loggers["security"] = security_logger
    
    # Clear existing handlers
    for logger_instance in loggers.values():
        logger_instance.handlers.clear()
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        
        for logger_instance in loggers.values():
            logger_instance.addHandler(console_handler)
    
    # File handlers
    if log_to_file:
        # Application log file
        app_file_handler = RotatingFileHandler(
            LOGS_DIR / "app.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        app_file_handler.setLevel(numeric_level)
        app_file_handler.setFormatter(formatter)
        app_logger.addHandler(app_file_handler)
        
        # Training log file (separate for training-specific logs)
        training_file_handler = RotatingFileHandler(
            LOGS_DIR / "training.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        training_file_handler.setLevel(numeric_level)
        training_file_handler.setFormatter(formatter)
        training_logger.addHandler(training_file_handler)
        
        # Error log file (for errors only)
        error_file_handler = RotatingFileHandler(
            LOGS_DIR / "errors.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(formatter)
        
        # Add error handler to all loggers
        for logger_instance in loggers.values():
            logger_instance.addHandler(error_file_handler)
        
        # Daily rotating logs for long-term storage
        daily_handler = TimedRotatingFileHandler(
            LOGS_DIR / "daily.log",
            when="midnight",
            interval=1,
            backupCount=30  # Keep 30 days
        )
        daily_handler.setLevel(numeric_level)
        daily_handler.setFormatter(formatter)
        
        # Add daily handler to main loggers
        app_logger.addHandler(daily_handler)
        training_logger.addHandler(daily_handler)
    
    # Setup external service logging (WandB, MLflow, etc.)
    setup_external_logging(numeric_level)
    
    # Log startup message
    app_logger.info("QLoRA Logging system initialized")
    app_logger.info(f"Log level: {log_level}")
    app_logger.info(f"Log format: {log_format}")
    app_logger.info(f"Logs directory: {LOGS_DIR}")
    
    return loggers

def setup_external_logging(level: int):
    """Setup logging untuk external services."""
    
    # Suppress overly verbose external libraries
    external_loggers = [
        "urllib3",
        "requests",
        "botocore",
        "boto3",
        "s3transfer",
        "transformers",
        "datasets",
        "accelerate",
        "wandb",
        "tensorboard",
        "matplotlib",
        "seaborn",
        "PIL",
        "Pillow"
    ]
    
    for logger_name in external_loggers:
        external_logger = logging.getLogger(logger_name)
        external_logger.setLevel(max(level, logging.WARNING))

def get_training_logger(job_id: str = None) -> logging.Logger:
    """
    Get training logger dengan job-specific context.
    
    Args:
        job_id: Training job ID untuk context
        
    Returns:
        Logger instance dengan training context
    """
    logger = logging.getLogger("qlora_training")
    
    if job_id:
        # Create custom adapter dengan job context
        class JobContextAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                kwargs.setdefault("extra", {})
                kwargs["extra"]["job_id"] = self.extra.get("job_id")
                return msg, kwargs
        
        return JobContextAdapter(logger, {"job_id": job_id})
    
    return logger

def get_api_logger(request_id: str = None, user_id: str = None) -> logging.Logger:
    """
    Get API logger dengan request context.
    
    Args:
        request_id: Request ID untuk tracking
        user_id: User ID untuk context
        
    Returns:
        Logger instance dengan API context
    """
    logger = logging.getLogger("qlora_api")
    
    if request_id or user_id:
        class APIContextAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                kwargs.setdefault("extra", {})
                if self.extra.get("request_id"):
                    kwargs["extra"]["request_id"] = self.extra.get("request_id")
                if self.extra.get("user_id"):
                    kwargs["extra"]["user_id"] = self.extra.get("user_id")
                return msg, kwargs
        
        return APIContextAdapter(logger, {"request_id": request_id, "user_id": user_id})
    
    return logger

def log_gpu_metrics(gpu_id: int, metrics: Dict[str, Any]):
    """Log GPU metrics untuk monitoring."""
    logger = logging.getLogger("qlora_gpu")
    logger.info(
        f"GPU {gpu_id} metrics",
        extra={
            "gpu_id": gpu_id,
            "gpu_metrics": metrics
        }
    )

def log_security_event(event: str, user_id: str = None, details: Dict[str, Any] = None):
    """Log security events untuk audit trail."""
    logger = logging.getLogger("qlora_security")
    logger.info(
        f"Security event: {event}",
        extra={
            "security_event": event,
            "user_id": user_id,
            "details": details or {}
        }
    )

def log_training_metrics(job_id: str, metrics: Dict[str, Any]):
    """Log training metrics untuk analysis."""
    logger = get_training_logger(job_id)
    logger.info(
        "Training metrics",
        extra={
            "metrics": metrics
        }
    )

# Performance monitoring
class PerformanceMonitor:
    """Performance monitoring untuk training operations."""
    
    def __init__(self, logger_name: str = "qlora_performance"):
        self.logger = logging.getLogger(logger_name)
        self.start_time = None
        self.operation_name = None
    
    def start_operation(self, operation_name: str):
        """Start performance monitoring untuk operation."""
        self.operation_name = operation_name
        self.start_time = datetime.now()
        self.logger.info(f"Started operation: {operation_name}")
    
    def end_operation(self, additional_metrics: Dict[str, Any] = None):
        """End performance monitoring dan log metrics."""
        if self.start_time and self.operation_name:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            metrics = {
                "operation": self.operation_name,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            }
            
            if additional_metrics:
                metrics.update(additional_metrics)
            
            self.logger.info(f"Completed operation: {self.operation_name}", extra=metrics)
            
            # Reset
            self.start_time = None
            self.operation_name = None

# Initialize logging system
loggers = setup_logging()

# Export commonly used loggers
app_logger = loggers["app"]
training_logger = loggers["training"]
api_logger = loggers["api"]
gpu_logger = loggers["gpu"]
db_logger = loggers["db"]
security_logger = loggers["security"]