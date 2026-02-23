import logging
import logging.handlers
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
from contextlib import contextmanager
import uuid

# Try to import additional logging libraries
try:
    import graypy
    GRAYLOG_AVAILABLE = True
except ImportError:
    GRAYLOG_AVAILABLE = False

try:
    import logstash
    LOGSTASH_AVAILABLE = True
except ImportError:
    LOGSTASH_AVAILABLE = False

class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    """Log categories for better organization."""
    SYSTEM = "system"
    TRAINING = "training"
    GPU = "gpu"
    DATABASE = "database"
    API = "api"
    AUTHENTICATION = "authentication"
    SECURITY = "security"
    DATA_PROCESSING = "data_processing"
    ERROR_HANDLING = "error_handling"
    MONITORING = "monitoring"
    PERFORMANCE = "performance"

@dataclass
class LogContext:
    """Structured log context."""
    job_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    environment: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class StructuredLogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    category: str
    message: str
    logger_name: str
    context: LogContext
    thread_name: str
    process_id: int
    exception: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self, include_context: bool = True, include_environment: bool = True):
        super().__init__()
        self.include_context = include_context
        self.include_environment = include_environment
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Extract structured data
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "category": getattr(record, "category", LogCategory.SYSTEM.value),
            "message": record.getMessage(),
            "logger_name": record.name,
            "thread_name": record.threadName,
            "process_id": record.process,
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName
        }
        
        # Add context if available
        if self.include_context and hasattr(record, "context"):
            log_entry["context"] = asdict(record.context) if isinstance(record.context, LogContext) else record.context
        
        # Add environment if available
        if self.include_environment and hasattr(record, "environment"):
            log_entry["environment"] = record.environment
        
        # Add performance metrics if available
        if hasattr(record, "performance_metrics"):
            log_entry["performance_metrics"] = record.performance_metrics
        
        # Add exception details if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "exc_info", "exc_text", "stack_info", 
                          "lineno", "funcName", "created", "msecs", "relativeCreated", 
                          "thread", "threadName", "processName", "process", "getMessage",
                          "category", "context", "environment", "performance_metrics"]:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)

class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for non-blocking logging."""
    
    def __init__(self, handler: logging.Handler, max_queue_size: int = 10000):
        super().__init__()
        self.handler = handler
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.worker_thread.start()
    
    def emit(self, record: logging.LogRecord):
        """Emit log record to queue."""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # Drop log if queue is full to prevent blocking
            pass
    
    def _process_logs(self):
        """Process logs from queue."""
        while self.running:
            try:
                record = self.queue.get(timeout=1)
                self.handler.emit(record)
            except queue.Empty:
                continue
            except Exception as e:
                # Log to stderr if handler fails
                print(f"Error processing log: {e}", file=sys.stderr)
    
    def close(self):
        """Close handler and cleanup."""
        self.running = False
        self.worker_thread.join(timeout=5)
        self.handler.close()
        super().close()

class StructuredLogger:
    """Enhanced structured logger with context management."""
    
    def __init__(self, name: str, category: LogCategory = LogCategory.SYSTEM):
        self.name = name
        self.category = category
        self.logger = logging.getLogger(name)
        self.context_stack = []
        self.performance_stack = []
    
    def _log_with_context(self, level: LogLevel, message: str, 
                         context: Optional[LogContext] = None, 
                         exc_info: Optional[Exception] = None,
                         **kwargs):
        """Log with context and structured data."""
        
        # Merge with current context
        final_context = self._merge_contexts(context)
        
        # Create log record with structured data
        extra = {
            "category": self.category.value,
            "context": final_context,
            "environment": self._get_environment_info()
        }
        
        # Add performance metrics if available
        if self.performance_stack:
            extra["performance_metrics"] = self.performance_stack[-1]
        
        # Add extra fields
        extra.update(kwargs)
        
        # Log the message
        self.logger.log(
            getattr(logging, level.value),
            message,
            exc_info=exc_info,
            extra=extra
        )
    
    def debug(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log debug message."""
        self._log_with_context(LogLevel.DEBUG, message, context, **kwargs)
    
    def info(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log info message."""
        self._log_with_context(LogLevel.INFO, message, context, **kwargs)
    
    def warning(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log warning message."""
        self._log_with_context(LogLevel.WARNING, message, context, **kwargs)
    
    def error(self, message: str, context: Optional[LogContext] = None, 
              exc_info: Optional[Exception] = None, **kwargs):
        """Log error message."""
        self._log_with_context(LogLevel.ERROR, message, context, exc_info, **kwargs)
    
    def critical(self, message: str, context: Optional[LogContext] = None, 
                  exc_info: Optional[Exception] = None, **kwargs):
        """Log critical message."""
        self._log_with_context(LogLevel.CRITICAL, message, context, exc_info, **kwargs)
    
    @contextmanager
    def context(self, **context_kwargs):
        """Context manager for adding structured context to logs."""
        context = LogContext(**context_kwargs)
        self.context_stack.append(context)
        try:
            yield self
        finally:
            if self.context_stack:
                self.context_stack.pop()
    
    @contextmanager
    def performance_timer(self, operation: str, **metadata):
        """Context manager for timing operations."""
        start_time = datetime.now()
        performance_data = {
            "operation": operation,
            "start_time": start_time.isoformat(),
            "metadata": metadata
        }
        
        self.performance_stack.append(performance_data)
        
        try:
            yield self
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if self.performance_stack:
                perf_data = self.performance_stack.pop()
                perf_data.update({
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration,
                    "duration_ms": duration * 1000
                })
                
                # Log performance metrics
                self.info(f"Performance: {operation} completed in {duration:.3f}s", 
                         context=LogContext(operation=operation, metadata=metadata),
                         performance_metrics=perf_data)
    
    def _merge_contexts(self, new_context: Optional[LogContext]) -> LogContext:
        """Merge new context with existing context stack."""
        if not self.context_stack and not new_context:
            return LogContext()
        
        # Start with current context or empty context
        if self.context_stack:
            merged = asdict(self.context_stack[-1])
        else:
            merged = asdict(LogContext())
        
        # Merge new context
        if new_context:
            new_context_dict = asdict(new_context)
            for key, value in new_context_dict.items():
                if value is not None:
                    merged[key] = value
        
        return LogContext(**merged)
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get current environment information."""
        try:
            import psutil
            import torch
            
            env_info = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
            
            # GPU information if available
            if torch.cuda.is_available():
                env_info.update({
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_cached_gb": torch.cuda.memory_reserved() / (1024**3)
                })
            
            return env_info
            
        except Exception:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version,
                "platform": sys.platform
            }

class LoggingManager:
    """Centralized logging manager."""
    
    def __init__(self, log_dir: str = "./logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.loggers = {}
        self.handlers = []
        self.initialized = False
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
    
    def initialize_logging(self, config: Optional[Dict[str, Any]] = None):
        """Initialize logging system with multiple handlers."""
        if self.initialized:
            return
        
        config = config or {}
        
        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create structured formatter
        formatter = StructuredFormatter(
            include_context=config.get("include_context", True),
            include_environment=config.get("include_environment", True)
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.log_level)
        self.handlers.append(console_handler)
        
        # File handlers
        # Main application log
        app_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "app.log",
            maxBytes=config.get("max_file_size", 50 * 1024 * 1024),  # 50MB
            backupCount=config.get("backup_count", 5)
        )
        app_handler.setFormatter(formatter)
        app_handler.setLevel(self.log_level)
        self.handlers.append(app_handler)
        
        # Error log
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=config.get("max_file_size", 50 * 1024 * 1024),
            backupCount=config.get("backup_count", 5)
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        self.handlers.append(error_handler)
        
        # Performance log
        performance_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "performance.log",
            maxBytes=config.get("max_file_size", 50 * 1024 * 1024),
            backupCount=config.get("backup_count", 5)
        )
        performance_handler.setFormatter(formatter)
        performance_handler.setLevel(logging.INFO)
        self.handlers.append(performance_handler)
        
        # Security log
        security_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "security.log",
            maxBytes=config.get("max_file_size", 50 * 1024 * 1024),
            backupCount=config.get("backup_count", 5)
        )
        security_handler.setFormatter(formatter)
        security_handler.setLevel(logging.INFO)
        self.handlers.append(security_handler)
        
        # External log handlers (if configured)
        if config.get("enable_graylog", False) and GRAYLOG_AVAILABLE:
            graylog_handler = graypy.GELFHandler(
                config.get("graylog_host", "localhost"),
                config.get("graylog_port", 12201)
            )
            graylog_handler.setFormatter(formatter)
            self.handlers.append(graylog_handler)
        
        if config.get("enable_logstash", False) and LOGSTASH_AVAILABLE:
            logstash_handler = logstash.LogstashHandler(
                config.get("logstash_host", "localhost"),
                config.get("logstash_port", 5000),
                version=1
            )
            logstash_handler.setFormatter(formatter)
            self.handlers.append(logstash_handler)
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        for handler in self.handlers:
            root_logger.addHandler(handler)
        
        self.initialized = True
        self.get_logger("logging_manager").info("Logging system initialized successfully")
    
    def get_logger(self, name: str, category: LogCategory = LogCategory.SYSTEM) -> StructuredLogger:
        """Get structured logger instance."""
        if not self.initialized:
            self.initialize_logging()
        
        if name not in self.loggers:
            self.loggers[name] = StructuredLogger(name, category)
        
        return self.loggers[name]
    
    def shutdown(self):
        """Shutdown logging system."""
        if not self.initialized:
            return
        
        self.get_logger("logging_manager").info("Shutting down logging system")
        
        # Remove handlers from root logger
        root_logger = logging.getLogger()
        for handler in self.handlers:
            root_logger.removeHandler(handler)
            handler.close()
        
        # Shutdown logging
        logging.shutdown()
        self.initialized = False

# Global logging manager instance
_global_logging_manager = None

def initialize_logging(log_dir: str = "./logs", log_level: str = "INFO", config: Optional[Dict[str, Any]] = None):
    """Initialize global logging system."""
    global _global_logging_manager
    
    if _global_logging_manager is None:
        _global_logging_manager = LoggingManager(log_dir, log_level)
        _global_logging_manager.initialize_logging(config)
        print(f"Global logging system initialized at {log_dir} with level {log_level}")
    
    return _global_logging_manager

def get_logger(name: str, category: LogCategory = LogCategory.SYSTEM) -> StructuredLogger:
    """Get global structured logger."""
    if _global_logging_manager is None:
        initialize_logging()
    
    return _global_logging_manager.get_logger(name, category)

def shutdown_logging():
    """Shutdown global logging system."""
    global _global_logging_manager
    
    if _global_logging_manager:
        _global_logging_manager.shutdown()
        _global_logging_manager = None

# Performance monitoring utilities
class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name, LogCategory.PERFORMANCE)
        self.metrics = {}
    
    @contextmanager
    def measure_operation(self, operation_name: str, **metadata):
        """Measure operation performance."""
        start_time = datetime.now()
        start_memory = self._get_memory_usage()
        
        try:
            yield self
        finally:
            end_time = datetime.now()
            end_memory = self._get_memory_usage()
            
            duration = (end_time - start_time).total_seconds()
            memory_delta = end_memory - start_memory
            
            # Log performance metrics
            self.logger.info(
                f"Performance measurement: {operation_name}",
                context=LogContext(operation=operation_name, metadata=metadata),
                performance_metrics={
                    "operation": operation_name,
                    "duration_seconds": duration,
                    "duration_ms": duration * 1000,
                    "memory_start_mb": start_memory,
                    "memory_end_mb": end_memory,
                    "memory_delta_mb": memory_delta,
                    "timestamp": start_time.isoformat()
                }
            )
            
            # Store metrics
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            
            self.metrics[operation_name].append({
                "duration": duration,
                "memory_delta": memory_delta,
                "timestamp": start_time.isoformat()
            })
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            return 0.0
    
    def get_metrics_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if operation_name:
            metrics = self.metrics.get(operation_name, [])
            if not metrics:
                return {"operation": operation_name, "count": 0}
            
            durations = [m["duration"] for m in metrics]
            return {
                "operation": operation_name,
                "count": len(metrics),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "recent_metrics": metrics[-10:]  # Last 10 measurements
            }
        else:
            summary = {}
            for op_name, metrics in self.metrics.items():
                if metrics:
                    durations = [m["duration"] for m in metrics]
                    summary[op_name] = {
                        "count": len(metrics),
                        "avg_duration": sum(durations) / len(durations),
                        "min_duration": min(durations),
                        "max_duration": max(durations)
                    }
            return summary

# Error tracking utilities
class ErrorTracker:
    """Error tracking and reporting utilities."""
    
    def __init__(self, logger_name: str = "error_tracker"):
        self.logger = get_logger(logger_name, LogCategory.ERROR_HANDLING)
        self.errors = []
    
    def track_error(self, error: Exception, context: Optional[LogContext] = None, **metadata):
        """Track error with context."""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": asdict(context) if context else {},
            "metadata": metadata,
            "traceback": traceback.format_exc()
        }
        
        self.errors.append(error_info)
        
        # Log the error
        self.logger.error(
            f"Error tracked: {type(error).__name__}: {str(error)}",
            context=context,
            exc_info=error,
            **metadata
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        if not self.errors:
            return {"total_errors": 0}
        
        error_types = {}
        for error in self.errors:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.errors),
            "error_types": error_types,
            "recent_errors": self.errors[-10:],  # Last 10 errors
            "first_error": self.errors[0] if self.errors else None,
            "last_error": self.errors[-1] if self.errors else None
        }
    
    def clear_errors(self):
        """Clear tracked errors."""
        self.errors.clear()

# Global performance monitor and error tracker
_global_performance_monitor = None
_global_error_tracker = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    global _global_performance_monitor
    
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    
    return _global_performance_monitor

def get_error_tracker() -> ErrorTracker:
    """Get global error tracker."""
    global _global_error_tracker
    
    if _global_error_tracker is None:
        _global_error_tracker = ErrorTracker()
    
    return _global_error_tracker