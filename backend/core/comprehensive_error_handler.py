import logging
import asyncio
import traceback
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import json
import hashlib
import time
from pathlib import Path
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories."""
    SYSTEM = "system"
    GPU = "gpu"
    MEMORY = "memory"
    NETWORK = "network"
    DATABASE = "database"
    TRAINING = "training"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"

class RecoveryAction(Enum):
    """Recovery actions."""
    RETRY = "retry"
    RESTART = "restart"
    ROLLBACK = "rollback"
    CLEANUP = "cleanup"
    NOTIFY = "notify"
    SKIP = "skip"
    TERMINATE = "terminate"

@dataclass
class ErrorContext:
    """Error context information."""
    job_id: str
    operation: str
    step: int
    epoch: float
    component: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: datetime
    environment: Dict[str, Any]
    stack_trace: str
    recovery_attempted: bool = False
    recovery_successful: bool = False

@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration."""
    action: RecoveryAction
    max_retries: int
    retry_delay: float
    backoff_multiplier: float
    cleanup_required: bool
    notification_required: bool
    rollback_required: bool
    custom_handler: Optional[Callable] = None

@dataclass
class ErrorReport:
    """Comprehensive error report."""
    error_id: str
    job_id: str
    error_type: str
    error_message: str
    context: ErrorContext
    recovery_strategy: RecoveryStrategy
    retry_count: int
    recovery_history: List[Dict[str, Any]]
    created_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_status: str = "pending"

class ErrorClassifier:
    """Classifies errors for appropriate handling."""
    
    def __init__(self):
        self.error_patterns = {
            ErrorCategory.GPU: [
                "CUDA out of memory",
                "CUDA error",
                "GPU memory",
                "device-side assert",
                "invalid device function"
            ],
            ErrorCategory.MEMORY: [
                "out of memory",
                "MemoryError",
                "malloc",
                "allocation failed",
                "cannot allocate memory"
            ],
            ErrorCategory.NETWORK: [
                "ConnectionError",
                "TimeoutError",
                "NetworkError",
                "HTTPError",
                "Connection refused"
            ],
            ErrorCategory.DATABASE: [
                "pymongo",
                "MongoDB",
                "database error",
                "connection failed",
                "timeout"
            ],
            ErrorCategory.TRAINING: [
                "RuntimeError",
                "ValueError",
                "KeyError",
                "IndexError",
                "Division by zero"
            ],
            ErrorCategory.VALIDATION: [
                "ValidationError",
                "invalid format",
                "schema validation",
                "data validation"
            ],
            ErrorCategory.AUTHENTICATION: [
                "AuthenticationError",
                "Unauthorized",
                "Forbidden",
                "Invalid token",
                "Authentication failed"
            ],
            ErrorCategory.SYSTEM: [
                "OSError",
                "IOError",
                "FileNotFoundError",
                "PermissionError",
                "SystemError"
            ]
        }
        
        self.severity_patterns = {
            ErrorSeverity.CRITICAL: [
                "CUDA out of memory",
                "system crash",
                "irrecoverable",
                "fatal error"
            ],
            ErrorSeverity.HIGH: [
                "out of memory",
                "database connection failed",
                "authentication failed"
            ],
            ErrorSeverity.MEDIUM: [
                "validation failed",
                "timeout",
                "retry"
            ],
            ErrorSeverity.LOW: [
                "warning",
                "info",
                "deprecated"
            ]
        }
    
    def classify_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> tuple:
        """Classify error and return category and severity."""
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Determine category
        category = ErrorCategory.UNKNOWN
        for cat, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern.lower() in error_message or pattern.lower() in error_type.lower():
                    category = cat
                    break
            if category != ErrorCategory.UNKNOWN:
                break
        
        # Determine severity
        severity = ErrorSeverity.MEDIUM  # Default
        for sev, patterns in self.severity_patterns.items():
            for pattern in patterns:
                if pattern.lower() in error_message:
                    severity = sev
                    break
            if severity != ErrorSeverity.MEDIUM:
                break
        
        return category, severity

class RecoveryManager:
    """Manages error recovery strategies."""
    
    def __init__(self, db=None):
        self.db = db
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.active_recoveries = {}  # job_id -> recovery state
        
    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, RecoveryStrategy]:
        """Initialize recovery strategies for different error categories."""
        return {
            ErrorCategory.GPU: RecoveryStrategy(
                action=RecoveryAction.CLEANUP,
                max_retries=3,
                retry_delay=30.0,
                backoff_multiplier=2.0,
                cleanup_required=True,
                notification_required=True,
                rollback_required=False
            ),
            ErrorCategory.MEMORY: RecoveryStrategy(
                action=RecoveryAction.RESTART,
                max_retries=2,
                retry_delay=60.0,
                backoff_multiplier=1.5,
                cleanup_required=True,
                notification_required=True,
                rollback_required=False
            ),
            ErrorCategory.NETWORK: RecoveryStrategy(
                action=RecoveryAction.RETRY,
                max_retries=5,
                retry_delay=10.0,
                backoff_multiplier=1.2,
                cleanup_required=False,
                notification_required=False,
                rollback_required=False
            ),
            ErrorCategory.DATABASE: RecoveryStrategy(
                action=RecoveryAction.RETRY,
                max_retries=3,
                retry_delay=15.0,
                backoff_multiplier=1.3,
                cleanup_required=False,
                notification_required=True,
                rollback_required=False
            ),
            ErrorCategory.TRAINING: RecoveryStrategy(
                action=RecoveryAction.ROLLBACK,
                max_retries=2,
                retry_delay=20.0,
                backoff_multiplier=1.4,
                cleanup_required=True,
                notification_required=True,
                rollback_required=True
            ),
            ErrorCategory.VALIDATION: RecoveryStrategy(
                action=RecoveryAction.SKIP,
                max_retries=1,
                retry_delay=0.0,
                backoff_multiplier=1.0,
                cleanup_required=False,
                notification_required=False,
                rollback_required=False
            ),
            ErrorCategory.AUTHENTICATION: RecoveryStrategy(
                action=RecoveryAction.TERMINATE,
                max_retries=0,
                retry_delay=0.0,
                backoff_multiplier=1.0,
                cleanup_required=False,
                notification_required=True,
                rollback_required=False
            ),
            ErrorCategory.SYSTEM: RecoveryStrategy(
                action=RecoveryAction.RESTART,
                max_retries=1,
                retry_delay=30.0,
                backoff_multiplier=2.0,
                cleanup_required=True,
                notification_required=True,
                rollback_required=True
            )
        }
    
    def get_recovery_strategy(self, error_category: ErrorCategory) -> RecoveryStrategy:
        """Get recovery strategy for error category."""
        return self.recovery_strategies.get(error_category, RecoveryStrategy(
            action=RecoveryAction.NOTIFY,
            max_retries=1,
            retry_delay=30.0,
            backoff_multiplier=1.0,
            cleanup_required=False,
            notification_required=True,
            rollback_required=False
        ))
    
    async def execute_recovery(self, error_report: ErrorReport, 
                             context: Dict[str, Any]) -> bool:
        """Execute recovery strategy."""
        job_id = error_report.job_id
        strategy = error_report.recovery_strategy
        
        logger.info(f"Executing recovery for job {job_id}: {strategy.action.value}")
        
        try:
            # Initialize recovery state
            if job_id not in self.active_recoveries:
                self.active_recoveries[job_id] = {
                    "retry_count": 0,
                    "last_retry": None,
                    "recovery_history": []
                }
            
            recovery_state = self.active_recoveries[job_id]
            
            # Check retry limits
            if recovery_state["retry_count"] >= strategy.max_retries:
                logger.warning(f"Max retries ({strategy.max_retries}) reached for job {job_id}")
                return False
            
            # Calculate retry delay with backoff
            if recovery_state["last_retry"]:
                time_since_last = (datetime.now() - recovery_state["last_retry"]).total_seconds()
                required_delay = strategy.retry_delay * (strategy.backoff_multiplier ** recovery_state["retry_count"])
                
                if time_since_last < required_delay:
                    wait_time = required_delay - time_since_last
                    logger.info(f"Waiting {wait_time:.1f}s before retry {recovery_state['retry_count'] + 1}")
                    await asyncio.sleep(wait_time)
            
            # Execute recovery action
            recovery_success = await self._execute_recovery_action(
                strategy, error_report, context
            )
            
            # Update recovery state
            recovery_state["retry_count"] += 1
            recovery_state["last_retry"] = datetime.now()
            recovery_state["recovery_history"].append({
                "attempt": recovery_state["retry_count"],
                "action": strategy.action.value,
                "success": recovery_success,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update error report
            error_report.recovery_history = recovery_state["recovery_history"]
            error_report.context.recovery_attempted = True
            error_report.context.recovery_successful = recovery_success
            
            logger.info(f"Recovery attempt {recovery_state['retry_count']} for job {job_id}: {'SUCCESS' if recovery_success else 'FAILED'}")
            
            return recovery_success
            
        except Exception as e:
            logger.error(f"Error executing recovery for job {job_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    async def _execute_recovery_action(self, strategy: RecoveryStrategy, 
                                     error_report: ErrorReport, 
                                     context: Dict[str, Any]) -> bool:
        """Execute specific recovery action."""
        action = strategy.action
        
        try:
            if action == RecoveryAction.RETRY:
                return await self._retry_operation(error_report, context)
            
            elif action == RecoveryAction.RESTART:
                return await self._restart_training(error_report, context)
            
            elif action == RecoveryAction.ROLLBACK:
                return await self._rollback_to_checkpoint(error_report, context)
            
            elif action == RecoveryAction.CLEANUP:
                return await self._cleanup_resources(error_report, context)
            
            elif action == RecoveryAction.NOTIFY:
                return await self._notify_admin(error_report, context)
            
            elif action == RecoveryAction.SKIP:
                return await self._skip_operation(error_report, context)
            
            elif action == RecoveryAction.TERMINATE:
                return await self._terminate_training(error_report, context)
            
            elif strategy.custom_handler:
                return await strategy.custom_handler(error_report, context)
            
            else:
                logger.warning(f"Unknown recovery action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing recovery action {action}: {str(e)}")
            return False
    
    async def _retry_operation(self, error_report: ErrorReport, 
                              context: Dict[str, Any]) -> bool:
        """Retry the failed operation."""
        logger.info(f"Retrying operation for job {error_report.job_id}")
        
        # Simple retry - the calling code should handle the actual retry logic
        await asyncio.sleep(2)  # Brief delay before retry
        return True
    
    async def _restart_training(self, error_report: ErrorReport, 
                              context: Dict[str, Any]) -> bool:
        """Restart training from the beginning."""
        logger.info(f"Restarting training for job {error_report.job_id}")
        
        try:
            # Cleanup current training state
            await self._cleanup_resources(error_report, context)
            
            # Reset training state in database
            if self.db:
                self.db.training_jobs.update_one(
                    {"id": error_report.job_id},
                    {
                        "$set": {
                            "status": "restarting",
                            "current_step": 0,
                            "current_epoch": 0,
                            "progress": 0.0,
                            "restarted_at": datetime.now().isoformat()
                        }
                    }
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error restarting training: {str(e)}")
            return False
    
    async def _rollback_to_checkpoint(self, error_report: ErrorReport, 
                                    context: Dict[str, Any]) -> bool:
        """Rollback to the last good checkpoint."""
        logger.info(f"Rolling back training for job {error_report.job_id}")
        
        try:
            # Find latest checkpoint
            if self.db:
                latest_checkpoint = self.db.checkpoints.find_one(
                    {"training_job_id": error_report.job_id},
                    sort=[("step", -1)]
                )
                
                if latest_checkpoint:
                    checkpoint_step = latest_checkpoint["step"]
                    logger.info(f"Rolling back to checkpoint at step {checkpoint_step}")
                    
                    # Update training state to rollback point
                    self.db.training_jobs.update_one(
                        {"id": error_report.job_id},
                        {
                            "$set": {
                                "status": "rolling_back",
                                "current_step": checkpoint_step,
                                "rolled_back_at": datetime.now().isoformat(),
                                "rollback_checkpoint": checkpoint_step
                            }
                        }
                    )
                    
                    return True
                else:
                    logger.warning("No checkpoint found for rollback")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error rolling back training: {str(e)}")
            return False
    
    async def _cleanup_resources(self, error_report: ErrorReport, 
                               context: Dict[str, Any]) -> bool:
        """Cleanup resources (GPU memory, temp files, etc.)."""
        logger.info(f"Cleaning up resources for job {error_report.job_id}")
        
        try:
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleaned")
            
            # Cleanup temporary files
            temp_dirs = [
                f"./temp/{error_report.job_id}",
                f"./cache/{error_report.job_id}",
                f"./logs/{error_report.job_id}"
            ]
            
            for temp_dir in temp_dirs:
                try:
                    if Path(temp_dir).exists():
                        shutil.rmtree(temp_dir)
                        logger.info(f"Cleaned up {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_dir}: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {str(e)}")
            return False
    
    async def _notify_admin(self, error_report: ErrorReport, 
                          context: Dict[str, Any]) -> bool:
        """Notify administrators of error."""
        logger.info(f"Notifying administrators for job {error_report.job_id}")
        
        try:
            # Create notification record
            notification = {
                "id": f"notification_{error_report.error_id}",
                "job_id": error_report.job_id,
                "error_id": error_report.error_id,
                "severity": error_report.context.severity.value,
                "category": error_report.context.category.value,
                "message": f"Training job {error_report.job_id} encountered {error_report.context.category.value} error",
                "created_at": datetime.now().isoformat(),
                "acknowledged": False
            }
            
            # Store in database
            if self.db:
                self.db.notifications.insert_one(notification)
            
            # TODO: Send email/Slack notification
            logger.info(f"Admin notification created for error {error_report.error_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error notifying admin: {str(e)}")
            return False
    
    async def _skip_operation(self, error_report: ErrorReport, 
                             context: Dict[str, Any]) -> bool:
        """Skip the failed operation and continue."""
        logger.info(f"Skipping operation for job {error_report.job_id}")
        
        # Mark operation as skipped in database
        if self.db:
            self.db.training_jobs.update_one(
                {"id": error_report.job_id},
                {
                    "$set": {
                        "skipped_operations": {
                            "operation": error_report.context.operation,
                            "step": error_report.context.step,
                            "reason": "error_recovery",
                            "skipped_at": datetime.now().isoformat()
                        }
                    }
                }
            )
        
        return True
    
    async def _terminate_training(self, error_report: ErrorReport, 
                                context: Dict[str, Any]) -> bool:
        """Terminate training due to unrecoverable error."""
        logger.info(f"Terminating training for job {error_report.job_id}")
        
        try:
            # Update training status
            if self.db:
                self.db.training_jobs.update_one(
                    {"id": error_report.job_id},
                    {
                        "$set": {
                            "status": "terminated",
                            "termination_reason": "unrecoverable_error",
                            "terminated_at": datetime.now().isoformat()
                        }
                    }
                )
            
            # Cleanup resources
            await self._cleanup_resources(error_report, context)
            
            return True
            
        except Exception as e:
            logger.error(f"Error terminating training: {str(e)}")
            return False
    
    def cleanup_job_recoveries(self, job_id: str):
        """Cleanup recovery state for a job."""
        if job_id in self.active_recoveries:
            del self.active_recoveries[job_id]
            logger.info(f"Cleaned up recovery state for job {job_id}")

class ErrorReporter:
    """Comprehensive error reporting system."""
    
    def __init__(self, db=None, recovery_manager: Optional[RecoveryManager] = None):
        self.db = db
        self.recovery_manager = recovery_manager or RecoveryManager(db)
        self.error_classifier = ErrorClassifier()
        self.error_history = deque(maxlen=1000)  # Keep last 1000 errors
        
    def create_error_report(self, error: Exception, context: Dict[str, Any]) -> ErrorReport:
        """Create comprehensive error report."""
        job_id = context.get("job_id", "unknown")
        operation = context.get("operation", "unknown")
        
        # Classify error
        category, severity = self.error_classifier.classify_error(error, context)
        
        # Create error context
        error_context = ErrorContext(
            job_id=job_id,
            operation=operation,
            step=context.get("step", 0),
            epoch=context.get("epoch", 0.0),
            component=context.get("component", "unknown"),
            severity=severity,
            category=category,
            timestamp=datetime.now(),
            environment=self._collect_environment_info(),
            stack_trace=traceback.format_exc()
        )
        
        # Get recovery strategy
        recovery_strategy = self.recovery_manager.get_recovery_strategy(category)
        
        # Generate error ID
        error_id = self._generate_error_id(error, job_id, operation)
        
        # Create error report
        error_report = ErrorReport(
            error_id=error_id,
            job_id=job_id,
            error_type=type(error).__name__,
            error_message=str(error),
            context=error_context,
            recovery_strategy=recovery_strategy,
            retry_count=0,
            recovery_history=[],
            created_at=datetime.now()
        )
        
        # Store in history
        self.error_history.append(error_report)
        
        return error_report
    
    def _generate_error_id(self, error: Exception, job_id: str, operation: str) -> str:
        """Generate unique error ID."""
        error_string = f"{job_id}_{operation}_{type(error).__name__}_{str(error)[:100]}"
        return hashlib.md5(error_string.encode()).hexdigest()[:12]
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect environment information."""
        try:
            import psutil
            import torch
            
            env_info = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
            
            # GPU information
            if torch.cuda.is_available():
                env_info.update({
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_cached_gb": torch.cuda.memory_reserved() / (1024**3)
                })
            
            return env_info
            
        except Exception as e:
            logger.error(f"Error collecting environment info: {str(e)}")
            return {"error": "Failed to collect environment info"}
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle error with comprehensive reporting and recovery."""
        try:
            # Create error report
            error_report = self.create_error_report(error, context)
            
            # Log error
            logger.error(f"Error reported for job {error_report.job_id}: {error_report.error_message}")
            logger.error(f"Category: {error_report.context.category.value}, Severity: {error_report.context.severity.value}")
            
            # Store in database
            if self.db:
                await self._store_error_report(error_report)
            
            # Execute recovery strategy
            recovery_success = False
            if error_report.recovery_strategy.max_retries > 0:
                recovery_success = await self.recovery_manager.execute_recovery(
                    error_report, context
                )
            
            # Update error report with recovery result
            error_report.context.recovery_successful = recovery_success
            if recovery_success:
                error_report.resolution_status = "recovered"
                error_report.resolved_at = datetime.now()
            
            # Return error handling result
            return {
                "error_id": error_report.error_id,
                "handled": True,
                "recovery_attempted": error_report.recovery_strategy.max_retries > 0,
                "recovery_successful": recovery_success,
                "severity": error_report.context.severity.value,
                "category": error_report.context.category.value,
                "recommended_action": error_report.recovery_strategy.action.value
            }
            
        except Exception as e:
            logger.error(f"Error in error handling system: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "error": "Error handling failed",
                "handled": False,
                "recovery_successful": False
            }
    
    async def _store_error_report(self, error_report: ErrorReport):
        """Store error report in database."""
        try:
            # Convert to dictionary
            report_dict = asdict(error_report)
            
            # Convert datetime objects to strings
            report_dict["created_at"] = error_report.created_at.isoformat()
            if error_report.resolved_at:
                report_dict["resolved_at"] = error_report.resolved_at.isoformat()
            report_dict["context"]["timestamp"] = error_report.context.timestamp.isoformat()
            
            # Store in database
            self.db.errors.insert_one(report_dict)
            
        except Exception as e:
            logger.error(f"Error storing error report: {str(e)}")
    
    def get_error_statistics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time range."""
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        
        # Filter recent errors from history
        recent_errors = [
            error for error in self.error_history
            if error.created_at >= cutoff_time
        ]
        
        if not recent_errors:
            return {
                "total_errors": 0,
                "categories": {},
                "severities": {},
                "recovery_rate": 0.0
            }
        
        # Calculate statistics
        categories = {}
        severities = {}
        recovered_count = 0
        
        for error in recent_errors:
            cat = error.context.category.value
            sev = error.context.severity.value
            
            categories[cat] = categories.get(cat, 0) + 1
            severities[sev] = severities.get(sev, 0) + 1
            
            if error.context.recovery_successful:
                recovered_count += 1
        
        recovery_rate = recovered_count / len(recent_errors) if recent_errors else 0.0
        
        return {
            "total_errors": len(recent_errors),
            "categories": categories,
            "severities": severities,
            "recovery_rate": recovery_rate,
            "most_common_category": max(categories.items(), key=lambda x: x[1])[0] if categories else None,
            "most_common_severity": max(severities.items(), key=lambda x: x[1])[0] if severities else None
        }
    
    def get_job_error_history(self, job_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get error history for a specific job."""
        job_errors = [
            error for error in self.error_history
            if error.job_id == job_id
        ]
        
        # Convert to dictionaries
        error_dicts = []
        for error in job_errors[-limit:]:
            error_dict = asdict(error)
            error_dict["created_at"] = error.created_at.isoformat()
            if error.resolved_at:
                error_dict["resolved_at"] = error.resolved_at.isoformat()
            error_dict["context"]["timestamp"] = error.context.timestamp.isoformat()
            error_dicts.append(error_dict)
        
        return error_dicts

class ErrorPreventionSystem:
    """Proactive error prevention system."""
    
    def __init__(self, error_reporter: ErrorReporter):
        self.error_reporter = error_reporter
        self.health_checks = {}
        self.monitoring_active = False
        
    def register_health_check(self, name: str, check_function: Callable, 
                            interval: int = 60, critical: bool = False):
        """Register a health check function."""
        self.health_checks[name] = {
            "function": check_function,
            "interval": interval,
            "critical": critical,
            "last_run": None,
            "last_result": None
        }
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        current_time = datetime.now()
        
        for name, check_config in self.health_checks.items():
            try:
                # Check if it's time to run this check
                if (check_config["last_run"] is None or 
                    (current_time - check_config["last_run"]).total_seconds() >= check_config["interval"]):
                    
                    # Run the health check
                    result = await check_config["function"]()
                    
                    # Update check state
                    check_config["last_run"] = current_time
                    check_config["last_result"] = result
                    
                    results[name] = result
                    
                    # Handle failed critical checks
                    if check_config["critical"] and not result.get("healthy", False):
                        logger.error(f"Critical health check failed: {name}")
                        # Could trigger automatic recovery here
                        
            except Exception as e:
                logger.error(f"Health check {name} failed with error: {str(e)}")
                results[name] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": current_time.isoformat()
                }
        
        return results
    
    async def monitor_system_health(self, interval: int = 300):
        """Continuously monitor system health."""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                health_results = await self.run_health_checks()
                
                # Check for critical issues
                critical_failures = [
                    name for name, result in health_results.items()
                    if not result.get("healthy", False) and 
                    self.health_checks.get(name, {}).get("critical", False)
                ]
                
                if critical_failures:
                    logger.error(f"Critical system health failures: {critical_failures}")
                    # Could trigger emergency recovery procedures here
                
                # Wait before next check
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in system health monitoring: {str(e)}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop system health monitoring."""
        self.monitoring_active = False

# Global error handling instance
_global_error_handler = None

def initialize_error_handling(db=None) -> ErrorReporter:
    """Initialize global error handling system."""
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = ErrorReporter(db)
        logger.info("Global error handling system initialized")
    
    return _global_error_handler

def get_error_handler() -> ErrorReporter:
    """Get the global error handler."""
    if _global_error_handler is None:
        raise RuntimeError("Error handling system not initialized. Call initialize_error_handling() first.")
    
    return _global_error_handler

async def handle_error_async(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to handle errors asynchronously."""
    error_handler = get_error_handler()
    return await error_handler.handle_error(error, context)

def handle_error_sync(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to handle errors synchronously."""
    error_handler = get_error_handler()
    
    # Run async function in event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(error_handler.handle_error(error, context))