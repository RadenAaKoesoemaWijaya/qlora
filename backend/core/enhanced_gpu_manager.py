import torch
import psutil
import pynvml
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class GPUHealthMetrics:
    """GPU health metrics data class."""
    gpu_id: int
    temperature: float
    power_draw: float
    memory_utilization: float
    gpu_utilization: float
    fan_speed: Optional[int]
    health_score: float
    timestamp: datetime

class GPUHealthMonitor:
    """Advanced GPU health monitoring system."""
    
    def __init__(self, gpu_manager):
        self.gpu_manager = gpu_manager
        self.monitoring_active = False
        self.monitoring_thread = None
        self.health_history = {}  # Store historical health data
        self.health_thresholds = {
            "temperature": {"warning": 75, "critical": 85},
            "memory_utilization": {"warning": 90, "critical": 95},
            "gpu_utilization": {"warning": 95, "critical": 100},
            "power_draw": {"warning": 90, "critical": 95}  # Percentage of max power
        }
        
    def start_monitoring(self, interval_seconds: int = 30):
        """Start GPU health monitoring."""
        if self.monitoring_active:
            logger.warning("GPU monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"GPU health monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop GPU health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("GPU health monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_gpu_health()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {str(e)}")
                time.sleep(interval_seconds)
    
    def _check_gpu_health(self):
        """Check GPU health and store metrics."""
        if not self.gpu_manager.gpu_available:
            return
            
        try:
            gpu_status = self.gpu_manager.get_gpu_status()
            if not gpu_status["available"]:
                return
                
            for gpu in gpu_status["gpus"]:
                if "error" in gpu:
                    continue
                    
                gpu_id = gpu["id"]
                
                # Calculate health score (0-100)
                health_score = self._calculate_health_score(gpu)
                
                # Create health metrics
                health_metrics = GPUHealthMetrics(
                    gpu_id=gpu_id,
                    temperature=gpu["temperature_celsius"],
                    power_draw=gpu["power_draw_watts"],
                    memory_utilization=gpu["memory_utilization_percent"],
                    gpu_utilization=gpu["gpu_utilization_percent"],
                    fan_speed=self._get_fan_speed(gpu_id),
                    health_score=health_score,
                    timestamp=datetime.now()
                )
                
                # Store in history
                if gpu_id not in self.health_history:
                    self.health_history[gpu_id] = deque(maxlen=1000)
                self.health_history[gpu_id].append(health_metrics)
                
                # Log warnings for critical conditions
                self._log_health_warnings(health_metrics)
                
        except Exception as e:
            logger.error(f"Error checking GPU health: {str(e)}")
    
    def _calculate_health_score(self, gpu_info: Dict) -> float:
        """Calculate overall GPU health score (0-100)."""
        score = 100.0
        
        # Temperature penalty
        temp = gpu_info["temperature_celsius"]
        if temp > self.health_thresholds["temperature"]["critical"]:
            score -= 30
        elif temp > self.health_thresholds["temperature"]["warning"]:
            score -= 15
        
        # Memory utilization penalty
        mem_util = gpu_info["memory_utilization_percent"]
        if mem_util > self.health_thresholds["memory_utilization"]["critical"]:
            score -= 25
        elif mem_util > self.health_thresholds["memory_utilization"]["warning"]:
            score -= 10
        
        # GPU utilization penalty
        gpu_util = gpu_info["gpu_utilization_percent"]
        if gpu_util > self.health_thresholds["gpu_utilization"]["critical"]:
            score -= 20
        elif gpu_util > self.health_thresholds["gpu_utilization"]["warning"]:
            score -= 10
        
        return max(0.0, score)
    
    def _get_fan_speed(self, gpu_id: int) -> Optional[int]:
        """Get fan speed for GPU."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            fan_count = pynvml.nvmlDeviceGetNumFans(handle)
            if fan_count > 0:
                return pynvml.nvmlDeviceGetFanSpeed(handle, 0)
        except pynvml.NVMLError:
            pass
        return None
    
    def _log_health_warnings(self, metrics: GPUHealthMetrics):
        """Log health warnings for critical conditions."""
        if metrics.health_score < 50:
            logger.warning(f"GPU {metrics.gpu_id} health score critical: {metrics.health_score:.1f}")
        elif metrics.health_score < 75:
            logger.warning(f"GPU {metrics.gpu_id} health score warning: {metrics.health_score:.1f}")
    
    def get_health_summary(self, gpu_id: int = None) -> Dict:
        """Get health summary for specific GPU or all GPUs."""
        if not self.health_history:
            return {"error": "No health data available"}
        
        summary = {}
        
        if gpu_id is not None:
            if gpu_id in self.health_history:
                summary[gpu_id] = self._summarize_gpu_health(gpu_id)
        else:
            for gid in self.health_history.keys():
                summary[gid] = self._summarize_gpu_health(gid)
        
        return summary
    
    def _summarize_gpu_health(self, gpu_id: int) -> Dict:
        """Summarize health metrics for a GPU."""
        if gpu_id not in self.health_history or not self.health_history[gpu_id]:
            return {"error": "No health data available"}
        
        recent_metrics = list(self.health_history[gpu_id])[-100:]  # Last 100 readings
        
        temperatures = [m.temperature for m in recent_metrics]
        health_scores = [m.health_score for m in recent_metrics]
        
        return {
            "current_temperature": recent_metrics[-1].temperature,
            "avg_temperature": sum(temperatures) / len(temperatures),
            "max_temperature": max(temperatures),
            "current_health_score": recent_metrics[-1].health_score,
            "avg_health_score": sum(health_scores) / len(health_scores),
            "min_health_score": min(health_scores),
            "readings_count": len(recent_metrics),
            "health_status": self._get_health_status(recent_metrics[-1])
        }
    
    def _get_health_status(self, metrics: GPUHealthMetrics) -> str:
        """Get health status string."""
        if metrics.health_score >= 90:
            return "excellent"
        elif metrics.health_score >= 75:
            return "good"
        elif metrics.health_score >= 50:
            return "warning"
        else:
            return "critical"

class GPUResourceAllocator:
    """Advanced GPU resource allocation system."""
    
    def __init__(self, gpu_manager):
        self.gpu_manager = gpu_manager
        self.allocated_resources = {}  # Track allocated resources
        self.allocation_history = deque(maxlen=1000)
        
    def allocate_gpu_for_training(self, job_id: str, model_size: str, 
                                 batch_size: int, sequence_length: int,
                                 multi_gpu: bool = False) -> Dict:
        """
        Allocate GPU resources for training job.
        
        Args:
            job_id: Training job ID
            model_size: Model size (e.g., "7B", "13B")
            batch_size: Training batch size
            sequence_length: Maximum sequence length
            multi_gpu: Whether to use multiple GPUs
            
        Returns:
            Allocation result with GPU IDs and resource information
        """
        try:
            # Check memory requirements
            memory_check = self.gpu_manager.check_memory_requirements(
                model_size, batch_size, sequence_length
            )
            
            if not memory_check["sufficient"]:
                return {
                    "success": False,
                    "error": f"Insufficient memory: {memory_check['message']}",
                    "required_gb": memory_check["required_gb"],
                    "available_gb": memory_check["available_gb"]
                }
            
            # Get current GPU status
            gpu_status = self.gpu_manager.get_gpu_status()
            if not gpu_status["available"]:
                return {
                    "success": False,
                    "error": "No GPU available",
                    "required_gb": memory_check["required_gb"],
                    "available_gb": 0
                }
            
            # Filter available GPUs
            available_gpus = []
            for gpu in gpu_status["gpus"]:
                if "error" in gpu:
                    continue
                
                # Check if GPU is already allocated
                if gpu["id"] in self.allocated_resources:
                    continue
                
                # Check if GPU has sufficient memory
                free_memory_gb = gpu["memory_free_mb"] / 1024
                if free_memory_gb >= memory_check["required_gb"]:
                    # Calculate allocation score (higher is better)
                    score = self._calculate_allocation_score(gpu)
                    available_gpus.append({
                        "gpu_id": gpu["id"],
                        "score": score,
                        "free_memory_gb": free_memory_gb,
                        "gpu_info": gpu
                    })
            
            if not available_gpus:
                return {
                    "success": False,
                    "error": "No suitable GPU available",
                    "required_gb": memory_check["required_gb"],
                    "available_gb": memory_check["available_gb"]
                }
            
            # Sort by score (descending)
            available_gpus.sort(key=lambda x: x["score"], reverse=True)
            
            # Select GPUs
            selected_gpus = []
            if multi_gpu:
                # Select multiple GPUs for distributed training
                num_gpus_needed = min(2, len(available_gpus))  # Max 2 GPUs for now
                selected_gpus = [gpu["gpu_id"] for gpu in available_gpus[:num_gpus_needed]]
            else:
                # Select single best GPU
                selected_gpus = [available_gpus[0]["gpu_id"]]
            
            # Record allocation
            allocation_info = {
                "job_id": job_id,
                "gpus": selected_gpus,
                "model_size": model_size,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "allocated_at": datetime.now().isoformat(),
                "memory_required_gb": memory_check["required_gb"],
                "multi_gpu": multi_gpu
            }
            
            for gpu_id in selected_gpus:
                self.allocated_resources[gpu_id] = allocation_info
            
            self.allocation_history.append(allocation_info)
            
            return {
                "success": True,
                "allocation": allocation_info,
                "selected_gpus": selected_gpus,
                "memory_check": memory_check
            }
            
        except Exception as e:
            logger.error(f"Error allocating GPU resources: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "required_gb": 0,
                "available_gb": 0
            }
    
    def _calculate_allocation_score(self, gpu_info: Dict) -> float:
        """Calculate allocation score for GPU (higher is better)."""
        score = 0.0
        
        # Memory availability (most important)
        free_memory_gb = gpu_info["memory_free_mb"] / 1024
        score += free_memory_gb * 10  # 10 points per GB
        
        # Lower utilization is better
        gpu_utilization = gpu_info["gpu_utilization_percent"]
        score += (100 - gpu_utilization) * 0.5  # 0.5 points per % of free utilization
        
        # Lower temperature is better
        temperature = gpu_info["temperature_celsius"]
        if temperature < 60:
            score += 20
        elif temperature < 70:
            score += 10
        elif temperature < 80:
            score += 5
        
        # Power efficiency
        power_draw = gpu_info["power_draw_watts"]
        # Prefer GPUs with reasonable power consumption
        if power_draw < 200:
            score += 10
        elif power_draw < 300:
            score += 5
        
        return score
    
    def release_gpu_resources(self, job_id: str) -> Dict:
        """Release GPU resources allocated to a job."""
        released_gpus = []
        
        # Find and release allocated GPUs
        gpus_to_release = []
        for gpu_id, allocation in self.allocated_resources.items():
            if allocation["job_id"] == job_id:
                gpus_to_release.append(gpu_id)
        
        # Release GPUs
        for gpu_id in gpus_to_release:
            del self.allocated_resources[gpu_id]
            released_gpus.append(gpu_id)
        
        return {
            "success": len(released_gpus) > 0,
            "released_gpus": released_gpus,
            "job_id": job_id
        }
    
    def get_allocation_status(self) -> Dict:
        """Get current allocation status."""
        return {
            "allocated_gpus": len(self.allocated_resources),
            "allocated_resources": self.allocated_resources,
            "free_gpus": self.gpu_manager.gpu_count - len(self.allocated_resources),
            "total_gpus": self.gpu_manager.gpu_count
        }
    
    def get_allocation_history(self, limit: int = 100) -> List[Dict]:
        """Get allocation history."""
        return list(self.allocation_history)[-limit:]

# Enhanced GPU Manager with health monitoring and resource allocation
class EnhancedGPUManager:
    """Enhanced GPU manager with comprehensive monitoring and resource management."""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.nvml_initialized = False
        
        # Initialize NVML
        try:
            if self.gpu_available:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                logger.info(f"Initialized NVML with {self.gpu_count} GPU(s)")
        except pynvml.NVMLError as e:
            logger.warning(f"Failed to initialize NVML: {str(e)}")
            self.nvml_initialized = False
        
        # Initialize health monitor and resource allocator
        self.health_monitor = GPUHealthMonitor(self)
        self.resource_allocator = GPUResourceAllocator(self)
        
        # Start health monitoring
        if self.gpu_available:
            self.health_monitor.start_monitoring()
    
    def get_gpu_status(self) -> Dict:
        """Get comprehensive GPU status information."""
        if not self.gpu_available:
            return {
                "available": False,
                "message": "No GPU available",
                "cuda_version": None,
                "gpu_count": 0,
                "gpus": []
            }
        
        try:
            gpus = []
            
            for i in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Get device info
                    name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                    
                    # Get additional info
                    compute_capability = torch.cuda.get_device_capability(i)
                    driver_version = pynvml.nvmlSystemGetDriverVersion().decode("utf-8")
                    
                    # Get PCI info
                    try:
                        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                        pci_bus_id = pci_info.busId.decode("utf-8")
                    except:
                        pci_bus_id = "unknown"
                    
                    gpu_info = {
                        "id": i,
                        "name": name,
                        "pci_bus_id": pci_bus_id,
                        "memory_total_mb": memory_info.total // (1024 * 1024),
                        "memory_used_mb": memory_info.used // (1024 * 1024),
                        "memory_free_mb": memory_info.free // (1024 * 1024),
                        "memory_utilization_percent": utilization.memory,
                        "gpu_utilization_percent": utilization.gpu,
                        "temperature_celsius": temperature,
                        "power_draw_watts": power,
                        "cuda_compute_capability": compute_capability,
                        "driver_version": driver_version,
                        "health_score": None,  # Will be filled by health monitor
                        "allocation_status": "free" if i not in self.resource_allocator.allocated_resources else "allocated"
                    }
                    
                    gpus.append(gpu_info)
                    
                except pynvml.NVMLError as e:
                    logger.error(f"Error getting GPU {i} info: {str(e)}")
                    gpus.append({
                        "id": i,
                        "error": str(e),
                        "available": False
                    })
            
            return {
                "available": True,
                "cuda_version": torch.version.cuda,
                "gpu_count": self.gpu_count,
                "gpus": gpus,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting GPU status: {str(e)}")
            return {
                "available": False,
                "error": str(e),
                "gpus": []
            }
    
    def check_memory_requirements(self, model_size: str, batch_size: int = 2, 
                                 sequence_length: int = 512) -> Dict:
        """
        Check if GPU memory is sufficient for training.
        
        Args:
            model_size: Model size (e.g., "7B", "13B")
            batch_size: Training batch size
            sequence_length: Maximum sequence length
            
        Returns:
            Dictionary with memory analysis
        """
        if not self.gpu_available:
            return {
                "sufficient": False,
                "message": "No GPU available",
                "required_gb": 0,
                "available_gb": 0
            }
        
        try:
            # Enhanced memory estimation for QLoRA training
            base_model_memory = {
                "3B": 8,
                "7B": 14,
                "8B": 16,
                "13B": 26,
                "30B": 60,
                "40B": 80,
                "70B": 140,
                "47B": 94  # Mixtral 8x7B
            }
            
            # Get base model memory requirement
            base_memory = base_model_memory.get(model_size, 0)
            if base_memory == 0:
                return {
                    "sufficient": False,
                    "message": f"Unknown model size: {model_size}",
                    "required_gb": 0,
                    "available_gb": 0
                }
            
            # Calculate memory components for QLoRA training
            # QLoRA reduces memory by ~75% through 4-bit quantization
            quantized_model_memory = base_memory * 0.25
            
            # Activations memory (rough estimation based on batch size and sequence length)
            activation_memory = (batch_size * sequence_length * 4096 * 4) / (1024**3)  # 4 bytes per float
            
            # Optimizer states (AdamW: 2x model parameters for full precision)
            # QLoRA only updates LoRA parameters, so much less memory
            optimizer_memory = quantized_model_memory * 0.1  # ~10% of model memory
            
            # Gradients (only for LoRA parameters)
            gradient_memory = quantized_model_memory * 0.05  # ~5% of model memory
            
            # Additional overhead for CUDA context, temporary buffers, etc.
            overhead_memory = 2  # GB
            
            # Total required memory
            total_required = quantized_model_memory + activation_memory + optimizer_memory + gradient_memory + overhead_memory
            
            # Get available memory considering current allocations
            gpu_status = self.get_gpu_status()
            if gpu_status["available"] and gpu_status["gpus"]:
                # Find GPU with most free memory that's not allocated
                max_free_memory = 0
                for gpu in gpu_status["gpus"]:
                    if "error" in gpu or gpu["allocation_status"] == "allocated":
                        continue
                    max_free_memory = max(max_free_memory, gpu["memory_free_mb"])
                available_gb = max_free_memory / 1024 if max_free_memory > 0 else 0
            else:
                available_gb = 0
            
            # Calculate safety margin (25% for QLoRA)
            required_with_margin = total_required * 1.25
            
            return {
                "sufficient": available_gb >= required_with_margin,
                "required_gb": required_with_margin,
                "available_gb": available_gb,
                "breakdown": {
                    "model_memory_gb": quantized_model_memory,
                    "activation_memory_gb": activation_memory,
                    "optimizer_memory_gb": optimizer_memory,
                    "gradient_memory_gb": gradient_memory,
                    "overhead_gb": overhead_memory
                },
                "model_size": model_size,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "safety_margin_percent": 25,
                "memory_efficiency": "qlora_4bit"
            }
            
        except Exception as e:
            logger.error(f"Error checking memory requirements: {str(e)}")
            return {
                "sufficient": False,
                "error": str(e),
                "required_gb": 0,
                "available_gb": 0
            }
    
    def get_system_resources(self) -> Dict:
        """Get comprehensive system resource information."""
        try:
            # CPU info
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory info
            memory = psutil.virtual_memory()
            memory_info = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent,
                "cached_gb": getattr(memory, 'cached', 0) / (1024**3) if hasattr(memory, 'cached') else 0
            }
            
            # Disk info
            disk = psutil.disk_usage('/')
            disk_info = {
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "used_gb": disk.used / (1024**3),
                "percent": (disk.used / disk.total) * 100
            }
            
            # GPU info
            gpu_info = self.get_gpu_status()
            
            # Health summary
            health_summary = self.health_monitor.get_health_summary()
            
            # Allocation status
            allocation_status = self.resource_allocator.get_allocation_status()
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None,
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else []
                },
                "memory": memory_info,
                "disk": disk_info,
                "gpu": gpu_info,
                "gpu_health": health_summary,
                "gpu_allocation": allocation_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system resources: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_gpu_recommendations(self, model_size: str, batch_size: int = 2, 
                               sequence_length: int = 512) -> Dict:
        """
        Get GPU recommendations for specific training requirements.
        
        Args:
            model_size: Model size (e.g., "7B", "13B")
            batch_size: Training batch size
            sequence_length: Maximum sequence length
            
        Returns:
            GPU recommendations with detailed analysis
        """
        try:
            # Check memory requirements
            memory_check = self.check_memory_requirements(model_size, batch_size, sequence_length)
            
            if not memory_check["sufficient"]:
                return {
                    "recommendation": "insufficient_resources",
                    "message": "Insufficient GPU memory for training requirements",
                    "memory_check": memory_check,
                    "suggested_changes": {
                        "reduce_batch_size": max(1, batch_size // 2),
                        "reduce_sequence_length": max(256, sequence_length // 2),
                        "consider_multi_gpu": True
                    }
                }
            
            # Get available GPUs
            gpu_status = self.get_gpu_status()
            if not gpu_status["available"]:
                return {
                    "recommendation": "no_gpu_available",
                    "message": "No GPU available",
                    "memory_check": memory_check
                }
            
            # Analyze available GPUs
            suitable_gpus = []
            for gpu in gpu_status["gpus"]:
                if "error" in gpu or gpu["allocation_status"] == "allocated":
                    continue
                
                # Calculate suitability score
                score = self._calculate_suitability_score(gpu, model_size)
                
                suitable_gpus.append({
                    "gpu_id": gpu["id"],
                    "name": gpu["name"],
                    "score": score,
                    "memory_free_gb": gpu["memory_free_mb"] / 1024,
                    "temperature": gpu["temperature_celsius"],
                    "utilization": gpu["gpu_utilization_percent"]
                })
            
            if not suitable_gpus:
                return {
                    "recommendation": "no_suitable_gpu",
                    "message": "No suitable GPU available for training",
                    "memory_check": memory_check
                }
            
            # Sort by score (descending)
            suitable_gpus.sort(key=lambda x: x["score"], reverse=True)
            
            # Get health status for recommended GPUs
            health_summary = self.health_monitor.get_health_summary()
            
            return {
                "recommendation": "suitable_gpu_found",
                "message": "Suitable GPU(s) found for training",
                "memory_check": memory_check,
                "recommended_gpus": suitable_gpus[:3],  # Top 3 recommendations
                "health_summary": {gpu["gpu_id"]: health_summary.get(gpu["gpu_id"], {}) 
                                 for gpu in suitable_gpus[:3]}
            }
            
        except Exception as e:
            logger.error(f"Error getting GPU recommendations: {str(e)}")
            return {
                "recommendation": "error",
                "message": f"Error analyzing GPU requirements: {str(e)}",
                "error": str(e)
            }
    
    def _calculate_suitability_score(self, gpu_info: Dict, model_size: str) -> float:
        """Calculate GPU suitability score for training."""
        score = 0.0
        
        # Memory availability (most important)
        free_memory_gb = gpu_info["memory_free_mb"] / 1024
        score += free_memory_gb * 10  # 10 points per GB
        
        # Temperature (lower is better)
        temperature = gpu_info["temperature_celsius"]
        if temperature < 60:
            score += 25
        elif temperature < 70:
            score += 15
        elif temperature < 80:
            score += 5
        else:
            score -= 10  # Penalty for high temperature
        
        # Current utilization (lower is better)
        gpu_utilization = gpu_info["gpu_utilization_percent"]
        score += (100 - gpu_utilization) * 0.3  # 0.3 points per % of free utilization
        
        # GPU generation/model preference (newer/better GPUs get higher score)
        gpu_name = gpu_info["name"].lower()
        if "h100" in gpu_name or "a100" in gpu_name:
            score += 50  # Premium GPUs
        elif "v100" in gpu_name or "rtx 4090" in gpu_name or "rtx 4080" in gpu_name:
            score += 30  # High-end GPUs
        elif "rtx 3090" in gpu_name or "rtx 3080" in gpu_name:
            score += 20  # Good GPUs
        elif "rtx 3070" in gpu_name or "rtx 3060" in gpu_name:
            score += 10  # Decent GPUs
        
        return score
    
    def cleanup(self):
        """Cleanup resources."""
        # Stop health monitoring
        if hasattr(self, 'health_monitor'):
            self.health_monitor.stop_monitoring()
        
        # Cleanup NVML
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVML shutdown completed")
            except pynvml.NVMLError as e:
                logger.error(f"Error during NVML shutdown: {str(e)}")

# Singleton instance
enhanced_gpu_manager = EnhancedGPUManager()