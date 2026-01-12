import torch
import psutil
import pynvml
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class GPUManager:
    """
    GPU resource manager untuk monitoring dan allocation resources.
    """
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.nvml_initialized = False
        
        try:
            if self.gpu_available:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                logger.info(f"Initialized NVML with {self.gpu_count} GPU(s)")
        except pynvml.NVMLError as e:
            logger.warning(f"Failed to initialize NVML: {str(e)}")
            self.nvml_initialized = False
    
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
                    
                    gpu_info = {
                        "id": i,
                        "name": name,
                        "memory_total_mb": memory_info.total // (1024 * 1024),
                        "memory_used_mb": memory_info.used // (1024 * 1024),
                        "memory_free_mb": memory_info.free // (1024 * 1024),
                        "memory_utilization_percent": utilization.memory,
                        "gpu_utilization_percent": utilization.gpu,
                        "temperature_celsius": temperature,
                        "power_draw_watts": power,
                        "cuda_compute_capability": torch.cuda.get_device_capability(i),
                        "driver_version": pynvml.nvmlSystemGetDriverVersion().decode("utf-8")
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
            model_size: Model size (e.g., "7B", "13B", "30B")
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
            # Rough estimation for QLoRA training memory requirements (in GB)
            # Based on: model params + activations + optimizer states + gradients
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
            
            # Calculate additional memory for training components
            # QLoRA reduces memory by ~75% through quantization
            quantized_model_memory = base_memory * 0.25
            
            # Activations memory (rough estimation)
            activation_memory = (batch_size * sequence_length * 4096 * 4) / (1024**3)  # 4 bytes per float
            
            # Optimizer states (AdamW: 2x model parameters)
            optimizer_memory = quantized_model_memory * 2
            
            # Gradients
            gradient_memory = quantized_model_memory
            
            # Additional overhead
            overhead_memory = 2  # GB
            
            # Total required memory
            total_required = quantized_model_memory + activation_memory + optimizer_memory + gradient_memory + overhead_memory
            
            # Get available memory from GPU with most free space
            gpu_status = self.get_gpu_status()
            if gpu_status["available"] and gpu_status["gpus"]:
                max_free_memory = max(gpu["memory_free_mb"] for gpu in gpu_status["gpus"])
                available_gb = max_free_memory / 1024
            else:
                available_gb = 0
            
            # Calculate safety margin (20%)
            required_with_margin = total_required * 1.2
            
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
                "safety_margin_percent": 20
            }
            
        except Exception as e:
            logger.error(f"Error checking memory requirements: {str(e)}")
            return {
                "sufficient": False,
                "error": str(e),
                "required_gb": 0,
                "available_gb": 0
            }
    
    def select_optimal_gpu(self, required_memory_gb: float) -> Optional[int]:
        """
        Select GPU with most available memory.
        
        Args:
            required_memory_gb: Required memory in GB
            
        Returns:
            GPU device ID or None if no suitable GPU found
        """
        if not self.gpu_available:
            return None
        
        try:
            gpu_status = self.get_gpu_status()
            if not gpu_status["available"]:
                return None
            
            best_gpu = None
            max_free_memory = 0
            
            for gpu in gpu_status["gpus"]:
                if "memory_free_mb" in gpu and gpu["memory_free_mb"] > max_free_memory:
                    free_gb = gpu["memory_free_mb"] / 1024
                    if free_gb >= required_memory_gb:
                        max_free_memory = gpu["memory_free_mb"]
                        best_gpu = gpu["id"]
            
            return best_gpu
            
        except Exception as e:
            logger.error(f"Error selecting optimal GPU: {str(e)}")
            return None
    
    def get_system_resources(self) -> Dict:
        """Get comprehensive system resource information."""
        try:
            # CPU info
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory info
            memory = psutil.virtual_memory()
            memory_info = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent
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
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else []
                },
                "memory": memory_info,
                "disk": disk_info,
                "gpu": gpu_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system resources: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def cleanup(self):
        """Cleanup NVML resources."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVML shutdown completed")
            except pynvml.NVMLError as e:
                logger.error(f"Error during NVML shutdown: {str(e)}")

# Singleton instance
gpu_manager = GPUManager()