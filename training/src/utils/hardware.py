import platform
import psutil
import torch
import copy
from typing import Dict, Any, Optional, Set

_LOGGED_ENVS: Set[str] = set()

def detect_environment() -> str:
    try:
        total_ram = psutil.virtual_memory().total / (1024**3) # em GB
        if total_ram > 64:
            return "SERVER_RTX3090"
    except Exception:
        pass
    
    system = platform.system()
    if system == "Windows":
        return "LOCAL_RTX3060"
        
    return "LOCAL_RTX3060" # Default para máquinas menores

def get_pytorch_device() -> torch.device:
    if torch.cuda.is_available():
        env = detect_environment()
        device = torch.device('cuda')
        print(f"\n[DEVICE] Usando GPU: {torch.cuda.get_device_name(0)} (Ambiente: {env})")
        return device
    
    print("\n[DEVICE] CUDA não disponível. Usando CPU.")
    return torch.device('cpu')

def _get_hardware_overrides(env: str) -> Dict[str, Any]:
    if env == "SERVER_RTX3090":
        return {
            "num_workers": 16,
            "pin_memory": True,
            "mixed_precision": True,
            "device": "cuda",
            "persistent_workers": True,   
            "prefetch_factor": 4,         
            "cudnn_benchmark": True, 
            "batch_size_cnn": [64, 128, 256],
            "batch_size_vit": [32, 64, 128],
            "batch_size_swin": [32, 64, 128]
        }
    elif env == "LOCAL_RTX3060":
        return {
            "num_workers": 2,
            "pin_memory": True,
            "mixed_precision": True,
            "device": "cuda",
            "persistent_workers": False,  
            "prefetch_factor": 2,         
            "cudnn_benchmark": False,
            "batch_size_cnn": [16, 32],
            "batch_size_vit": [8, 16],
            "batch_size_swin": [8, 16]
        }
    else:
        return {
            "num_workers": 2,
            "pin_memory": False,
            "mixed_precision": False,
            "device": "cpu",
            "persistent_workers": False,
            "prefetch_factor": 2,
            "cudnn_benchmark": False,
            "batch_size_cnn": [16, 32],
            "batch_size_vit": [16],
            "batch_size_swin": [16]
        }

def apply_hardware_overrides(config_dict: Dict[str, Any], config_name: str, env: str) -> Dict[str, Any]:
    config = copy.deepcopy(config_dict)
    overrides = _get_hardware_overrides(env)

    global _LOGGED_ENVS

    if config_name == "hyperparameters":
        if "hardware" not in config:
            config["hardware"] = {}
            
        for key in ["num_workers", "pin_memory", "mixed_precision", "device", 
                    "persistent_workers", "prefetch_factor", "cudnn_benchmark"]:
            config["hardware"][key] = overrides[key]
            
        if env not in _LOGGED_ENVS:
            print(f"\n[HW] Ambiente: {env} | workers={overrides['num_workers']} | "
                  f"mixed_precision={overrides['mixed_precision']}\n")
            _LOGGED_ENVS.add(env)

        if "hyperparameter_grids" in config:
            grids = config["hyperparameter_grids"]
            for arch in ["cnn", "vit", "swin"]:
                if arch in grids:
                    grids[arch]["batch_size"] = overrides[f"batch_size_{arch}"]

    return config

if __name__ == "__main__":
    env = detect_environment()
    print(f"Detected Environment: {env}")
    print(f"Device: {get_pytorch_device()}")
