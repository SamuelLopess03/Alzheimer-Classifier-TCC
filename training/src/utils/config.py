import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from .hardware import apply_hardware_overrides

TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_PATH = TRAINING_ROOT / "configs"

SHARED_PATH = TRAINING_ROOT.parent / "shared"

LOGS_PATH = SHARED_PATH / "logs"
MODELS_PATH = SHARED_PATH / "models"

def load_yaml(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config

def load_config(config_name: str, config_dir: str = CONFIGS_PATH, env: Optional[str] = None) -> Dict[str, Any]:
    config_path = os.path.join(config_dir, f"{config_name}.yaml")
    config = load_yaml(config_path)
    
    if env:
        return apply_hardware_overrides(config, config_name, env)
    
    return config

def load_binary_config() -> Dict[str, Any]:
    return load_config("config_binary")

def load_multiclass_config() -> Dict[str, Any]:
    return load_config("config_multiclass")

def load_augmentation_config() -> Dict[str, Any]:
    return load_config("augmentation")

def load_hyperparameters_config(env: Optional[str] = None) -> Dict[str, Any]:
    return load_config("hyperparameters", env=env)
