import yaml
import os
from typing import Dict, Any
from pathlib import Path

# Caminho absoluto para a pasta configs
CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs"

def load_yaml(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config

def load_config(config_name: str, config_dir: str = CONFIGS_PATH) -> Dict[str, Any]:
    config_path = os.path.join(config_dir, f"{config_name}.yaml")
    return load_yaml(config_path)

def load_binary_config(config_dir: str = CONFIGS_PATH) -> Dict[str, Any]:
    return load_config("config_binary", config_dir)

def load_multiclass_config(config_dir: str = CONFIGS_PATH) -> Dict[str, Any]:
    return load_config("config_multiclass", config_dir)

def load_augmentation_config(config_dir: str = CONFIGS_PATH) -> Dict[str, Any]:
    return load_config("augmentation", config_dir)

def load_hyperparameters_config(config_dir: str = CONFIGS_PATH) -> Dict[str, Any]:
    return load_config("hyperparameters", config_dir)