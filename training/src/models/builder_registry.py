import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Type
from abc import ABC, abstractmethod

class ModelBuilder(ABC):
    @abstractmethod
    def build_base(self, architecture_name: str, config: Dict[str, Any]) -> nn.Module:
        pass

    @abstractmethod
    def adapt_grayscale(self, model: nn.Module, arch_config: Dict[str, Any], preserve_weights: bool) -> nn.Module:
        pass

    @abstractmethod
    def replace_head(self, model: nn.Module, arch_config: Dict[str, Any], in_features: int, hidden_units: int, dropout: float, num_classes: int) -> nn.Module:
        pass

    @abstractmethod
    def get_in_features(self, model: nn.Module, arch_config: Dict[str, Any]) -> int:
        pass

    @abstractmethod
    def verify_grayscale(self, model: nn.Module, arch_config: Dict[str, Any]) -> bool:
        pass

class Registry:  
    def __init__(self):
        self._builders: Dict[str, ModelBuilder] = {}
        self._name_to_family: Dict[str, str] = {}

    def register_builder(self, family_name: str, builder: ModelBuilder, architectures: List[str]):
        self._builders[family_name.lower()] = builder
        for arch in architectures:
            self._name_to_family[arch.lower()] = family_name.lower()

    def get_builder(self, architecture_name: str) -> Optional[ModelBuilder]:
        family = self._name_to_family.get(architecture_name.lower())
        if family:
            return self._builders.get(family)
        return None

    def get_all_supported_architectures(self) -> List[str]:
        return list(self._name_to_family.keys())

model_registry = Registry()
