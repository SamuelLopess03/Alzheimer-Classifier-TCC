import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Dict, Any, cast, Optional

from .builder_registry import ModelBuilder, model_registry
from .layer_utils import get_module_by_name, set_module_by_name

class CNNBuilder(ModelBuilder):
    def build_base(self, architecture_name: str, config: Dict[str, Any]) -> nn.Module:
        arch_lower = architecture_name.lower()

        if 'resnext50' in arch_lower:
            return models.resnext50_32x4d(weights=config.get('weights', 'IMAGENET1K_V2'))
        elif 'convnext_tiny' in arch_lower:
            return models.convnext_tiny(weights=config.get('weights', 'IMAGENET1K_V1'))
        elif 'efficientnet_v2_s' in arch_lower:
            return models.efficientnet_v2_s(weights=config.get('weights', 'IMAGENET1K_V1'))
        elif 'densenet121' in arch_lower:
            return models.densenet121(weights=config.get('weights', 'IMAGENET1K_V1'))

        raise ValueError(f"Arquitetura CNN não suportada: {architecture_name}")

    def adapt_grayscale(self, model: nn.Module, arch_config: Dict[str, Any], preserve_weights: bool) -> nn.Module:
        layer_path = arch_config.get('adaptation_layer')
        original_conv = cast(nn.Conv2d, get_module_by_name(model, layer_path))
        
        if not original_conv:
            raise ValueError(f"Camada de adaptação '{layer_path}' não encontrada no modelo.")

        new_conv = nn.Conv2d(
            1, 
            original_conv.out_channels, 
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=original_conv.bias is not None
        )
        
        if preserve_weights:
            with torch.no_grad():
                new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
                if original_conv.bias is not None:
                    new_conv.bias = original_conv.bias
        
        set_module_by_name(model, layer_path, new_conv)

        return model

    def replace_head(self, model: nn.Module, arch_config: Dict[str, Any], in_features: int, hidden_units: int, dropout: float, num_classes: int) -> nn.Module:
        layer_path = arch_config.get('classifier_layer')
        
        head = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes)
        )
        
        if 'convnext' in arch_config['type'].lower() or 'convnext' in layer_path:
            original_classifier = get_module_by_name(model, layer_path)
            layernorm = original_classifier[0]
            flatten = original_classifier[1]
            head = nn.Sequential(
                layernorm, 
                flatten, 
                nn.Linear(in_features, hidden_units), 
                nn.ReLU(True), 
                nn.Dropout(dropout), 
                nn.Linear(hidden_units, num_classes)
            )
        elif 'efficientnet' in arch_config['type'].lower() or 'efficientnet' in layer_path:
            head = nn.Sequential(
                nn.Dropout(dropout), 
                nn.Linear(in_features, hidden_units), 
                nn.ReLU(True), 
                nn.Dropout(dropout), 
                nn.Linear(hidden_units, num_classes)
            )

        set_module_by_name(model, layer_path, head)
        return model

    def get_in_features(self, model: nn.Module, arch_config: Dict[str, Any]) -> int:
        layer_path = arch_config.get('classifier_layer')
        classifier = get_module_by_name(model, layer_path)
        
        if isinstance(classifier, nn.Sequential):
            for layer in classifier:
                if hasattr(layer, 'in_features'):
                    return int(layer.in_features)
        
        if hasattr(classifier, 'in_features'):
            return int(classifier.in_features)
            
        raise AttributeError(f"Não foi possível determinar in_features para '{layer_path}'")

    def verify_grayscale(self, model: nn.Module, arch_config: Dict[str, Any]) -> bool:
        layer_path = arch_config.get('adaptation_layer')
        conv = cast(nn.Conv2d, get_module_by_name(model, layer_path))
        return conv.in_channels == 1

class TransformerBuilder(ModelBuilder):
    def build_base(self, architecture_name: str, config: Dict[str, Any]) -> nn.Module:
        return timm.create_model(
            config['timm_model'],
            pretrained=config.get('pretrained', True),
            num_classes=0 
        )

    def adapt_grayscale(self, model: nn.Module, arch_config: Dict[str, Any], preserve_weights: bool) -> nn.Module:
        layer_path = arch_config.get('adaptation_layer')
        original_conv = cast(nn.Conv2d, get_module_by_name(model, layer_path))
        
        if not original_conv:
            raise ValueError(f"Camada de adaptação '{layer_path}' não encontrada no modelo.")

        new_conv = nn.Conv2d(
            1, 
            original_conv.out_channels, 
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=original_conv.bias is not None
        )
        
        if preserve_weights:
            with torch.no_grad():
                new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
                if original_conv.bias is not None:
                    new_conv.bias = original_conv.bias
        
        set_module_by_name(model, layer_path, new_conv)
        return model

    def replace_head(self, model: nn.Module, arch_config: Dict[str, Any], in_features: int, hidden_units: int, dropout: float, num_classes: int) -> nn.Module:
        layer_path = arch_config.get('classifier_layer')
        head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes)
        )
        set_module_by_name(model, layer_path, head)
        return model

    def get_in_features(self, model: nn.Module, arch_config: Dict[str, Any]) -> int:
        if hasattr(model, 'num_features'):
            return int(model.num_features)
        
        layer_path = arch_config.get('classifier_layer')
        classifier = get_module_by_name(model, layer_path)
        return int(classifier.in_features)

    def verify_grayscale(self, model: nn.Module, arch_config: Dict[str, Any]) -> bool:
        layer_path = arch_config.get('adaptation_layer')
        conv = cast(nn.Conv2d, get_module_by_name(model, layer_path))
        return conv.in_channels == 1

cnn_builder = CNNBuilder()
transformer_builder = TransformerBuilder()

model_registry.register_builder('cnn', cnn_builder, ['resnext50_32x4d', 'convnext_tiny', 'efficientnetv2_s', 'densenet121'])
model_registry.register_builder('transformer', transformer_builder, ['vit_b_16', 'swin_v2_tiny'])
