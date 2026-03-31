import torch
import torch.nn as nn
from typing import Any, cast

def get_target_layer(model: nn.Module, architecture_name: str) -> nn.Module:
    arch_lower = architecture_name.lower()

    if 'resnext' in arch_lower:
        return cast(Any, model.layer4[-1])

    elif 'efficientnet' in arch_lower:
        return cast(Any, model.features[-1])

    elif 'densenet' in arch_lower:
        return model.features.denseblock4

    elif 'vit' in arch_lower:
        block = cast(Any, model.blocks[-1])
        return block.norm1

    elif 'swin' in arch_lower:
        layer = cast(Any, model.layers[-1])
        block = layer.blocks[-1]
        return block.norm1

    elif 'convnext' in arch_lower:
        return cast(Any, model.stages[-1])

    else:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module

        return list(model.modules())[-1]
