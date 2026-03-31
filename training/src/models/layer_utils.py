import torch
import torch.nn as nn
from typing import Any, cast, Optional

def get_module_by_name(model: nn.Module, layer_name: str) -> Optional[nn.Module]:
    if not layer_name:
        return None
        
    tokens = layer_name.replace('[', '.').replace(']', '').split('.')
    curr = model
    try:
        for token in tokens:
            if not token: continue
            if token.isdigit():
                curr = curr[int(token)]
            else:
                curr = getattr(curr, token)
        return curr
    except (AttributeError, KeyError, IndexError, TypeError):
        return None

def set_module_by_name(model: nn.Module, layer_name: str, new_module: nn.Module):
    if not layer_name:
        return
        
    tokens = layer_name.replace('[', '.').replace(']', '').split('.')

    parent = model
    for token in tokens[:-1]:
        if not token: continue
        if token.isdigit():
            parent = parent[int(token)]
        else:
            parent = getattr(parent, token)
            
    last_token = tokens[-1]
    if last_token.isdigit():
        parent[int(last_token)] = new_module
    else:
        setattr(parent, last_token, new_module)

def get_target_layer(model: nn.Module, target_layer_path: Optional[str] = None) -> nn.Module:
    if target_layer_path:
        target_layer = get_module_by_name(model, target_layer_path)
        if target_layer:
            return target_layer
            
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module

    return list(model.modules())[-1]
