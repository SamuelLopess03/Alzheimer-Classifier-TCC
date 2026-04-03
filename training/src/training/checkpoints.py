import torch
import torch.nn as nn
from typing import Dict

def save_best_checkpoint(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer,
    epoch: int, 
    val_f1: float, 
    metrics: Dict,
    config: Dict, 
    checkpoint_file: str,
    architecture_name: str, 
    hyperparameters: Dict
):
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
        'metrics': metrics,
        'config': config,
        'architecture_name': architecture_name,
        'hyperparameters': hyperparameters,
        'class_names': config['model']['class_names'],
        'num_classes': config['model']['num_classes']
    }
    
    torch.save(checkpoint_data, checkpoint_file)
