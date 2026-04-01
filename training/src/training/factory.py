import torch
import numpy as np
import itertools
from torch.optim import lr_scheduler
from typing import Dict, Optional, List, Tuple

from src.utils.config import load_binary_config, load_multiclass_config

def get_training_config(is_multiclass: bool = False) -> Dict:
    if is_multiclass:
        return load_multiclass_config()
    else:
        return load_binary_config()

def generate_random_combinations(
        param_grid: Dict,
        max_combinations: int,
        random_state: int = 42
) -> Tuple[List[tuple], List[str]]:
    np.random.seed(random_state)
    param_names = list(param_grid.keys())
    all_combinations = list(itertools.product(*[param_grid[k] for k in param_names]))

    if len(all_combinations) <= max_combinations:
        selected_combinations = all_combinations
    else:
        indices = np.random.choice(
            len(all_combinations),
            size=max_combinations,
            replace=False
        )
        selected_combinations = [all_combinations[i] for i in indices]

    return selected_combinations, param_names

def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_config: Dict,
        num_epochs: int
) -> Optional[lr_scheduler.LRScheduler]:
    scheduler_type = scheduler_config.get('type', 'cosine')
    params = scheduler_config.get('params', {})

    if scheduler_type == 'step':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=int(params.get('step_size', 10)),
            gamma=float(params.get('gamma', 0.1))
        )

    elif scheduler_type == 'cosine':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=float(params.get('eta_min', 1e-7))
        )

    elif scheduler_type == 'reduce_on_plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=params.get('mode', 'min'),
            factor=float(params.get('factor', 0.5)),
            patience=int(params.get('patience', 5)),
            min_lr=float(params.get('min_lr', 1e-7))
        )

    return None
