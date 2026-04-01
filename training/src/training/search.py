import torch
import numpy as np
import wandb
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from torchvision.datasets import ImageFolder

from ..models import (
    create_model_with_architecture,
    get_architecture_specific_param_grid,
    get_supported_architectures
)
from .factory import get_training_config, generate_random_combinations
from .trainer import run_training_process
from src.utils.config import load_hyperparameters_config
from ..evaluation import calculate_combined_score
from ..visualization import (
    init_wandb_run, 
    finish_wandb_run, 
    summarize_wandb_repetitions
)
from ..data import (
    create_stratified_holdout_split,
    augment_minority_class
)

class SearchCheckpointManager:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_search_directory(self, key: str) -> Path:
        search_dir = self.base_path / key
        search_dir.mkdir(parents=True, exist_ok=True)
        return search_dir

    def save_state(self, key: str, executed_indices: Set[int], results: Dict):
        search_dir = self.get_search_directory(key)
        state_file = search_dir / 'execution_state.json'
        state = {'executed_indices': list(executed_indices), 'results': results}
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, key: str) -> Tuple[Set[int], Dict]:
        state_file = self.get_search_directory(key) / 'execution_state.json'
        if not state_file.exists():
            return set(), {'best_score': 0.0, 'best_params': None, 'best_metrics': None, 'best_combination_index': -1}
        with open(state_file, 'r') as f:
            state = json.load(f)
        return set(state['executed_indices']), state['results']

    def save_combination(self, key: str, idx: int, params: Dict, metrics: Dict):
        result_file = self.get_search_directory(key) / f'combination_{idx}.json'
        result = {'combination_index': idx, 'params': params, 'aggregated_metrics': metrics}
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

def _initialize_search_session(architecture_name: str, model_type: str, n_repetitions: int, max_combos: int):
    model_type_display = model_type.upper()
    print(f"\n{'-' * 60}")
    print(f"SEARCH SESSION: {architecture_name.upper()} ({model_type_display})")
    print(f"{'-' * 60}")
    print(f"Random Search | {n_repetitions} reps | Max {max_combos} combos")

def _prepare_search_splits(train_dataset, n_repetitions: int, config: Dict, architecture_name: str, model_type: str):
    data_config = config['data']
    minority_config = data_config['minority_augmentation']
    all_splits = []

    print(f"\nCriando {n_repetitions} splits fixos para consistência...")
    for rep in range(n_repetitions):
        train_split, val_split = create_stratified_holdout_split(
            train_dataset, data_config['split_ratios']['train'], 
            data_config['split_ratios']['train_val'], 
            random_state=data_config['random_seed'] + rep,
            max_slices_per_subject=data_config.get('max_slices_per_subject')
        )

        if minority_config['enabled']:
            train_split = augment_minority_class(
                train_split=train_split, architecture_name=architecture_name,
                target_strategy=minority_config['strategy'],
                minority_classes=data_config.get('minority_classes', []),
                target_ratio=minority_config.get('ratio', {}).get('default_ratio', 0.6) if model_type == 'binary' else None,
                target_percentage={int(k): float(v) for k, v in minority_config.get('percentage', {}).get('targets', {}).items()} if model_type == 'multiclass' else None
            )
        all_splits.append((train_split, val_split))
    return all_splits

def _run_combination_repetitions(idx, params, n_repetitions, all_splits, architecture_name, class_names, device, model_type):
    repetition_results = []
    is_multiclass = (model_type == 'multiclass')

    for rep in range(n_repetitions):
        print(f"\nRepetição {rep + 1}/{n_repetitions}")
        train_split, val_split = all_splits[rep]
        
        try:
            model, criterion, optimizer = create_model_with_architecture(
                hyperparams=params, architecture_name=architecture_name,
                class_names=class_names, device=device, train_dataset=train_split
            )
            
            result = run_training_process(
                model=model, criterion=criterion, optimizer=optimizer,
                train_split=train_split, val_split=val_split, device=device,
                hyperparams=params, architecture_name=architecture_name,
                repetition_number=rep + 1, is_multiclass=is_multiclass
            )
            result.update({'repetition': rep + 1, 'model_type': model_type})
            repetition_results.append(result)
        except Exception as e:
            print(f"  [ERRO] Falha na repetição {rep + 1}: {str(e)}")
            continue
    return repetition_results

def _process_combination_results(idx, params, repetition_results, results, architecture_name, model_type, class_names, checkpoint_manager, executed_indices, total_combos):
    is_multiclass = (model_type == 'multiclass')
    res = summarize_wandb_repetitions(repetition_results, params, idx, is_multiclass=is_multiclass, class_names=class_names)
    aggregated = res['aggregated']
    
    checkpoint_key = f"{architecture_name}_{model_type}"
    checkpoint_manager.save_combination(checkpoint_key, idx, params, aggregated)

    score = calculate_combined_score(aggregated, is_multiclass=is_multiclass)
    if score > results['best_score']:
        print(f"\nNOVA MELHOR COMBINAÇÃO! Score: {score:.6f}")
        results.update({'best_score': score, 'best_params': params.copy(), 'best_metrics': aggregated.copy(), 'best_combination_index': idx, 'model_type': model_type})
        if wandb.run: wandb.run.summary.update({"is_best": True, "best_score": score})

    executed_indices.add(idx)
    checkpoint_manager.save_state(checkpoint_key, executed_indices, results)
    print(f"Progresso: {len(executed_indices)}/{total_combos} | Score Atual: {score:.4f} | Melhor: {results['best_score']:.4f}")
    return results

def _report_final_results(results, executed_indices, total_combos, architecture_name, model_type, checkpoint_manager):
    print(f"\n{'-' * 60}\nSEARCH FINISHED: {architecture_name.upper()} | Best Score: {results['best_score']:.6f}\n{'-' * 60}")
    
    final_path = checkpoint_manager.get_search_directory(f"{architecture_name}_{model_type}") / 'final_results.json'
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Resultados detalhados salvos em: {final_path}")
    return results

def evaluate_hyperparameters(param_grid, architecture_name, device, train_dataset, model_type='binary', n_repetitions=1, max_combinations=None):
    config = get_training_config(model_type == 'multiclass')
    hyperparams_config = load_hyperparameters_config()
    class_names = config['model']['class_names']
    
    _initialize_search_session(architecture_name, model_type, n_repetitions, max_combinations)
    
    save_path = os.path.join(os.path.dirname(__file__), str(hyperparams_config['results']['save_path']))
    checkpoint_manager = SearchCheckpointManager(os.path.join(save_path, 'experiments'))
    
    combinations, param_names = generate_random_combinations(param_grid, max_combinations, random_state=config['data']['random_seed'])
    total_combinations = len(combinations)
    
    checkpoint_key = f"{architecture_name}_{model_type}"
    executed_indices, results = checkpoint_manager.load_state(checkpoint_key)
    all_splits = _prepare_search_splits(train_dataset, n_repetitions, config, architecture_name, model_type)

    for idx in [i for i in range(total_combinations) if i not in executed_indices]:
        params = dict(zip(param_names, combinations[idx]))
        print(f"\n{'=' * 60}\nCOMBINAÇÃO [{idx + 1}/{total_combinations}] | Params: {params}\n{'=' * 60}")

        if config['logging']['wandb']['enabled']:
            wandb_cfg = config['logging']['wandb']
            init_wandb_run(
                project_name=wandb_cfg['project'],
                run_name=f"{architecture_name}_combo_{idx+1}_{model_type}",
                config={"architecture": architecture_name, "model_type": model_type, "index": idx, **params},
                entity=wandb_cfg.get('entity'),
                tags=["random_search", architecture_name, model_type],
                group=f"search/{model_type}"  # Grupo aninhado: pasta 'search' > subpasta por tipo
            )

        repetition_results = _run_combination_repetitions(idx, params, n_repetitions, all_splits, architecture_name, class_names, device, model_type)
        
        if repetition_results:
            results = _process_combination_results(idx, params, repetition_results, results, architecture_name, model_type, class_names, checkpoint_manager, executed_indices, total_combinations)
        
        finish_wandb_run()

    return _report_final_results(results, executed_indices, total_combinations, architecture_name, model_type, checkpoint_manager)

def run_random_search(train_dataset: ImageFolder, model_type: str = 'binary', architectures: Optional[List[str]] = None) -> Dict:
    hyperparams_config = load_hyperparameters_config()
    architectures = architectures or get_supported_architectures()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    search_strategy = hyperparams_config.get('search_strategy', {})
    all_results = {}

    for arch in architectures:
        try:
            param_grid = get_architecture_specific_param_grid(arch)
            all_results[arch] = evaluate_hyperparameters(
                param_grid=param_grid, architecture_name=arch, device=device, 
                train_dataset=train_dataset, model_type=model_type,
                n_repetitions=search_strategy.get('n_repetitions', 3),
                max_combinations=search_strategy.get('max_combinations', 60)
            )
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed on architecture {arch}: {e}")
            continue

    return all_results