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
from ..evaluation import calculate_combined_score, aggregate_fold_metrics
from ..visualization.wandb_logger import (
    init_wandb_run, 
    finish_wandb_run, 
    log_search_fold_progress
)
from ..visualization.terminal import print_fold_summary
from ..data.dataset_wrappers import (
    create_kfold_splits,
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
        default_results = {'best_score': 0.0, 'best_params': None, 'best_metrics': None, 'best_combination_index': -1}
        
        if not state_file.exists():
            return set(), default_results
            
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            executed_indices = set(state.get('executed_indices', []))
            results = state.get('results', default_results)
            return executed_indices, results
        except Exception as e:
            print(f"Erro ao carregar estado em {state_file}: {e}. Iniciando novo estado.")
            return set(), default_results

    def save_combination_init(self, key: str, idx: int, params: Dict):
        result_file = self.get_search_directory(key) / f'combination_{idx}.json'
        result = {
            'combination_index': idx,
            'params': params,
            'folds': {},
            'status': 'in_progress'
        }
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

    def save_fold_result(self, key: str, combo_idx: int, fold_idx: int, result: Dict):
        result_file = self.get_search_directory(key) / f'combination_{combo_idx}.json'
        
        if not result_file.exists():
            return

        clean_result = {k: v for k, v in result.items() if k != 'history'}

        with open(result_file, 'r') as f:
            data = json.load(f)
        
        data['folds'][str(fold_idx)] = clean_result
        
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def save_combination_final(self, key: str, idx: int, metrics: Dict):
        result_file = self.get_search_directory(key) / f'combination_{idx}.json'
        
        if not result_file.exists():
            return

        with open(result_file, 'r') as f:
            data = json.load(f)
        
        data['aggregated_metrics'] = metrics
        data['status'] = 'completed'
        
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

def _initialize_search_session(architecture_name: str, model_type: str, n_folds: int, max_combos: int):
    model_type_display = model_type.upper()
    print(f"\n{'-' * 60}")
    print(f"SEARCH SESSION: {architecture_name.upper()} ({model_type_display})")
    print(f"{'-' * 60}")
    print(f"Random Search | {n_folds}-Fold CV | Max {max_combos} combos")

def _prepare_search_splits(train_dataset, n_folds: int, config: Dict, architecture_name: str, model_type: str):
    data_config = config['data']
    minority_config = data_config['minority_augmentation']
    
    print(f"\nCriando {n_folds}-Fold CV para busca...")
    folds = create_kfold_splits(
        dataset=train_dataset,
        n_folds=n_folds,
        random_state=data_config['random_seed'],
        max_slices_per_subject=data_config.get('max_slices_per_subject'),
        minority_classes=data_config.get('minority_classes', []),
        architecture_name=architecture_name,
        minority_config=minority_config if minority_config.get('enabled') else None
    )
    return folds

def _run_combination_folds(idx, params, n_folds, all_splits, architecture_name, class_names, device, model_type, checkpoint_manager):
    fold_results = []
    is_multiclass = (model_type == 'multiclass')
    checkpoint_key = f"{architecture_name}_{model_type}"

    for fold_idx in range(n_folds):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        train_split, val_split = all_splits[fold_idx]
        
        try:
            model, criterion, optimizer = create_model_with_architecture(
                hyperparams=params, architecture_name=architecture_name,
                class_names=class_names, device=device, train_dataset=train_split
            )
            
            result = run_training_process(
                model=model, criterion=criterion, optimizer=optimizer,
                train_split=train_split, val_split=val_split, device=device,
                hyperparams=params, architecture_name=architecture_name,
                fold_number=fold_idx + 1, is_multiclass=is_multiclass
            )
            result.update({'fold': fold_idx + 1, 'model_type': model_type})
            fold_results.append(result)

            checkpoint_manager.save_fold_result(checkpoint_key, idx + 1, fold_idx + 1, result)

            f1_key = 'macro_f1' if is_multiclass else 'f1_score'
            print(f"  > [OK] Fold {fold_idx + 1} concluído. {f1_key.replace('_', ' ').upper()}: {result.get(f1_key, 0):.4f}")

            if wandb.run:
                log_search_fold_progress(
                    fold_results=fold_results,
                    combination_index=idx,
                    is_multiclass=is_multiclass
                )
        except Exception as e:
            print(f"  [ERRO] Falha no fold {fold_idx}: {str(e)}")
            continue
    return fold_results

def _process_combination_results(idx, params, fold_results, results, architecture_name, model_type, class_names, checkpoint_manager, executed_indices, total_combos):
    is_multiclass = (model_type == 'multiclass')
    aggregated = aggregate_fold_metrics(fold_results, is_multiclass)
    
    print_fold_summary(aggregated, idx)

    checkpoint_key = f"{architecture_name}_{model_type}"
    checkpoint_manager.save_combination_final(checkpoint_key, idx, aggregated)

    score = calculate_combined_score(aggregated, is_multiclass=is_multiclass)
    if score > results['best_score']:
        print(f"\nNOVA MELHOR COMBINAÇÃO! Score: {score:.6f}")
        results.update({'best_score': score, 'best_params': params.copy(), 'best_metrics': aggregated.copy(), 'best_combination_index': idx, 'model_type': model_type})
        if wandb.run: wandb.run.summary.update({"is_best": True, "best_score": score})

    checkpoint_manager.save_state(checkpoint_key, executed_indices, results)
    print(f"Progresso: {len(executed_indices)}/{total_combos} | Score Atual: {score:.4f} | Melhor: {results['best_score']:.4f}")
    return results

def _mark_combination_failed(idx, architecture_name, model_type, checkpoint_manager, executed_indices, results):
    print(f"\n[AVISO] Combinação {idx+1} falhou em todos os folds. Pulando para evitar loop infinito.")
    executed_indices.add(idx)
    checkpoint_key = f"{architecture_name}_{model_type}"
    checkpoint_manager.save_state(checkpoint_key, executed_indices, results)

def _report_final_results(results, executed_indices, total_combos, architecture_name, model_type, checkpoint_manager):
    print(f"\n{'-' * 60}\nSEARCH FINISHED: {architecture_name.upper()} | Best Score: {results['best_score']:.6f}\n{'-' * 60}")
    
    final_path = checkpoint_manager.get_search_directory(f"{architecture_name}_{model_type}") / 'final_results.json'
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Resultados detalhados salvos em: {final_path}")
    return results

def evaluate_hyperparameters(param_grid, architecture_name, device, train_dataset, model_type='binary', n_folds=5, max_combinations=None):
    config = get_training_config(model_type == 'multiclass')
    hyperparams_config = load_hyperparameters_config()
    class_names = config['model']['class_names']
    
    _initialize_search_session(architecture_name, model_type, n_folds, max_combinations)
    
    save_path = os.path.join(os.path.dirname(__file__), str(hyperparams_config['results']['save_path']))
    checkpoint_manager = SearchCheckpointManager(os.path.join(save_path, 'experiments'))
    
    combinations, param_names = generate_random_combinations(param_grid, max_combinations, random_state=config['data']['random_seed'])
    total_combinations = len(combinations)
    
    checkpoint_key = f"{architecture_name}_{model_type}"
    executed_indices, results = checkpoint_manager.load_state(checkpoint_key)
    all_splits = _prepare_search_splits(train_dataset, n_folds, config, architecture_name, model_type)

    for idx in [i for i in range(total_combinations) if i not in executed_indices]:
        params = dict(zip(param_names, combinations[idx]))
        print(f"\n{'=' * 60}\nCOMBINAÇÃO [{idx + 1}/{total_combinations}] | Params: {params}\n{'=' * 60}")
        
        checkpoint_key = f"{architecture_name}_{model_type}"
        checkpoint_manager.save_combination_init(checkpoint_key, idx + 1, params)

        try:
            if config['logging']['wandb']['enabled']:
                wandb_cfg = config['logging']['wandb']
                init_wandb_run(
                    project_name=wandb_cfg['project'],
                    run_name=f"{architecture_name}_combo_{idx+1}_{model_type}",
                    config={"architecture": architecture_name, "model_type": model_type, "index": idx, **params},
                    entity=wandb_cfg.get('entity'),
                    tags=["random_search", architecture_name, model_type],
                    group=f"kfold_search/{model_type}"
                )

            fold_results = _run_combination_folds(idx, params, n_folds, all_splits, architecture_name, class_names, device, model_type, checkpoint_manager)
            if fold_results:
                results = _process_combination_results(idx, params, fold_results, results, architecture_name, model_type, class_names, checkpoint_manager, executed_indices, total_combinations)
            else:
                results = _mark_combination_failed(idx, architecture_name, model_type, checkpoint_manager, executed_indices, results)
            
        except Exception as e:
            print(f"\n[ERRO CRÍTICO] Falha catastrófica na combinação {idx+1}: {e}")
            results = _mark_combination_failed(idx, architecture_name, model_type, checkpoint_manager, executed_indices, results)
        finally:
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
                n_folds=search_strategy.get('n_folds', 5),
                max_combinations=search_strategy.get('max_combinations', 60)
            )
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed on architecture {arch}: {e}")
            continue

    return all_results