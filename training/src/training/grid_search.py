import torch
import numpy as np
import wandb
import json
import os
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from torchvision.datasets import ImageFolder

from training.src.models import (
    create_model_with_architecture,
    get_architecture_specific_param_grid
)
from training.src.utils import (
    create_stratified_holdout_split,
    load_binary_config,
    load_multiclass_config,
    load_hyperparameters_config
)
from training.src.data import augment_minority_class
from training.src.evaluation import calculate_combined_score
from training.src.visualization import (
    init_wandb_run,
    summarize_wandb_repetitions,
    finish_wandb_run
)
from .trainer import train_holdout_model

def get_model_config(model_type: str = 'binary') -> Dict:
    if model_type == 'multiclass':
        return load_multiclass_config()
    else:
        return load_binary_config()

class GridSearchCheckpointManager:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_search_directory(self, architecture_name: str) -> Path:
        search_dir = self.base_path / architecture_name
        search_dir.mkdir(parents=True, exist_ok=True)
        return search_dir

    def save_combination_result(
            self,
            architecture_name: str,
            combination_idx: int,
            params: Dict,
            aggregated_metrics: Dict
    ):
        search_dir = self.get_search_directory(architecture_name)

        result_file = search_dir / f'combination_{combination_idx}.json'

        result = {
            'combination_index': combination_idx,
            'params': params,
            'aggregated_metrics': aggregated_metrics
        }

        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

    def save_execution_state(
            self,
            architecture_name: str,
            executed_indices: Set[int],
            results: Dict
    ):
        search_dir = self.get_search_directory(architecture_name)

        state_file = search_dir / 'execution_state.json'

        state = {
            'executed_indices': list(executed_indices),
            'results': results
        }

        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_execution_state(
            self,
            architecture_name: str
    ) -> Tuple[Set[int], Dict]:
        search_dir = self.get_search_directory(architecture_name)
        state_file = search_dir / 'execution_state.json'

        if not state_file.exists():
            return set(), {
                'best_score': 0.0,
                'best_params': None,
                'best_metrics': None,
                'best_combination_index': -1
            }

        with open(state_file, 'r') as f:
            state = json.load(f)

        executed_indices = set(state['executed_indices'])
        results = state['results']

        return executed_indices, results

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

def improved_combination_evaluation(
        repetition_results: List[Dict],
        params: Dict,
        index_combination: int,
        architecture_name: str,
        checkpoint_manager: GridSearchCheckpointManager,
        executed_indices: Set[int],
        results: Dict,
        total_combinations: int,
        model_type: str = 'binary'
) -> Dict:
    model_type_display = "BINÁRIO" if model_type == 'binary' else "MULTICLASSE"

    print(f"\n{'-' * 60}")
    print(f"AGREGANDO RESULTADOS DA COMBINAÇÃO #{index_combination + 1} ({model_type_display})")
    print(f"{'-' * 60}\n")

    result = summarize_wandb_repetitions(repetition_results, params, index_combination)
    aggregated = result['aggregated']

    checkpoint_key = f"{architecture_name}_{model_type}"
    checkpoint_manager.save_combination_result(
        checkpoint_key, index_combination, params, aggregated
    )

    current_score = calculate_combined_score(aggregated)

    print(f"Score Combinado ({model_type_display}): {current_score:.6f}\n")

    if current_score > results['best_score']:
        improvement = current_score - results['best_score']
        old_best_score = results['best_score']

        results['best_score'] = current_score
        results['best_params'] = params.copy()
        results['best_metrics'] = aggregated.copy()
        results['best_combination_index'] = index_combination
        results['model_type'] = model_type

        wandb.run.summary["is_best"] = True
        wandb.run.summary["best_score"] = current_score
        wandb.run.summary["model_type"] = model_type_display

        print(f"\nNOVA MELHOR COMBINAÇÃO ({model_type_display})!")
        print(f"  Score Anterior: {old_best_score:.6f}")
        print(f"  Score Atual: {current_score:.6f}\n")

        if old_best_score > 0:
            print(f"  Melhoria: +{improvement:.6f} ({improvement / old_best_score * 100:.2f}%)\n")
    else:
        score_diff = results['best_score'] - current_score
        print(f"\nNão superou a melhor combinação ({model_type_display}):")
        print(f"   Score Atual:  {current_score:.6f}")
        print(f"   Melhor Score: {results['best_score']:.6f}")
        print(f"   Diferença: {score_diff:.6f} ({score_diff / results['best_score'] * 100:.2f}%)\n")

    executed_indices.add(index_combination)
    checkpoint_manager.save_execution_state(checkpoint_key, executed_indices, results)

    progress_percent = len(executed_indices) / total_combinations * 100
    remaining = total_combinations - len(executed_indices)

    print(f"\nCheckpoint Salvo!")
    print(f"  Progresso: {len(executed_indices)}/{total_combinations} ({progress_percent:.1f}%)")
    print(f"  Combinações Restantes: {remaining}")

    print(f"\n{'-' * 60}\n")

    return results

def final_search_summary(
        results: Dict,
        executed_indices: Set[int],
        total_combinations: int,
        architecture_name: str,
        checkpoint_manager: GridSearchCheckpointManager,
        model_type: str = 'binary'
) -> Dict:
    model_type_display = "BINÁRIO" if model_type == 'binary' else "MULTICLASSE"

    config = get_model_config(model_type)
    class_names = config['model']['class_names']

    print(f"\n{'-' * 60}")
    print(f"BUSCA DE HIPERPARÂMETROS CONCLUÍDA ({model_type_display})")
    print(f"{'-' * 60}")
    print(f"\nArquitetura: {architecture_name.upper()}")
    print(f"Tipo de Modelo: {model_type_display}")
    print(f"Classes: {class_names}")
    print(f"Combinações Testadas: {len(executed_indices)}/{total_combinations} "
          f"({len(executed_indices) / total_combinations * 100:.1f}%)\n")

    if results['best_params']:
        print(f"\nScore Final ({model_type_display}): {results['best_score']:.6f}")

        print(f"\nHiperparâmetros Ótimos ({model_type_display}):")
        for param_name, param_value in results['best_params'].items():
            print(f"  {param_name:20s}: {param_value}")

        if results.get('best_metrics'):
            best_metrics = results['best_metrics']

            print(f"\nMétricas Finais ({model_type_display}):")
            metrics_to_show = [
                ('Balanced Accuracy', 'mean_balanced_accuracy', 'std_balanced_accuracy'),
                ('F1-Score', 'mean_f1', 'std_f1'),
                ('Sensitivity', 'mean_recall', 'std_recall'),
                ('Specificity', 'mean_specificity', 'std_specificity'),
                ('Precision', 'mean_precision', 'std_precision'),
                ('MCC', 'mean_mcc', 'std_mcc'),
                ('Val Loss', 'mean_loss', 'std_loss')
            ]

            for metric_name, mean_key, std_key in metrics_to_show:
                mean_val = best_metrics.get(mean_key, 0.0)
                std_val = best_metrics.get(std_key, 0.0)
                print(f"  {metric_name:<25}: {mean_val:.4f} +- {std_val:.4f}")

            print(f"\nInterpretação Clínica ({model_type_display}):")
            sens = best_metrics.get('mean_recall', 0.0)
            spec = best_metrics.get('mean_specificity', 0.0)

            if model_type == 'binary':
                print(f"  De cada 100 pacientes COM demência:")
                print(f"    ~{sens * 100:.0f} são detectados corretamente")
                print(f"  De cada 100 pacientes SEM demência:")
                print(f"    ~{spec * 100:.0f} são identificados corretamente\n")
            else:
                print(f"  Classifica entre: {', '.join(class_names)}")
                print(f"  Acurácia Balanceada: {best_metrics.get('mean_balanced_accuracy', 0.0) * 100:.2f}%\n")

    final_path = checkpoint_manager.get_search_directory(
        f"{architecture_name}_{model_type}"
    ) / 'final_results.json'

    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_serializable[key] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in value.items()
            }
        else:
            results_serializable[key] = value

    with open(final_path, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)

    print(f"\nResultados salvos em:")
    print(f"   {final_path}")
    print(f"\n{'-' * 60}\n")

    return results

def search_best_hyperparameters_holdout(
        param_grid: Dict,
        architecture_name: str,
        device: torch.device,
        train_dataset: ImageFolder,
        model_type: str = 'binary',
        n_repetitions: int = 1,
        max_combinations: Optional[int] = None
) -> Dict:
    config = get_model_config(model_type)
    hyperparams_config = load_hyperparameters_config()

    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    logging_config = config['logging']

    class_names = model_config['class_names']
    num_classes = model_config['num_classes']

    num_epochs = training_config['epochs']
    early_stopping_patience = training_config['patience']

    train_ratio = data_config['split_ratios']['train']
    val_ratio = data_config['split_ratios']['val']
    random_seed = data_config['random_seed']
    stratify = data_config['stratify']

    minority_config = data_config['minority_augmentation']
    augment_minority = minority_config['enabled']
    minority_target_strategy = minority_config['strategy']
    minority_target_ratio = minority_config.get('target_ratio', 0.6)
    minority_classes = data_config.get('minority_classes', [])

    wandb_config = logging_config.get('wandb', {})
    wandb_enabled = wandb_config.get('enabled', False)
    wandb_project = wandb_config.get('project', 'alzheimer-detection')
    wandb_entity = wandb_config.get('entity', None)

    save_path = hyperparams_config['results']['save_path']

    if max_combinations is None:
        arch_type = hyperparams_config['model_config'][architecture_name.lower()]['type']
        if arch_type == 'transformer':
            max_combinations = 200
        else:
            max_combinations = 300

    model_type_display = "BINÁRIO" if model_type == 'binary' else "MULTICLASSE"

    print(f"\n{'-' * 60}")
    print(f"BUSCA DE HIPERPARÂMETROS: {architecture_name.upper()} ({model_type_display})")
    print(f"{'-' * 60}")
    print(f"Configurações Carregadas:")
    print(f"  Método: Random Search com Holdout ({n_repetitions} repetições)")
    print(f"  Classes: {class_names}")
    print(f"  Número de Classes: {num_classes}")
    print(f"  Divisão: {train_ratio:.0%} treino, {val_ratio:.0%} validação")
    print(f"  Épocas: {num_epochs}")
    print(f"  Patience: {early_stopping_patience}")
    print(f"  Max Combinações: {max_combinations}")
    print(f"  Stratify: {stratify}")
    print(f"  Random Seed: {random_seed}")
    print(f"  Augmentação Minoritária: {'SIM' if augment_minority else 'NÃO'}")
    if augment_minority:
        print(f"    Estratégia: {minority_target_strategy}")
        if minority_target_strategy == 'ratio':
            print(f"    Target Ratio: {minority_target_ratio}")
        print(f"    Classes Minoritárias: {minority_classes}")
    print(f"  WandB: {'Habilitado' if wandb_enabled else 'Desabilitado'}")
    if wandb_enabled:
        print(f"    Project: {wandb_project}")
        if wandb_entity:
            print(f"    Entity: {wandb_entity}")
    print(f"  Save Path: {save_path}")
    print(f"{'-' * 60}\n")

    checkpoint_manager = GridSearchCheckpointManager(save_path)

    combinations, param_names = generate_random_combinations(
        param_grid, max_combinations, random_state=random_seed
    )
    total_combinations = len(combinations)

    print(f"Geradas {total_combinations} combinações aleatórias\n")

    checkpoint_key = f"{architecture_name}_{model_type}"
    executed_indices, results = checkpoint_manager.load_execution_state(checkpoint_key)

    remaining_indices = [i for i in range(total_combinations) if i not in executed_indices]

    if executed_indices:
        print(f"Retomando busca: {len(executed_indices)} combinações já executadas\n")

    print(f"{'-' * 60}")
    print("CRIANDO SPLITS FIXOS PARA TODAS AS COMBINAÇÕES")
    print(f"{'-' * 60}\n")

    all_splits = []

    for rep in range(n_repetitions):
        print(f"Criando Split da Repetição {rep + 1}/{n_repetitions}...")

        train_split, val_split = create_stratified_holdout_split(
            train_dataset,
            train_ratio,
            val_ratio,
            random_state=random_seed + rep
        )

        if augment_minority:
            print(f"  Aplicando augmentação nas classes minoritárias ({model_type_display})...")

            train_split = augment_minority_class(
                train_split=train_split,
                target_strategy=minority_target_strategy,
                target_ratio=min(minority_target_ratio, 0.6),
                architecture_name=architecture_name,
                minority_classes=minority_classes,
            )

        all_splits.append((train_split, val_split))

    print(f"\nSPLITS CRIADOS COM SUCESSO!")
    print(f"  Todas as {total_combinations} combinações serão testadas nesses {n_repetitions} splits.")
    print(f"{'-' * 60}\n")

    for idx in remaining_indices:
        combination = combinations[idx]
        params = dict(zip(param_names, combination))

        print(f"\n{'-' * 60}")
        print(f"COMBINAÇÃO [{idx + 1}/{total_combinations}] #{idx} ({model_type_display})")
        print(f"{'-' * 60}")
        print(f"Parâmetros:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        print(f"Progresso: {len(executed_indices)}/{total_combinations} executadas")
        print(f"{'-' * 60}\n")

        if wandb_enabled:
            run_name = f"{architecture_name}_combo_{idx + 1}_{model_type}"

            wandb_dir = os.path.join(save_path, 'wandb_logs')

            run = init_wandb_run(
                project_name=wandb_project,
                run_name=run_name,
                config={
                    "architecture": architecture_name,
                    "model_type": model_type_display,
                    "combination_index": idx,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    **params,
                    "n_repetitions": n_repetitions,
                    "num_epochs": num_epochs,
                    "patience": early_stopping_patience,
                    "train_ratio": train_ratio,
                    "val_ratio": val_ratio,
                    "augment_minority": augment_minority,
                    "minority_strategy": minority_target_strategy if augment_minority else None,
                    "minority_classes": minority_classes if augment_minority else None,
                    "stratify": stratify
                },
                entity=wandb_entity,
                tags=["grid_search", architecture_name, model_type],
                group=f"{architecture_name}_search_{model_type}",
                save_code=False,
                directory=wandb_dir
            )

            if run is None:
                wandb_enabled = False

        repetition_results = []

        for rep in range(n_repetitions):
            print(f"\n{'-' * 40}")
            print(f"REPETIÇÃO {rep + 1}/{n_repetitions}")
            print(f"{'-' * 40}\n")

            try:
                train_split, val_split = all_splits[rep]

                print(f"Usando split fixo da repetição {rep + 1}")
                print(f"  Train size: {len(train_split)}")
                print(f"  Val size: {len(val_split)}\n")

                model, criterion, optimizer = create_model_with_architecture(
                    hyperparams=params,
                    architecture_name=architecture_name,
                    class_names=class_names,
                    device=device,
                    train_dataset=train_split,
                    label_smoothing=None
                )

                rep_result = train_holdout_model(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    train_split=train_split,
                    val_split=val_split,
                    device=device,
                    hyperparams=params,
                    architecture_name=architecture_name,
                    use_gradient_clipping=True,
                    max_grad_norm=1.0,
                    repetition_number=rep + 1,
                    is_multiclass=(model_type == 'multiclass')
                )

                rep_result['repetition'] = rep + 1
                rep_result['model_type'] = model_type
                repetition_results.append(rep_result)

            except Exception as e:
                print(f"\nERRO na Repetição {rep + 1}:")
                print(f"   {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        if repetition_results:
            results = improved_combination_evaluation(
                repetition_results=repetition_results,
                params=params,
                index_combination=idx,
                architecture_name=architecture_name,
                checkpoint_manager=checkpoint_manager,
                executed_indices=executed_indices,
                results=results,
                total_combinations=total_combinations,
                model_type=model_type
            )

        if wandb_enabled:
            finish_wandb_run()

    results = final_search_summary(
        results=results,
        executed_indices=executed_indices,
        total_combinations=total_combinations,
        architecture_name=architecture_name,
        checkpoint_manager=checkpoint_manager,
        model_type=model_type
    )

    return results

def run_grid_search(
        train_dataset: ImageFolder,
        model_type: str = 'binary',
        architectures: Optional[List[str]] = None
) -> Dict:
    hyperparams_config = load_hyperparameters_config()
    config = get_model_config(model_type)

    class_names = config['model']['class_names']

    model_type_display = "BINÁRIO" if model_type == 'binary' else "MULTICLASSE"

    print(f"\n{'=' * 80}")
    print(f"INICIANDO BUSCA DE HIPERPARÂMETROS ({model_type_display})")
    print(f"{'=' * 80}\n")

    if architectures is None:
        architectures = (
                hyperparams_config['supported_architectures']['cnn'] +
                hyperparams_config['supported_architectures']['transformer']
        )

    print(f"Arquiteturas a serem testadas: {architectures}")
    print(f"Tipo de Modelo: {model_type_display}")
    print(f"Classes: {class_names}\n")

    hardware_config = hyperparams_config.get('hardware', {})
    device_config = hardware_config.get('device', 'cuda')

    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)

    print(f"Device: {device}\n")

    all_results = {}

    for arch in architectures:
        try:
            print(f"\n{'-' * 80}")
            print(f"INICIANDO: {arch.upper()} ({model_type_display})")
            print(f"{'-' * 80}\n")

            param_grid = get_architecture_specific_param_grid(arch)

            results = search_best_hyperparameters_holdout(
                param_grid=param_grid,
                architecture_name=arch,
                device=device,
                train_dataset=train_dataset,
                model_type=model_type,
                n_repetitions=1,
                max_combinations=None
            )

            all_results[arch] = results

        except Exception as e:
            print(f"\nERRO COM {arch.upper()}: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}")
    print(f"BUSCA DE HIPERPARÂMETROS CONCLUÍDA ({model_type_display})")
    print(f"{'=' * 80}\n")

    return all_results