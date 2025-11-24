import torch
import numpy as np
import wandb
import json
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

from training.src.models import (
    create_model_with_architecture,
    get_architecture_specific_param_grid
)
from training.src.utils import create_stratified_holdout_split
from training.src.data import augment_minority_class
from training.src.evaluation import calculate_combined_score
from training.src.visualization import (
    init_wandb_run,
    summarize_wandb_repetitions,
    finish_wandb_run
)

from .trainer import train_holdout_model

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
        idx: int,
        architecture_name: str,
        checkpoint_manager: GridSearchCheckpointManager,
        executed_indices: Set[int],
        results: Dict,
        total_combinations: int
) -> Dict:
    print(f"\n{'-' * 60}")
    print(f"AGREGANDO RESULTADOS DA COMBINAÇÃO #{idx + 1}")
    print(f"{'-' * 60}\n")

    result = summarize_wandb_repetitions(repetition_results, params, idx)
    aggregated = result['aggregated']

    checkpoint_manager.save_combination_result(
        architecture_name, idx, params, aggregated
    )

    current_score = calculate_combined_score(aggregated)

    print(f"Score Combinado: {current_score:.6f}\n")

    if current_score > results['best_score']:
        improvement = current_score - results['best_score']
        old_best_score = results['best_score']

        results['best_score'] = current_score
        results['best_params'] = params.copy()
        results['best_metrics'] = aggregated.copy()
        results['best_combination_index'] = idx

        wandb.run.summary["is_best"] = True
        wandb.run.summary["best_score"] = current_score

        print(f"\nNOVA MELHOR COMBINAÇÃO!")
        print(f"  Score Anterior: {old_best_score:.6f}")
        print(f"  Score Atual: {current_score:.6f}\n")

        if old_best_score > 0:
            print(f"  Melhoria: +{improvement:.6f} ({improvement / old_best_score * 100:.2f}%)\n")
    else:
        score_diff = results['best_score'] - current_score
        print(f"\nNão superou a melhor combinação:")
        print(f"   Score Atual:  {current_score:.6f}")
        print(f"   Melhor Score: {results['best_score']:.6f}")
        print(f"   Diferença: {score_diff:.6f} ({score_diff / results['best_score'] * 100:.2f}%)\n")

    executed_indices.add(idx)
    checkpoint_manager.save_execution_state(architecture_name, executed_indices, results)

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
        checkpoint_manager: GridSearchCheckpointManager
) -> Dict:
    print(f"\n{'-' * 60}")
    print(f"BUSCA DE HIPERPARÂMETROS CONCLUÍDA")
    print(f"{'-' * 60}")
    print(f"\nArquitetura: {architecture_name.upper()}")
    print(f"Combinações Testadas: {len(executed_indices)}/{total_combinations} "
          f"({len(executed_indices) / total_combinations * 100:.1f}%)\n")

    if results['best_params']:
        print(f"\nScore Final: {results['best_score']:.6f}")

        print(f"\nHiperparâmetros Ótimos:")
        for param_name, param_value in results['best_params'].items():
            print(f"  {param_name:20s}: {param_value}")

        if results.get('best_metrics'):
            best_metrics = results['best_metrics']

            print(f"\nMétricas Finais:")
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

            print(f"\nInterpretação Clínica:")
            sens = best_metrics.get('mean_recall', 0.0)
            spec = best_metrics.get('mean_specificity', 0.0)

            print(f"  De cada 100 pacientes COM demência:")
            print(f"    ~{sens * 100:.0f} são detectados corretamente")
            print(f"  De cada 100 pacientes SEM demência:")
            print(f"    ~{spec * 100:.0f} são identificados corretamente\n")

    final_path = checkpoint_manager.get_search_directory(architecture_name) / 'final_results.json'

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
        train_dataset,
        class_names: List[str],
        max_combinations: int,
        num_epochs: int,
        early_stopping_patience: int,
        n_repetitions: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.3,
        save_path: str = './shared/logs/experiments',
        augment_minority: bool = True,
        minority_target_strategy: str = 'balance',
        minority_target_ratio: float = 0.5,
        wandb_project: str = "alzheimer-detection",
        wandb_entity: Optional[str] = None
) -> Dict:
    print(f"\n{'-' * 60}")
    print(f"BUSCA DE HIPERPARÂMETROS: {architecture_name.upper()}")
    print(f"{'-' * 60}")
    print(f"Método: Random Search com Holdout ({n_repetitions} repetições)")
    print(f"Divisão: {train_ratio:.0%} treino, {val_ratio:.0%} validação")
    print(f"Max combinações: {max_combinations}")
    print(f"Augmentação Minoritária: {'SIM' if augment_minority else 'NÃO'}")
    if augment_minority:
        print(f"  Estratégia: {minority_target_strategy}")
        if minority_target_strategy == 'ratio':
            print(f"  Target Ratio: {minority_target_ratio}")
    print(f"{'-' * 60}\n")

    checkpoint_manager = GridSearchCheckpointManager(save_path)

    combinations, param_names = generate_random_combinations(
        param_grid, max_combinations
    )
    total_combinations = len(combinations)

    print(f"Geradas {total_combinations} combinações aleatórias\n")

    executed_indices, results = checkpoint_manager.load_execution_state(architecture_name)

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
            random_state=42 + rep
        )

        if augment_minority:
            print(f"\nAplicando augmentação na classe minoritária...")
            train_split = augment_minority_class(
                train_split=train_split,
                target_strategy=minority_target_strategy,
                target_ratio=min(minority_target_ratio, 0.6),
                architecture_name=architecture_name,
                minority_class=0,  # Demented
            )

        all_splits.append((train_split, val_split))

    print(f"\nSPLITS CRIADOS COM SUCESSO!")
    print(f"  Todas as {total_combinations} combinações serão testadas nesses {n_repetitions} splits.")
    print(f"{'-' * 60}\n")

    for idx in remaining_indices:
        combination = combinations[idx]
        params = dict(zip(param_names, combination))

        print(f"\n{'-' * 60}")
        print(f"COMBINAÇÃO [{idx + 1}/{total_combinations}] #{idx}")
        print(f"{'-' * 60}")
        print(f"Parâmetros: {params}")
        print(f"Progresso: {len(executed_indices)}/{total_combinations} combinações executadas")
        print(f"{'-' * 60}\n")

        run_name = f"{architecture_name}_combo_{idx + 1}"
        init_wandb_run(
            project_name=wandb_project,
            run_name=run_name,
            config={
                "architecture": architecture_name,
                "combination_index": idx,
                **params,
                "n_repetitions": n_repetitions,
                "num_epochs": num_epochs,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "augment_minority": augment_minority,
                "minority_strategy": minority_target_strategy if augment_minority else None
            },
            entity=wandb_entity,
            tags=["grid_search", architecture_name],
            group=f"{architecture_name}_search"
        )

        repetition_results = []

        for rep in range(n_repetitions):
            print(f"REPETIÇÃO {rep + 1}/{n_repetitions}")

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
                    focal_gamma=2.0,
                    label_smoothing=0.1
                )

                rep_result = train_holdout_model(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    train_split=train_split,
                    val_split=val_split,
                    device=device,
                    class_names=class_names,
                    hyperparams=params,
                    num_epochs=num_epochs,
                    early_stopping_patience=early_stopping_patience,
                    architecture_name=architecture_name,
                    use_gradient_clipping=True,
                    max_grad_norm=1.0,
                    repetition_number=rep + 1
                )

                rep_result['repetition'] = rep + 1
                repetition_results.append(rep_result)

            except Exception as e:
                print(f"ERRO na Repetição {rep + 1}:")
                print(f"Detalhe do Erro: {e}")
                import traceback
                traceback.print_exc()
                continue

        if repetition_results:
            results = improved_combination_evaluation(
                repetition_results=repetition_results,
                params=params,
                idx=idx,
                architecture_name=architecture_name,
                checkpoint_manager=checkpoint_manager,
                executed_indices=executed_indices,
                results=results,
                total_combinations=total_combinations
            )

        finish_wandb_run()

    results = final_search_summary(
        results=results,
        executed_indices=executed_indices,
        total_combinations=total_combinations,
        architecture_name=architecture_name,
        checkpoint_manager=checkpoint_manager
    )

    return results

def run_grid_search(
        train_dataset,
        classes: List[str],
        architectures: Optional[List[str]] = None,
        save_path: str = './shared/logs/experiments',
        wandb_project: str = "alzheimer-detection",
        wandb_entity: Optional[str] = None
) -> Dict:
    print(f"\n{'=' * 80}")
    print("INICIANDO BUSCA DE HIPERPARÂMETROS PARA TODAS AS ARQUITETURAS")
    print(f"{'=' * 80}\n")

    if architectures is None:
        architectures = [
            'resnext50_32x4d',
            'convnext_tiny',
            'efficientnetv2_s',
            'densenet121',
            'vit_b_16',
            'swin_v2_tiny',
        ]

    print(f"Arquiteturas a serem testadas: {architectures}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    all_results = {}

    for arch in architectures:
        try:
            print(f"\n{'-' * 80}")
            print(f"INICIANDO: {arch.upper()}")
            print(f"{'-' * 80}\n")

            param_grid = get_architecture_specific_param_grid(arch)

            if arch in ['vit_b_16', 'swin_v2_tiny']:
                num_epochs = 500
                patience = 50
                max_combinations = 200
            else:
                num_epochs = 350
                patience = 30
                max_combinations = 300

            results = search_best_hyperparameters_holdout(
                param_grid=param_grid,
                architecture_name=arch,
                device=device,
                train_dataset=train_dataset,
                class_names=classes,
                max_combinations=max_combinations,
                num_epochs=num_epochs,
                early_stopping_patience=patience,
                n_repetitions=1,
                train_ratio=0.7,
                val_ratio=0.3,
                save_path=save_path,
                augment_minority=True,
                minority_target_strategy='ratio',
                minority_target_ratio=0.6,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity
            )

            all_results[arch] = results

            print(f"\n{'-' * 80}")
            print(f"{arch.upper()} CONCLUÍDO!")
            print(f"{'-' * 80}")

        except Exception as e:
            print(f"ERRO COM {arch.upper()}: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}")
    print("BUSCA CONCLUÍDA PARA TODAS AS ARQUITETURAS")
    print(f"{'=' * 80}\n")

    return all_results