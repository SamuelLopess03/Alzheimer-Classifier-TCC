import wandb
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
import torch
import torch.nn as nn

from ..evaluation import aggregate_repetition_metrics

def log_search_metrics(aggregated: Dict, combination_index: int):
    if wandb.run is None:
        return

    wandb.log({
        "search/f1":           aggregated.get('mean_f1', 0.0),
        "search/balanced_acc": aggregated.get('mean_balanced_accuracy', 0.0),
        "search/mcc":          aggregated.get('mean_mcc', 0.0),
        "search/val_loss":     aggregated.get('mean_loss', 0.0),
        "search/combination":  combination_index,
    })

def summarize_wandb_repetitions(
        repetition_results: List[Dict],
        params: Dict,
        idx: int,
        is_multiclass: bool = False,
        class_names: Optional[List[str]] = None
) -> Dict:
    aggregated = aggregate_repetition_metrics(repetition_results, is_multiclass)
    
    log_search_metrics(aggregated, idx)

    val_f1_scores = [r['best_f1_score'] for r in repetition_results]
    best_rep_idx = int(np.argmax(val_f1_scores))

    print(f"\nAgregação da Combinação #{idx + 1}")
    print(f"  Mean F1: {aggregated['mean_f1'] * 100:.2f}% ± {aggregated['std_f1'] * 100:.2f}%")
    print(f"  Mean Balanced Acc: {aggregated['mean_balanced_accuracy'] * 100:.2f}% ± {aggregated['std_balanced_accuracy'] * 100:.2f}%")

    return {
        'params': params,
        'combination_index': idx,
        'aggregated': aggregated,
        'n_successful_repetitions': len(repetition_results),
        'best_repetition_index': best_rep_idx,
        'best_repetition_f1_score': float(val_f1_scores[best_rep_idx])
    }

def log_final_training_metrics(
        metrics: Dict,
        epoch: int,
        repetition: int,
        class_names: List[str],
        train_loss: float = None,
        learning_rate: float = None
):
    if wandb.run is None:
        return

    prefix = f"rep_{repetition}"

    log_dict = {
        "epoch": epoch,
        f"{prefix}/val/f1":           metrics.get('f1_score', 0.0),
        f"{prefix}/val/balanced_acc": metrics.get('balanced_accuracy', 0.0),
        f"{prefix}/val/accuracy":     metrics.get('accuracy', 0.0),
        f"{prefix}/val/mcc":          metrics.get('matthews_correlation_coefficient', 0.0),
        f"{prefix}/val/kappa":        metrics.get('cohen_kappa', 0.0),
        f"{prefix}/val/loss":         metrics.get('val_loss', 0.0),
        f"{prefix}/val/precision":    metrics.get('precision', 0.0),
        f"{prefix}/val/recall":       metrics.get('recall', 0.0),
    }

    if 'specificity' in metrics:
        log_dict[f"{prefix}/val/specificity"] = metrics['specificity']

    if 'negative_predictive_value' in metrics:
        log_dict[f"{prefix}/val/npv"] = metrics['negative_predictive_value']

    if 'f1_macro' in metrics:
        log_dict[f"{prefix}/val/f1_macro"] = metrics['f1_macro']

    if train_loss is not None:
        log_dict[f"{prefix}/train/loss"] = train_loss

    if learning_rate is not None:
        log_dict[f"{prefix}/train/lr"] = learning_rate

    if 'f1_per_class' in metrics:
        for i, name in enumerate(class_names):
            if i < len(metrics['f1_per_class']):
                log_dict[f"{prefix}/val/f1_class_{name}"] = metrics['f1_per_class'][i]

    wandb.log(log_dict)

def log_inference_results(
        test_metrics: Dict,
        class_names: List[str],
        is_multiclass: bool,
        cm_fig=None,
        roc_fig=None
):
    if wandb.run is None:
        return

    log_dict = {
        "test/f1":           test_metrics.get('f1_score', 0.0),
        "test/balanced_acc": test_metrics.get('balanced_accuracy', 0.0),
        "test/accuracy":     test_metrics.get('accuracy', 0.0),
        "test/mcc":          test_metrics.get('matthews_correlation_coefficient', 0.0),
        "test/kappa":        test_metrics.get('cohen_kappa', 0.0),
        "test/loss":         test_metrics.get('val_loss', 0.0),
        "test/precision":    test_metrics.get('precision', 0.0),
        "test/recall":       test_metrics.get('recall', 0.0),
    }

    if not is_multiclass:
        log_dict["test/specificity"] = test_metrics.get('specificity', 0.0)
        log_dict["test/npv"] = test_metrics.get('negative_predictive_value', 0.0)
    else:
        log_dict["test/f1_macro"] = test_metrics.get('f1_macro', 0.0)

    if cm_fig is not None:
        log_dict["test/confusion_matrix"] = wandb.Image(cm_fig)

    if roc_fig is not None:
        log_dict["test/roc_curve"] = wandb.Image(roc_fig)

    if class_names:
        per_class_table = create_per_class_metrics_table(test_metrics, class_names)
        log_dict["test/per_class_metrics"] = per_class_table

    scalar_rows = [
        [k.replace("test/", ""), f"{v:.4f}" if isinstance(v, float) else str(v)]
        for k, v in log_dict.items()
        if isinstance(v, (int, float))
    ]
    if scalar_rows:
        log_dict["test/summary_table"] = wandb.Table(
            columns=["Metric", "Value"],
            data=scalar_rows
        )

    wandb.log(log_dict)

def create_per_class_metrics_table(best_metrics: Dict, class_names: List[str]) -> wandb.Table:
    precision_pc = best_metrics.get('precision_per_class', [])
    recall_pc    = best_metrics.get('recall_per_class', [])
    f1_pc        = best_metrics.get('f1_per_class', [])
    support_pc   = best_metrics.get('support_per_class', [])

    data = [
        [
            name,
            f"{precision_pc[i] * 100:.2f}%" if i < len(precision_pc) else "N/A",
            f"{recall_pc[i] * 100:.2f}%"    if i < len(recall_pc)    else "N/A",
            f"{f1_pc[i] * 100:.2f}%"        if i < len(f1_pc)        else "N/A",
            str(support_pc[i])              if i < len(support_pc)   else "N/A",
        ]
        for i, name in enumerate(class_names)
    ]

    return wandb.Table(
        columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        data=data
    )

def init_wandb_run(
        project_name: str,
        run_name: str,
        config: Dict,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
        save_code: bool = False,
        directory: Optional[str] = None
) -> Optional[wandb.Run]:
    load_dotenv()
    api_key = os.getenv('WANDB_API_KEY')
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_CONSOLE'] = 'off'

    if directory is None:
        os.environ['WANDB_DIR'] = os.path.join(os.getcwd(), 'wandb_temp')
        os.environ['WANDB_CACHE_DIR'] = os.path.join(os.getcwd(), 'wandb_temp', 'cache')

    try:
        if not wandb.api.api_key:
            wandb.login(key=api_key, relogin=False)
    except Exception as e:
        print(f"\nErro ao fazer login no WandB: {e}")
        return None

    try:
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            entity=entity,
            tags=tags or [],
            notes=notes,
            group=group,
            resume='allow',
            settings=wandb.Settings(
                start_method='thread',
                console='off',
                quiet=True,
            ),
            save_code=save_code,
            dir=directory
        )
        print(f"\nW&B run: {run.name}")
        print(f"  URL: {run.url}")
        print(f"  Project: {project_name} | Group: {group}")
        print()
        return run
    except Exception as e:
        print(f"\nErro ao inicializar WandB run: {e}")
        print("Continuando sem logging do WandB...\n")
        return None

def finish_wandb_run(quiet: bool = True):
    try:
        if wandb.run is not None:
            wandb.finish(quiet=quiet)
            if not quiet:
                print("\nWandB run finalizado")
    except Exception as e:
        if not quiet:
            print(f"\nErro ao finalizar WandB run: {e}")