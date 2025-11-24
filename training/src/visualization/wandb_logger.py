import wandb
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

from training.src.evaluation import aggregate_repetition_metrics

def init_wandb_run(
        project_name: str,
        run_name: str,
        config: Dict,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None
) -> wandb.Run:
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        entity=entity,
        tags=tags or [],
        notes=notes,
        group=group,
        reinit=True
    )

    print(f"\nW&B run initialized: {run.name}")
    print(f"  URL: {run.url}\n")

    return run

def log_metrics(
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True
):
    wandb.log(metrics, step=step, commit=commit)

def log_confusion_matrix_figure(
        fig,
        key: str = "confusion_matrix",
        step: Optional[int] = None
):
    wandb.log({key: wandb.Image(fig)}, step=step)

def log_roc_curve_figure(
        fig,
        key: str = "roc_curve",
        step: Optional[int] = None
):
    wandb.log({key: wandb.Image(fig)}, step=step)

def create_repetition_summary_table(
        repetition_results: List[Dict],
        aggregated: Dict
) -> wandb.Table:
    repetition_summary_data = []

    for i, rep_result in enumerate(repetition_results):
        best_m = rep_result.get('best_metrics', {})

        repetition_summary_data.append({
            'Repetition': str(i + 1),
            'Balanced Acc': f"{best_m.get('balanced_accuracy', 0.0) * 100:.2f}%",
            'Accuracy': f"{best_m.get('accuracy', 0.0) * 100:.2f}%",
            'F1-Score': f"{rep_result.get('best_f1_score', 0.0) * 100:.2f}%",
            'Specificity': f"{best_m.get('specificity', 0.0) * 100:.2f}%",
            'Precision': f"{best_m.get('precision', 0.0) * 100:.2f}%",
            'Recall': f"{best_m.get('recall', 0.0) * 100:.2f}%",
            'MCC': f"{best_m.get('matthews_correlation_coefficient', 0.0):.4f}",
            'Val Loss': f"{best_m.get('val_loss', 0.0):.4f}",
        })

    if aggregated:
        aggregated_row = {
            'Repetition': 'MÉDIA',
            'Balanced Acc': f"{aggregated['mean_balanced_accuracy'] * 100:.2f}% +- {aggregated['std_balanced_accuracy'] * 100:.2f}%",
            'Accuracy': f"{aggregated['mean_accuracy'] * 100:.2f}% +- {aggregated['std_accuracy'] * 100:.2f}%",
            'F1-Score': f"{aggregated['mean_f1'] * 100:.2f}% +- {aggregated['std_f1'] * 100:.2f}%",
            'Specificity': f"{aggregated['mean_specificity'] * 100:.2f}% +- {aggregated['std_specificity'] * 100:.2f}%",
            'Precision': f"{aggregated['mean_precision'] * 100:.2f}% +- {aggregated['std_precision'] * 100:.2f}%",
            'Recall': f"{aggregated['mean_recall'] * 100:.2f}% +- {aggregated['std_recall'] * 100:.2f}%",
            'MCC': f"{aggregated['mean_mcc']:.4f} +- {aggregated['std_mcc']:.4f}",
            'Val Loss': f"{aggregated['mean_loss']:.4f} +- {aggregated['std_loss']:.4f}",
        }

        repetition_summary_data.append(aggregated_row)

    table = wandb.Table(
        columns=list(repetition_summary_data[0].keys()),
        data=[list(row.values()) for row in repetition_summary_data]
    )

    return table

def create_detailed_metrics_table(
        best_metrics: Dict
) -> wandb.Table:
    detailed_metrics_data = [
        {
            'Metric': 'Balanced Accuracy',
            'Value': f"{best_metrics.get('balanced_accuracy', 0.0) * 100:.2f}%",
            'Description': 'Média de Sensitivity e Specificity'
        },
        {
            'Metric': 'Accuracy',
            'Value': f"{best_metrics.get('accuracy', 0.0) * 100:.2f}%",
            'Description': 'Acertos totais'
        },
        {
            'Metric': 'Sensitivity (Recall)',
            'Value': f"{best_metrics.get('recall', 0.0) * 100:.2f}%",
            'Description': 'Detecta Demented (TPR)'
        },
        {
            'Metric': 'Specificity',
            'Value': f"{best_metrics.get('specificity', 0.0) * 100:.2f}%",
            'Description': 'Detecta NonDemented (TNR)'
        },
        {
            'Metric': 'Precision',
            'Value': f"{best_metrics.get('precision', 0.0) * 100:.2f}%",
            'Description': 'Se prediz Demented, acerta X%'
        },
        {
            'Metric': 'Negative Predictive Value',
            'Value': f"{best_metrics.get('negative_predictive_value', 0.0) * 100:.2f}%",
            'Description': 'Se prediz NonDemented, acerta X%'
        },
        {
            'Metric': 'F1-Score',
            'Value': f"{best_metrics.get('f1_score', 0.0) * 100:.2f}%",
            'Description': 'Harmônica de Precision e Recall'
        },
        {
            'Metric': 'MCC',
            'Value': f"{best_metrics.get('matthews_correlation_coefficient', 0.0):.4f}",
            'Description': 'Matthews Correlation Coef (-1 a 1)'
        },
        {
            'Metric': "Cohen's Kappa",
            'Value': f"{best_metrics.get('cohen_kappa', 0.0):.4f}",
            'Description': 'Concordância considerando chance'
        },
    ]

    table = wandb.Table(
        columns=['Metric', 'Value', 'Description'],
        data=[[row['Metric'], row['Value'], row['Description']] for row in detailed_metrics_data]
    )

    return table

def summarize_wandb_repetitions(
        repetition_results: List[Dict],
        params: Dict,
        idx: int
) -> Dict:
    print(f"{'-' * 60}")
    print(f"AGREGANDO RESULTADOS DA COMBINAÇÃO #{idx + 1}")
    print(f"{'-' * 60}\n")

    aggregated = aggregate_repetition_metrics(repetition_results)

    repetition_summary_table = create_repetition_summary_table(
        repetition_results,
        aggregated
    )

    val_f1_scores = [r['best_f1_score'] for r in repetition_results]
    best_rep_idx = np.argmax(val_f1_scores)
    best_rep_metrics = repetition_results[best_rep_idx]['best_metrics']

    detailed_metrics_table = create_detailed_metrics_table(best_rep_metrics)

    log_dict = {
        "tables/repetition_summary": repetition_summary_table,
        "tables/detailed_metrics": detailed_metrics_table,
    }

    wandb.log(log_dict)

    result = {
        'params': params,
        'combination_index': idx,
        'aggregated': aggregated,
        'n_successful_repetitions': len(repetition_results),
        'best_repetition_index': int(best_rep_idx),
        'best_repetition_f1_score': float(val_f1_scores[best_rep_idx])
    }

    print(f"\nResultados agregados e logados no W&B")
    print(f"  Mean F1: {aggregated['mean_f1'] * 100:.2f}% +- {aggregated['std_f1'] * 100:.2f}%")
    print(
        f"  Mean Balanced Acc: {aggregated['mean_balanced_accuracy'] * 100:.2f}% +- {aggregated['std_balanced_accuracy'] * 100:.2f}%")
    print(f"{'-' * 60}\n")

    return result

def finish_wandb_run(
        summary_dict: Optional[Dict] = None
):
    if summary_dict:
        for key, value in summary_dict.items():
            wandb.run.summary[key] = value

    wandb.finish()
    print("\nW&B run finished\n")

def log_model_artifact(
        model_path: Path,
        artifact_name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict] = None
):
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=metadata or {}
    )

    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)

    print(f"\nModel artifact logged: {artifact_name}\n")

def log_config_artifact(
        config_path: Path,
        artifact_name: str = "config",
        artifact_type: str = "config"
):
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type
    )

    artifact.add_file(str(config_path))
    wandb.log_artifact(artifact)

    print(f"\nConfig artifact logged: {artifact_name}\n")