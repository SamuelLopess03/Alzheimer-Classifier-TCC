import wandb
import os
import numpy as np
from typing import Dict, List, Optional
from dotenv import load_dotenv

from training.src.evaluation import aggregate_repetition_metrics

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

        print(f"\nW&B run inicializado: {run.name}")
        print(f"  URL: {run.url}")
        print(f"  Project: {project_name}")
        if entity:
            print(f"  Entity: {entity}")
        print()

        return run

    except Exception as e:
        print(f"\nErro ao inicializar WandB run: {e}")
        print("Continuando sem logging do WandB...\n")
        return None

def log_confusion_matrix_figure(
        fig,
        key: str = "confusion_matrix",
        step: Optional[int] = None
):
    if wandb.run is not None:
        wandb.log({key: wandb.Image(fig)}, step=step)

def log_roc_curve_figure(
        fig,
        key: str = "roc_curve",
        step: Optional[int] = None
):
    if wandb.run is not None:
        wandb.log({key: wandb.Image(fig)}, step=step)

def create_repetition_summary_table(
        repetition_results: List[Dict],
        aggregated: Dict,
        is_multiclass: bool = False
) -> wandb.Table:
    repetition_summary_data = []

    for i, rep_result in enumerate(repetition_results):
        best_m = rep_result.get('best_metrics', {})

        if not is_multiclass:
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
        else:
            repetition_summary_data.append({
                'Repetition': str(i + 1),
                'Balanced Acc': f"{best_m.get('balanced_accuracy', 0.0) * 100:.2f}%",
                'Accuracy': f"{best_m.get('accuracy', 0.0) * 100:.2f}%",
                'F1 (Weighted)': f"{rep_result.get('best_f1_score', 0.0) * 100:.2f}%",
                'F1 (Macro)': f"{best_m.get('f1_macro', 0.0) * 100:.2f}%",
                'Precision (W)': f"{best_m.get('precision', 0.0) * 100:.2f}%",
                'Recall (W)': f"{best_m.get('recall', 0.0) * 100:.2f}%",
                'MCC': f"{best_m.get('matthews_correlation_coefficient', 0.0):.4f}",
                'Val Loss': f"{best_m.get('val_loss', 0.0):.4f}",
            })

    if aggregated:
        if not is_multiclass:
            aggregated_row = {
                'Repetition': 'MÉDIA',
                'Balanced Acc': f"{aggregated['mean_balanced_accuracy'] * 100:.2f}% ± {aggregated['std_balanced_accuracy'] * 100:.2f}%",
                'Accuracy': f"{aggregated['mean_accuracy'] * 100:.2f}% ± {aggregated['std_accuracy'] * 100:.2f}%",
                'F1-Score': f"{aggregated['mean_f1'] * 100:.2f}% ± {aggregated['std_f1'] * 100:.2f}%",
                'Specificity': f"{aggregated['mean_specificity'] * 100:.2f}% ± {aggregated['std_specificity'] * 100:.2f}%",
                'Precision': f"{aggregated['mean_precision'] * 100:.2f}% ± {aggregated['std_precision'] * 100:.2f}%",
                'Recall': f"{aggregated['mean_recall'] * 100:.2f}% ± {aggregated['std_recall'] * 100:.2f}%",
                'MCC': f"{aggregated['mean_mcc']:.4f} ± {aggregated['std_mcc']:.4f}",
                'Val Loss': f"{aggregated['mean_loss']:.4f} ± {aggregated['std_loss']:.4f}",
            }
        else:
            aggregated_row = {
                'Repetition': 'MÉDIA',
                'Balanced Acc': f"{aggregated['mean_balanced_accuracy'] * 100:.2f}% ± {aggregated['std_balanced_accuracy'] * 100:.2f}%",
                'Accuracy': f"{aggregated['mean_accuracy'] * 100:.2f}% ± {aggregated['std_accuracy'] * 100:.2f}%",
                'F1 (Weighted)': f"{aggregated['mean_f1'] * 100:.2f}% ± {aggregated['std_f1'] * 100:.2f}%",
                'F1 (Macro)': f"{aggregated.get('mean_f1_macro', 0.0) * 100:.2f}% ± {aggregated.get('std_f1_macro', 0.0) * 100:.2f}%",
                'Precision (W)': f"{aggregated['mean_precision'] * 100:.2f}% ± {aggregated['std_precision'] * 100:.2f}%",
                'Recall (W)': f"{aggregated['mean_recall'] * 100:.2f}% ± {aggregated['std_recall'] * 100:.2f}%",
                'MCC': f"{aggregated['mean_mcc']:.4f} ± {aggregated['std_mcc']:.4f}",
                'Val Loss': f"{aggregated['mean_loss']:.4f} ± {aggregated['std_loss']:.4f}",
            }

        repetition_summary_data.append(aggregated_row)

    table = wandb.Table(
        columns=list(repetition_summary_data[0].keys()),
        data=[list(row.values()) for row in repetition_summary_data]
    )

    return table

def create_detailed_metrics_table(
        best_metrics: Dict,
        is_multiclass: bool = False
) -> wandb.Table:
    if not is_multiclass:
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
                'Description': 'Detecta Non Demented (TNR)'
            },
            {
                'Metric': 'Precision',
                'Value': f"{best_metrics.get('precision', 0.0) * 100:.2f}%",
                'Description': 'Se prediz Demented, acerta X%'
            },
            {
                'Metric': 'Negative Predictive Value',
                'Value': f"{best_metrics.get('negative_predictive_value', 0.0) * 100:.2f}%",
                'Description': 'Se prediz Non Demented, acerta X%'
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
    else:
        detailed_metrics_data = [
            {
                'Metric': 'Balanced Accuracy',
                'Value': f"{best_metrics.get('balanced_accuracy', 0.0) * 100:.2f}%",
                'Description': 'Média balanceada entre classes'
            },
            {
                'Metric': 'Accuracy',
                'Value': f"{best_metrics.get('accuracy', 0.0) * 100:.2f}%",
                'Description': 'Acertos totais'
            },
            {
                'Metric': 'F1-Score (Weighted)',
                'Value': f"{best_metrics.get('f1_score', 0.0) * 100:.2f}%",
                'Description': 'F1 ponderado pela distribuição'
            },
            {
                'Metric': 'F1-Score (Macro)',
                'Value': f"{best_metrics.get('f1_macro', 0.0) * 100:.2f}%",
                'Description': 'F1 média simples entre classes'
            },
            {
                'Metric': 'Precision (Weighted)',
                'Value': f"{best_metrics.get('precision', 0.0) * 100:.2f}%",
                'Description': 'Precision ponderada'
            },
            {
                'Metric': 'Precision (Macro)',
                'Value': f"{best_metrics.get('precision_macro', 0.0) * 100:.2f}%",
                'Description': 'Precision média entre classes'
            },
            {
                'Metric': 'Recall (Weighted)',
                'Value': f"{best_metrics.get('recall', 0.0) * 100:.2f}%",
                'Description': 'Recall ponderado'
            },
            {
                'Metric': 'Recall (Macro)',
                'Value': f"{best_metrics.get('recall_macro', 0.0) * 100:.2f}%",
                'Description': 'Recall médio entre classes'
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

def create_per_class_metrics_table(
        best_metrics: Dict,
        class_names: List[str]
) -> wandb.Table:
    per_class_data = []

    precision_per_class = best_metrics.get('precision_per_class', [])
    recall_per_class = best_metrics.get('recall_per_class', [])
    f1_per_class = best_metrics.get('f1_per_class', [])
    support_per_class = best_metrics.get('support_per_class', [])

    for i, class_name in enumerate(class_names):
        per_class_data.append({
            'Class': class_name,
            'Precision': f"{precision_per_class[i] * 100:.2f}%" if i < len(precision_per_class) else "N/A",
            'Recall': f"{recall_per_class[i] * 100:.2f}%" if i < len(recall_per_class) else "N/A",
            'F1-Score': f"{f1_per_class[i] * 100:.2f}%" if i < len(f1_per_class) else "N/A",
            'Support': str(support_per_class[i]) if i < len(support_per_class) else "N/A"
        })

    table = wandb.Table(
        columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        data=[[row['Class'], row['Precision'], row['Recall'], row['F1-Score'], row['Support']]
              for row in per_class_data]
    )

    return table

def summarize_wandb_repetitions(
        repetition_results: List[Dict],
        params: Dict,
        idx: int,
        is_multiclass: bool = False,
        class_names: Optional[List[str]] = None
) -> Dict:
    aggregated = aggregate_repetition_metrics(repetition_results, is_multiclass)

    repetition_summary_table = create_repetition_summary_table(
        repetition_results,
        aggregated,
        is_multiclass
    )

    val_f1_scores = [r['best_f1_score'] for r in repetition_results]
    best_rep_idx = np.argmax(val_f1_scores)
    best_rep_metrics = repetition_results[best_rep_idx]['best_metrics']

    detailed_metrics_table = create_detailed_metrics_table(
        best_rep_metrics,
        is_multiclass
    )

    log_dict = {
        "tables/repetition_summary": repetition_summary_table,
        "tables/detailed_metrics": detailed_metrics_table,
    }

    if is_multiclass and class_names:
        per_class_table = create_per_class_metrics_table(
            best_rep_metrics,
            class_names
        )
        log_dict["tables/per_class_metrics"] = per_class_table

    if wandb.run is not None:
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

    if is_multiclass and 'mean_f1_macro' in aggregated:
        print(f"  Mean F1 (Macro): {aggregated['mean_f1_macro'] * 100:.2f}% +- {aggregated['std_f1_macro'] * 100:.2f}%")

    return result

def finish_wandb_run(quiet: bool = True):
    try:
        if wandb.run is not None:
            wandb.finish(quiet=quiet)
            if not quiet:
                print("\nWandB run finalizado")
    except Exception as e:
        if not quiet:
            print(f"\nErro ao finalizar WandB run: {e}")