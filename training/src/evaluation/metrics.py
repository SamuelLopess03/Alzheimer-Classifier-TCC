import numpy as np
import wandb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
    roc_curve,
    auc
)
from typing import Dict, List, Tuple, Optional

def calculate_binary_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        val_loss: float = 0.0,
        train_loss: float = 0.0,
        repetition_number: int = 1,
        epoch_number: int = 1,
        log_to_wandb: bool = True
) -> Dict:
    if class_names is None:
        class_names = ['Demented', 'NonDemented']

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        labels=[0, 1],
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp, fn, fp, tn = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate

    precision_binary = precision[0]
    recall_binary = recall[0]
    f1_binary = f1[0]

    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    metrics = {
        # Loss
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),

        # Overall metrics
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),

        # Binary metrics (class 0: Demented)
        'precision': float(precision_binary),
        'recall': float(recall_binary),
        'f1_score': float(f1_binary),

        # Medical metrics
        'specificity': float(specificity),
        'negative_predictive_value': float(npv),

        # Error rates
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),

        # Correlation
        'matthews_correlation_coefficient': float(mcc),
        'cohen_kappa': float(kappa),

        # Per-class details
        'precision_per_class': [float(p) for p in precision],
        'recall_per_class': [float(r) for r in recall],
        'f1_per_class': [float(f) for f in f1],
        'support_per_class': [int(s) for s in support],

        # Confusion matrix components
        'confusion_matrix': cm.tolist(),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp),

        # Classification report
        'classification_report': classification_report(
            y_true, y_pred,
            target_names=class_names,
            labels=[0, 1],
            zero_division=0
        )
    }

    if log_to_wandb:
        wandb_log = {
            f"rep_{repetition_number}/epoch": epoch_number,

            # Loss
            f"rep_{repetition_number}/train_loss": train_loss,
            f"rep_{repetition_number}/val_loss": val_loss,

            # Overall
            f"rep_{repetition_number}/accuracy": accuracy,
            f"rep_{repetition_number}/balanced_accuracy": balanced_acc,

            # Binary
            f"rep_{repetition_number}/precision": precision_binary,
            f"rep_{repetition_number}/recall": recall_binary,
            f"rep_{repetition_number}/f1_score": f1_binary,

            # Medical
            f"rep_{repetition_number}/specificity": specificity,
            f"rep_{repetition_number}/negative_predictive_value": npv,

            # Correlation
            f"rep_{repetition_number}/matthews_correlation_coefficient": mcc,
            f"rep_{repetition_number}/cohen_kappa": kappa,

            # Confusion matrix
            f"rep_{repetition_number}/true_negatives": int(tn),
            f"rep_{repetition_number}/false_positives": int(fp),
            f"rep_{repetition_number}/false_negatives": int(fn),
            f"rep_{repetition_number}/true_positives": int(tp),
        }

        wandb.log(wandb_log)

    return metrics

def calculate_roc_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
) -> Dict:
    if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
        if y_pred_proba.shape[1] == 2:
            proba_class_0 = y_pred_proba[:, 0]
        else:
            proba_class_0 = y_pred_proba.flatten()
    else:
        proba_class_0 = y_pred_proba

    fpr, tpr, thresholds = roc_curve(y_true, proba_class_0)
    roc_auc = auc(fpr, tpr)

    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]

    roc_metrics = {
        'auc_roc': float(roc_auc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'optimal_threshold': float(optimal_threshold),
        'optimal_fpr': float(optimal_fpr),
        'optimal_tpr': float(optimal_tpr),
        'optimal_idx': optimal_idx
    }

    return roc_metrics

def calculate_combined_score(aggregated_metrics: Dict) -> float:
    weights = {
        'f1': 0.35,
        'balanced_acc': 0.25,
        'recall': 0.15,
        'specificity': 0.15,
        'mcc': 0.10,
    }

    stability_weight = 0.15

    f1_component = aggregated_metrics.get('mean_f1', 0.0) * weights['f1']

    balanced_acc_component = (
            aggregated_metrics.get('mean_balanced_accuracy', 0.0) * weights['balanced_acc']
    )

    recall_component = aggregated_metrics.get('mean_recall', 0.0) * weights['recall']

    specificity_component = (
            aggregated_metrics.get('mean_specificity', 0.0) * weights['specificity']
    )

    # Normalize MCC from [-1, 1] to [0, 1]
    mcc_normalized = (aggregated_metrics.get('mean_mcc', 0.0) + 1) / 2
    mcc_component = mcc_normalized * weights['mcc']

    stability_penalty = (aggregated_metrics.get('std_f1', 0.0) * 0.4 +
                         aggregated_metrics.get('std_balanced_accuracy', 0.0) * 0.3 +
                         aggregated_metrics.get('std_recall', 0.0) * 0.15 +
                         aggregated_metrics.get('std_specificity', 0.0) * 0.15) * stability_weight

    score = (
            f1_component +
            balanced_acc_component +
            recall_component +
            specificity_component +
            mcc_component -
            stability_penalty
    )

    return float(score)

def print_metrics_summary(
        metrics: Dict,
        class_names: Optional[List[str]] = None,
        title: str = "METRICS SUMMARY"
):
    if class_names is None:
        class_names = ['Demented', 'NonDemented']

    print(f"\n{'-' * 60}")
    print(f"{title}")
    print(f"{'-' * 60}\n")

    print("Overall Metrics:")
    print(f"   Accuracy:          {metrics['accuracy'] * 100:>6.2f}%")
    print(f"   Balanced Accuracy: {metrics['balanced_accuracy'] * 100:>6.2f}%")
    print(f"   F1-Score:          {metrics['f1_score'] * 100:>6.2f}%\n")

    print("Medical Metrics:")
    print(f"   Sensitivity (Recall): {metrics['recall'] * 100:>6.2f}%  (detects Demented)")
    print(f"   Specificity:          {metrics['specificity'] * 100:>6.2f}%  (detects NonDemented)")
    print(f"   Precision:            {metrics['precision'] * 100:>6.2f}%")
    print(f"   NPV:                  {metrics['negative_predictive_value'] * 100:>6.2f}%\n")

    print("Correlation:")
    print(f"   Matthews Corr:  {metrics['matthews_correlation_coefficient']:>7.4f}")
    print(f"   Cohen's Kappa:  {metrics['cohen_kappa']:>7.4f}\n")

    print("Confusion Matrix:")
    cm = metrics['confusion_matrix']
    tp, fn = cm[1]
    fp, tn = cm[0]

    print(f"                    Predicted")
    print(f"                 Dem    NonDem")
    print(f"   Actual  Dem   {tp:4d}   {fn:4d}")
    print(f"         NonDem  {fp:4d}   {tn:4d}\n")

    print("Clinical Interpretation:")
    sens = metrics['recall'] * 100
    spec = metrics['specificity'] * 100
    print(f"   Of 100 patients WITH dementia:    ~{sens:.0f} detected")
    print(f"   Of 100 patients WITHOUT dementia: ~{spec:.0f} identified correctly")

    print(f"\n{'-' * 60}\n")

def aggregate_repetition_metrics(
        repetition_results: List[Dict]
) -> Dict:
    if not repetition_results:
        return {}

    val_f1_scores = [r['best_f1_score'] for r in repetition_results]
    best_metrics_list = [
        r['best_metrics'] for r in repetition_results
        if r.get('best_metrics')
    ]

    if not best_metrics_list:
        return {'mean_f1': 0.0, 'std_f1': 0.0}

    aggregated = {
        # F1-Score
        'mean_f1': float(np.mean(val_f1_scores)),
        'std_f1': float(np.std(val_f1_scores)),

        # Accuracy
        'mean_accuracy': float(np.mean([m['accuracy'] for m in best_metrics_list])),
        'std_accuracy': float(np.std([m['accuracy'] for m in best_metrics_list])),

        # Balanced Accuracy
        'mean_balanced_accuracy': float(np.mean([m['balanced_accuracy'] for m in best_metrics_list])),
        'std_balanced_accuracy': float(np.std([m['balanced_accuracy'] for m in best_metrics_list])),

        # Precision
        'mean_precision': float(np.mean([m['precision'] for m in best_metrics_list])),
        'std_precision': float(np.std([m['precision'] for m in best_metrics_list])),

        # Recall (Sensitivity)
        'mean_recall': float(np.mean([m['recall'] for m in best_metrics_list])),
        'std_recall': float(np.std([m['recall'] for m in best_metrics_list])),

        # Specificity
        'mean_specificity': float(np.mean([m['specificity'] for m in best_metrics_list])),
        'std_specificity': float(np.std([m['specificity'] for m in best_metrics_list])),

        # Loss
        'mean_loss': float(np.mean([m['val_loss'] for m in best_metrics_list])),
        'std_loss': float(np.std([m['val_loss'] for m in best_metrics_list])),

        # Matthews Correlation
        'mean_mcc': float(np.mean([m.get('matthews_correlation_coefficient', 0.0) for m in best_metrics_list])),
        'std_mcc': float(np.std([m.get('matthews_correlation_coefficient', 0.0) for m in best_metrics_list])),

        # Number of repetitions
        'n_repetitions': len(repetition_results)
    }

    return aggregated

def print_aggregated_metrics(
        aggregated: Dict,
        title: str = "AGGREGATED METRICS"
):
    print(f"\n{'-' * 60}")
    print(f"{title}")
    print(f"{'-' * 60}\n")

    print(f"Repetitions: {aggregated.get('n_repetitions', 0)}\n")

    metrics_to_show = [
        ('Balanced Accuracy', 'mean_balanced_accuracy', 'std_balanced_accuracy'),
        ('F1-Score', 'mean_f1', 'std_f1'),
        ('Sensitivity (Recall)', 'mean_recall', 'std_recall'),
        ('Specificity', 'mean_specificity', 'std_specificity'),
        ('Precision', 'mean_precision', 'std_precision'),
        ('Matthews Correlation', 'mean_mcc', 'std_mcc'),
        ('Validation Loss', 'mean_loss', 'std_loss')
    ]

    for metric_name, mean_key, std_key in metrics_to_show:
        mean_val = aggregated.get(mean_key, 0.0)
        std_val = aggregated.get(std_key, 0.0)

        if 'Loss' in metric_name:
            print(f"  {metric_name:<25}: {mean_val:>7.4f} ± {std_val:>7.4f}")
        else:
            print(f"  {metric_name:<25}: {mean_val * 100:>6.2f}% ± {std_val * 100:>6.2f}%")

    combined = calculate_combined_score(aggregated)
    print(f"\n  {'Combined Score':<25}: {combined:>7.4f}")

    print(f"\n{'-' * 60}\n")