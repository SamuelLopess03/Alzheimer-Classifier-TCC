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
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Optional

def calculate_metrics_model(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        val_loss: float = 0.0,
        train_loss: float = 0.0,
        repetition_number: int = 1,
        epoch_number: int = 1,
        log_to_wandb: bool = True,
        is_multiclass: bool = False
) -> Dict:
    if class_names is None:
        if is_multiclass:
            class_names = ['Mild Dementia', 'Moderate Dementia', 'Very mild Dementia']
        else:
            class_names = ['Demented', 'Non Demented']

    num_classes = len(class_names)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        labels=labels,
        zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average='weighted',
        zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average='macro',
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp, fn, fp, tn = 0, 0, 0, 0

    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    metrics = {
        # Loss
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),

        # Overall metrics
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),

        # Averaged metrics
        'precision': float(precision_weighted),
        'recall': float(recall_weighted),
        'f1_score': float(f1_weighted),

        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),

        # Correlation
        'matthews_correlation_coefficient': float(mcc),
        'cohen_kappa': float(kappa),

        # Per-class details
        'precision_per_class': [float(p) for p in precision],
        'recall_per_class': [float(r) for r in recall],
        'f1_per_class': [float(f) for f in f1],
        'support_per_class': [int(s) for s in support],

        # Confusion matrix
        'confusion_matrix': cm.tolist(),

        # Classification report
        'classification_report': classification_report(
            y_true, y_pred,
            target_names=class_names,
            labels=labels,
            zero_division=0
        )
    }

    if not is_multiclass and num_classes == 2:
        tp, fn, fp, tn = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        metrics.update({
            'specificity': float(specificity),
            'negative_predictive_value': float(npv),
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
        })

    else:
        specificities = []
        npvs = []

        for i in range(num_classes):
            tn = np.sum((y_true != i) & (y_pred != i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            tp = np.sum((y_true == i) & (y_pred == i))

            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

            specificities.append(float(spec))
            npvs.append(float(npv))

        total_samples = len(y_true)
        weighted_spec = sum(
            specificities[i] * np.sum(y_true == i) / total_samples
            for i in range(num_classes)
        )
        weighted_npv = sum(
            npvs[i] * np.sum(y_true == i) / total_samples
            for i in range(num_classes)
        )

        macro_spec = float(np.mean(specificities))
        macro_npv = float(np.mean(npvs))

        metrics.update({
            'specificity': float(weighted_spec),
            'specificity_macro': macro_spec,
            'specificity_per_class': specificities,
            'negative_predictive_value': float(weighted_npv),
            'npv_macro': macro_npv,
            'npv_per_class': npvs,
        })

    if log_to_wandb and wandb.run is not None:
        wandb_log = {
            f"rep_{repetition_number}/epoch": epoch_number,

            # Loss
            f"rep_{repetition_number}/train_loss": train_loss,
            f"rep_{repetition_number}/val_loss": val_loss,

            # Overall
            f"rep_{repetition_number}/accuracy": accuracy,
            f"rep_{repetition_number}/balanced_accuracy": balanced_acc,

            # Averaged metrics
            f"rep_{repetition_number}/precision_weighted": precision_weighted,
            f"rep_{repetition_number}/recall_weighted": recall_weighted,
            f"rep_{repetition_number}/f1_weighted": f1_weighted,

            f"rep_{repetition_number}/precision_macro": precision_macro,
            f"rep_{repetition_number}/recall_macro": recall_macro,
            f"rep_{repetition_number}/f1_macro": f1_macro,

            # Correlation
            f"rep_{repetition_number}/matthews_correlation_coefficient": mcc,
            f"rep_{repetition_number}/cohen_kappa": kappa,
        }

        if 'specificity' in metrics:
            wandb_log[f"rep_{repetition_number}/specificity"] = metrics['specificity']

        if not is_multiclass and num_classes == 2:
            wandb_log.update({
                f"rep_{repetition_number}/true_negatives": int(tn),
                f"rep_{repetition_number}/false_positives": int(fp),
                f"rep_{repetition_number}/false_negatives": int(fn),
                f"rep_{repetition_number}/true_positives": int(tp),
            })

        for i, class_name in enumerate(class_names):
            wandb_log[f"rep_{repetition_number}/precision_{class_name}"] = precision[i]
            wandb_log[f"rep_{repetition_number}/recall_{class_name}"] = recall[i]
            wandb_log[f"rep_{repetition_number}/f1_{class_name}"] = f1[i]

        wandb.log(wandb_log)

    return metrics

def calculate_roc_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        is_multiclass: bool = False
) -> Dict:
    if not is_multiclass:
        if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
            if y_pred_proba.shape[1] == 2:
                proba_class_0 = y_pred_proba[:, 0]
            else:
                proba_class_0 = y_pred_proba.flatten()
        else:
            proba_class_0 = y_pred_proba

        fpr, tpr, thresholds = roc_curve(y_true, proba_class_0)
        roc_auc = auc(fpr, tpr)

        # Threshold ótimo (Youden's Index)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        return {
            'auc_roc': float(roc_auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'optimal_threshold': float(optimal_threshold),
            'optimal_fpr': float(optimal_fpr),
            'optimal_tpr': float(optimal_tpr),
            'optimal_idx': int(optimal_idx)
        }

    else:
        num_classes = y_pred_proba.shape[1]

        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

        roc_metrics = {}

        for i in range(num_classes):
            fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            roc_metrics[f'class_{i}'] = {
                'auc_roc': float(roc_auc),
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }

        try:
            macro_roc_auc = roc_auc_score(y_true_bin, y_pred_proba, average='macro')
            roc_metrics['macro_auc'] = float(macro_roc_auc)
        except Exception as e:
            print(f"Error ao calcular Macro AUC-ROC: {e}\n")
            roc_metrics['macro_auc'] = 0.0

        try:
            weighted_roc_auc = roc_auc_score(y_true_bin, y_pred_proba, average='weighted')
            roc_metrics['weighted_auc'] = float(weighted_roc_auc)
        except Exception as e:
            print(f"Error ao calcular Weighted AUC-ROC: {e}\n")
            roc_metrics['weighted_auc'] = 0.0

        return roc_metrics

def calculate_combined_score(
        aggregated_metrics: Dict,
        is_multiclass: bool = False
) -> float:
    if not is_multiclass:
        weights = {
            'f1': 0.35,
            'balanced_acc': 0.25,
            'recall': 0.15,
            'specificity': 0.15,
            'mcc': 0.10,
        }

        stability_weight = 0.15

        f1_component = aggregated_metrics.get('mean_f1', 0.0) * weights['f1']
        balanced_acc_component = aggregated_metrics.get('mean_balanced_accuracy', 0.0) * weights['balanced_acc']
        recall_component = aggregated_metrics.get('mean_recall', 0.0) * weights['recall']
        specificity_component = aggregated_metrics.get('mean_specificity', 0.0) * weights['specificity']

        mcc_normalized = (aggregated_metrics.get('mean_mcc', 0.0) + 1) / 2
        mcc_component = mcc_normalized * weights['mcc']

        stability_penalty = (aggregated_metrics.get('std_f1', 0.0) * 0.4 +
                             aggregated_metrics.get('std_balanced_accuracy', 0.0) * 0.3 +
                             aggregated_metrics.get('std_recall', 0.0) * 0.15 +
                             aggregated_metrics.get('std_specificity', 0.0) * 0.15
                            ) * stability_weight

        score = (
                f1_component +
                balanced_acc_component +
                recall_component +
                specificity_component +
                mcc_component -
                stability_penalty
        )

    else:
        weights = {
            'f1_macro': 0.30,
            'f1_weighted': 0.20,
            'balanced_acc': 0.25,
            'recall_macro': 0.15,
            'mcc': 0.10,
        }

        stability_weight = 0.15

        f1_macro_component = aggregated_metrics.get('mean_f1_macro', 0.0) * weights['f1_macro']
        f1_weighted_component = aggregated_metrics.get('mean_f1', 0.0) * weights['f1_weighted']
        balanced_acc_component = aggregated_metrics.get('mean_balanced_accuracy', 0.0) * weights['balanced_acc']
        recall_component = aggregated_metrics.get('mean_recall_macro', 0.0) * weights['recall_macro']

        mcc_normalized = (aggregated_metrics.get('mean_mcc', 0.0) + 1) / 2
        mcc_component = mcc_normalized * weights['mcc']

        stability_penalty = (aggregated_metrics.get('std_f1_macro', 0.0) * 0.4 +
                             aggregated_metrics.get('std_balanced_accuracy', 0.0) * 0.3 +
                             aggregated_metrics.get('std_recall_macro', 0.0) * 0.3
                            ) * stability_weight

        score = (
                f1_macro_component +
                f1_weighted_component +
                balanced_acc_component +
                recall_component +
                mcc_component -
                stability_penalty
        )

    return float(score)

def aggregate_repetition_metrics(
        repetition_results: List[Dict],
        is_multiclass: bool = False
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

        # Recall
        'mean_recall': float(np.mean([m['recall'] for m in best_metrics_list])),
        'std_recall': float(np.std([m['recall'] for m in best_metrics_list])),

        # Loss
        'mean_loss': float(np.mean([m['val_loss'] for m in best_metrics_list])),
        'std_loss': float(np.std([m['val_loss'] for m in best_metrics_list])),

        # MCC
        'mean_mcc': float(np.mean([m.get('matthews_correlation_coefficient', 0.0) for m in best_metrics_list])),
        'std_mcc': float(np.std([m.get('matthews_correlation_coefficient', 0.0) for m in best_metrics_list])),

        # Repetitions
        'n_repetitions': len(repetition_results)
    }

    if is_multiclass:
        if 'precision_macro' in best_metrics_list[0]:
            aggregated.update({
                'mean_precision_macro': float(np.mean([m['precision_macro'] for m in best_metrics_list])),
                'std_precision_macro': float(np.std([m['precision_macro'] for m in best_metrics_list])),

                'mean_recall_macro': float(np.mean([m['recall_macro'] for m in best_metrics_list])),
                'std_recall_macro': float(np.std([m['recall_macro'] for m in best_metrics_list])),

                'mean_f1_macro': float(np.mean([m['f1_macro'] for m in best_metrics_list])),
                'std_f1_macro': float(np.std([m['f1_macro'] for m in best_metrics_list])),
            })

    if 'specificity' in best_metrics_list[0]:
        aggregated.update({
            'mean_specificity': float(np.mean([m['specificity'] for m in best_metrics_list])),
            'std_specificity': float(np.std([m['specificity'] for m in best_metrics_list])),
        })

    return aggregated