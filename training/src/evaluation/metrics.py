import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

DEFAULT_BINARY_CLASSES = ['Non Demented', 'Demented']
DEFAULT_MULTICLASS_CLASSES = ['Very mild Dementia', 'Mild+Moderate Dementia']

BINARY_SCORE_WEIGHTS = {
    'f1': 0.35,
    'balanced_acc': 0.25,
    'recall': 0.15,
    'specificity': 0.15,
    'mcc': 0.10,
    'std_penalty': 0.50
}

MULTICLASS_SCORE_WEIGHTS = {
    'f1_macro': 0.30,
    'f1_weighted': 0.20,
    'balanced_acc': 0.25,
    'recall_macro': 0.15,
    'mcc': 0.10,
    'std_penalty': 0.60
}

def _aggregate_predictions_by_subject(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    subject_ids: List[str],
    top_k: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    unique_subjects = list(dict.fromkeys(subject_ids))
    subj_y_true = []
    subj_y_pred = []
    
    subject_ids = np.array(subject_ids)
    
    for subj in unique_subjects:
        indices = np.where(subject_ids == subj)[0]
        subj_y_true.append(y_true[indices[0]])
        
        subj_probs = y_prob[indices]
        num_slices = len(indices)
        k = min(top_k, num_slices)
        
        dementia_class_idx = 1 if subj_probs.shape[1] > 1 else 0
        dementia_probs = subj_probs[:, dementia_class_idx]
        top_k_idx = np.argsort(dementia_probs)[::-1][:k]
        
        avg_prob = np.mean(subj_probs[top_k_idx], axis=0)
        subj_y_pred.append(np.argmax(avg_prob))
        
    return np.array(subj_y_true), np.array(subj_y_pred)

def _aggregate_probabilities_by_subject(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subject_ids: List[str],
    top_k: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    unique_subjects = list(dict.fromkeys(subject_ids))
    subj_y_true = []
    subj_y_prob = []
    
    subject_ids = np.array(subject_ids)
    
    for subj in unique_subjects:
        indices = np.where(subject_ids == subj)[0]
        subj_y_true.append(y_true[indices[0]])
        
        subj_probs = y_prob[indices]
        num_slices = len(indices)
        k = min(top_k, num_slices)
        
        if subj_probs.ndim == 1 or (subj_probs.ndim == 2 and subj_probs.shape[1] == 1):
            flat_probs = subj_probs.flatten()
            top_k_idx = np.argsort(flat_probs)[::-1][:k]
            avg_prob = np.mean(flat_probs[top_k_idx])
            subj_y_prob.append(avg_prob)
        else:
            dementia_class_idx = 1 if subj_probs.shape[1] > 1 else 0
            dementia_probs = subj_probs[:, dementia_class_idx]
            top_k_idx = np.argsort(dementia_probs)[::-1][:k]
            avg_prob = np.mean(subj_probs[top_k_idx], axis=0)
            subj_y_prob.append(avg_prob)
            
    return np.array(subj_y_true), np.array(subj_y_prob)

def _compute_binary_error_metrics(cm: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = cm.ravel()
    
    eps = 1e-10
    specificity = tn / (tn + fp + eps)
    npv = tn / (tn + fn + eps)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)

    return {
        'specificity': float(specificity),
        'negative_predictive_value': float(npv),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp),
    }

def _compute_multiclass_error_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, Any]:
    specificities = []
    npvs = []
    eps = 1e-10
    
    for i in range(num_classes):
        tn = np.sum((y_true != i) & (y_pred != i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        tp = np.sum((y_true == i) & (y_pred == i))
        
        specificities.append(tn / (tn + fp + eps))
        npvs.append(tn / (tn + fn + eps))
    
    mean_spec = np.mean(specificities)
    mean_npv = np.mean(npvs)
    
    return {
        'specificity': float(mean_spec),
        'specificity_macro': float(mean_spec),
        'negative_predictive_value': float(mean_npv),
        'npv_macro': float(mean_npv),
        'specificity_per_class': [float(s) for s in specificities],
        'npv_per_class': [float(n) for n in npvs]
    }

def _compute_base_error_metrics(cm: np.ndarray, is_multiclass: bool, y_true=None, y_pred=None) -> Dict[str, Any]:
    if not is_multiclass and cm.shape == (2, 2):
        return _compute_binary_error_metrics(cm)

    return _compute_multiclass_error_metrics(y_true, y_pred, cm.shape[0])

def evaluate_performance(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        subject_ids: List[str],
        class_names: Optional[List[str]] = None,
        val_loss: float = 0.0,
        is_multiclass: bool = False,
        fold_number: int = 1,
        epoch_number: int = 1
) -> Dict[str, Any]:
    if len(y_true) != len(subject_ids):
        raise ValueError(f"Dimensões incompatíveis: y_true ({len(y_true)}) != subject_ids ({len(subject_ids)})")

    y_subj_true, y_subj_pred = _aggregate_predictions_by_subject(y_true, y_prob, subject_ids, top_k=20)
    
    if class_names is None:
        class_names = DEFAULT_MULTICLASS_CLASSES if is_multiclass else DEFAULT_BINARY_CLASSES
    
    num_classes = len(class_names)
    labels = list(range(num_classes))

    accuracy = accuracy_score(y_subj_true, y_subj_pred)
    balanced_acc = balanced_accuracy_score(y_subj_true, y_subj_pred)
    mcc = matthews_corrcoef(y_subj_true, y_subj_pred)
    kappa = cohen_kappa_score(y_subj_true, y_subj_pred)

    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_subj_true, y_subj_pred, average=None, labels=labels, zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_subj_true, y_subj_pred, average='weighted', labels=labels, zero_division=0)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_subj_true, y_subj_pred, average='macro', labels=labels, zero_division=0)

    cm = confusion_matrix(y_subj_true, y_subj_pred, labels=labels)
    error_metrics = _compute_base_error_metrics(cm, is_multiclass, y_subj_true, y_subj_pred)
    
    metrics = {
        'fold': int(fold_number),
        'epoch': int(epoch_number),
        'val_loss': float(val_loss),
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'matthews_correlation_coefficient': float(mcc),
        'cohen_kappa': float(kappa),
        'precision': float(p_w),
        'recall': float(r_w),
        'f1_score': float(f1_w),
        'precision_macro': float(p_m),
        'recall_macro': float(r_m),
        'f1_macro': float(f1_m),
        'precision_per_class': [float(p) for p in p_class],
        'recall_per_class': [float(r) for r in r_class],
        'f1_per_class': [float(f) for f in f1_class],
        'support_per_class': [int(s) for s in support],
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            y_subj_true, y_subj_pred, target_names=class_names, labels=labels, zero_division=0)
    }
    metrics.update(error_metrics)

    return metrics

def _roc_metrics_multiclass(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
    num_classes = y_pred_proba.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    
    if num_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.concatenate((1 - y_true_bin, y_true_bin), axis=1)
        
    roc_metrics = {}

    for i in range(num_classes):
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_metrics[f'class_{i}'] = {
            'auc_roc': float(auc(fpr, tpr)),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }

    try:
        roc_metrics['macro_auc'] = float(roc_auc_score(y_true_bin, y_pred_proba, average='macro'))
        roc_metrics['weighted_auc'] = float(roc_auc_score(y_true_bin, y_pred_proba, average='weighted'))
    except Exception as e:
        roc_metrics['macro_auc'] = 0.0
        roc_metrics['weighted_auc'] = 0.0

    return roc_metrics

def _roc_metrics_binary(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
    if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
        proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba.flatten()
    else:
        proba = y_pred_proba

    fpr, tpr, thresholds = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)

    return {
        'auc_roc': float(roc_auc),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'optimal_threshold': float(thresholds[optimal_idx]),
        'optimal_fpr': float(fpr[optimal_idx]),
        'optimal_tpr': float(tpr[optimal_idx]),
        'optimal_idx': int(optimal_idx)
    }

def calculate_roc_metrics(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    is_multiclass: bool = False,
    subject_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    if subject_ids is not None:
        y_true, y_pred_proba = _aggregate_probabilities_by_subject(y_true, y_pred_proba, subject_ids, top_k=20)

    num_classes = y_pred_proba.shape[1] if (isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2) else 2
    if is_multiclass and num_classes > 2:
        return _roc_metrics_multiclass(y_true, y_pred_proba)

    return _roc_metrics_binary(y_true, y_pred_proba)

def _calculate_combined_score_multiclass(metrics: Dict[str, float]) -> float:
    w = MULTICLASS_SCORE_WEIGHTS
    
    score = (
        metrics.get('mean_f1_macro', 0.0) * w['f1_macro'] + 
        metrics.get('mean_f1', 0.0) * w['f1_weighted'] + 
        metrics.get('mean_balanced_accuracy', 0.0) * w['balanced_acc'] + 
        metrics.get('mean_recall_macro', 0.0) * w['recall_macro'] + 
        ((metrics.get('mean_mcc', 0.0) + 1) / 2) * w['mcc']
    )

    penalty = (
        metrics.get('std_f1_macro', 0.0) * 0.3 + 
        metrics.get('std_balanced_accuracy', 0.0) * 0.3 + 
        metrics.get('std_precision_macro', 0.0) * 0.2 + 
        metrics.get('std_mcc', 0.0) * 0.2
    ) * w['std_penalty']
    
    return float(score - penalty)

def _calculate_combined_score_binary(metrics: Dict[str, float]) -> float:
    w = BINARY_SCORE_WEIGHTS
    
    score = (
        metrics.get('mean_f1', 0.0) * w['f1'] + 
        metrics.get('mean_balanced_accuracy', 0.0) * w['balanced_acc'] + 
        metrics.get('mean_recall', 0.0) * w['recall'] + 
        metrics.get('mean_specificity', 0.0) * w['specificity'] + 
        ((metrics.get('mean_mcc', 0.0) + 1) / 2) * w['mcc']
    )
    
    penalty = (
        metrics.get('std_f1', 0.0) * 0.4 + 
        metrics.get('std_balanced_accuracy', 0.0) * 0.3 + 
        metrics.get('std_recall', 0.0) * 0.15 + 
        metrics.get('std_specificity', 0.0) * 0.15
    ) * w['std_penalty']
    
    return float(score - penalty)

def calculate_combined_score(aggregated_metrics: Dict[str, float], is_multiclass: bool = False) -> float:
    if is_multiclass:
        return _calculate_combined_score_multiclass(aggregated_metrics)
        
    return _calculate_combined_score_binary(aggregated_metrics)

def aggregate_fold_metrics(fold_results: List[Dict], is_multiclass: bool = False) -> Dict[str, Any]:
    if not fold_results:
        return {}

    best_metrics_list = [
        r['best_metrics'] for r in fold_results if r.get('best_metrics')
    ]

    if not best_metrics_list:
        return {'mean_f1': 0.0, 'std_f1': 0.0}

    agg = {
        'mean_f1': float(np.mean([m['f1_score'] for m in best_metrics_list])),
        'std_f1': float(np.std([m['f1_score'] for m in best_metrics_list])),
        'mean_accuracy': float(np.mean([m['accuracy'] for m in best_metrics_list])),
        'std_accuracy': float(np.std([m['accuracy'] for m in best_metrics_list])),
        'mean_balanced_accuracy': float(np.mean([m['balanced_accuracy'] for m in best_metrics_list])),
        'std_balanced_accuracy': float(np.std([m['balanced_accuracy'] for m in best_metrics_list])),
        'mean_precision': float(np.mean([m['precision'] for m in best_metrics_list])),
        'std_precision': float(np.std([m['precision'] for m in best_metrics_list])),
        'mean_recall': float(np.mean([m['recall'] for m in best_metrics_list])),
        'std_recall': float(np.std([m['recall'] for m in best_metrics_list])),
        'mean_loss': float(np.mean([m['val_loss'] for m in best_metrics_list])),
        'std_loss': float(np.std([m['val_loss'] for m in best_metrics_list])),
        'mean_mcc': float(np.mean([m.get('matthews_correlation_coefficient', 0.0) for m in best_metrics_list])),
        'std_mcc': float(np.std([m.get('matthews_correlation_coefficient', 0.0) for m in best_metrics_list])),
        'n_folds': len(fold_results)
    }

    if 'specificity' in best_metrics_list[0]:
        agg.update({
            'mean_specificity': float(np.mean([m['specificity'] for m in best_metrics_list])),
            'std_specificity': float(np.std([m['specificity'] for m in best_metrics_list])),
        })

    if is_multiclass:
        macro_map = {'precision_macro': 'mean_precision_macro', 'recall_macro': 'mean_recall_macro', 'f1_macro': 'mean_f1_macro'}
        for src, dest in macro_map.items():
            if src in best_metrics_list[0]:
                agg[dest] = float(np.mean([m[src] for m in best_metrics_list]))
                agg[f'std_{src}'] = float(np.std([m[src] for m in best_metrics_list]))

    return agg