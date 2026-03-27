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
from typing import Dict, List, Optional, Any
import pandas as pd
from ..utils import extract_subject_id

def evaluate_performance(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        subject_ids: List[str],
        class_names: Optional[List[str]] = None,
        val_loss: float = 0.0,
        repetition_number: int = 1,
        epoch_number: int = 1,
        log_to_wandb: bool = True,
        is_multiclass: bool = False
) -> Dict:
    if len(y_true) != len(subject_ids):
        print(f"Erro: Tamanho de y_true ({len(y_true)}) diferente de subject_ids ({len(subject_ids)})")
        return {}

    # 1. Agregação e Voto Majoritário por Sujeito
    df = pd.DataFrame({
        'subject_id': subject_ids,
        'y_true': y_true,
        'y_pred': y_pred
    })

    subject_agg = df.groupby('subject_id').agg({
        'y_true': 'first',
        'y_pred': lambda x: x.mode().iloc[0]
    }).reset_index()

    y_subj_true = subject_agg['y_true'].values
    y_subj_pred = subject_agg['y_pred'].values
    
    # 2. Cálculo das Métricas (Sklearn)
    if class_names is None:
        if is_multiclass:
            class_names = ['Mild+Moderate Dementia', 'Very mild Dementia']
        else:
            class_names = ['Demented', 'Non Demented']
    
    num_classes = len(class_names)
    labels = list(range(num_classes))
    
    # Métricas Globais
    accuracy = accuracy_score(y_subj_true, y_subj_pred)
    balanced_acc = balanced_accuracy_score(y_subj_true, y_subj_pred)
    mcc = matthews_corrcoef(y_subj_true, y_subj_pred)
    kappa = cohen_kappa_score(y_subj_true, y_subj_pred)
    
    # Precision, Recall, F1 (Weighted, Macro e Per-Class)
    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_subj_true, y_subj_pred, average=None, labels=labels, zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_subj_true, y_subj_pred, average='weighted', labels=labels, zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_subj_true, y_subj_pred, average='macro', labels=labels, zero_division=0
    )
    
    cm = confusion_matrix(y_subj_true, y_subj_pred, labels=labels)
    
    # Dicionário Consolidado
    metrics = {
        'val_loss': float(val_loss),
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'matthews_correlation_coefficient': float(mcc),
        'cohen_kappa': float(kappa),
        
        # Weighted Metrics (Padrão)
        'precision': float(p_w),
        'recall': float(r_w),
        'f1_score': float(f1_w),
        
        # Macro Metrics (Úteis para desbalanceamento)
        'precision_macro': float(p_m),
        'recall_macro': float(r_m),
        'f1_macro': float(f1_m),
        
        # Detalhes por classe
        'precision_per_class': [float(p) for p in p_class],
        'recall_per_class': [float(r) for r in r_class],
        'f1_per_class': [float(f) for f in f1_class],
        'support_per_class': [int(s) for s in support],
        
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            y_subj_true, y_subj_pred, target_names=class_names, labels=labels, zero_division=0
        )
    }

    # Métricas de Erro Especializadas (Specificity, etc.)
    if not is_multiclass and num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
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
        # Multiclasse: Macro Médias para Specificity e NPV
        specificities = []
        npvs = []
        for i in range(num_classes):
            tn = np.sum((y_subj_true != i) & (y_subj_pred != i))
            fp = np.sum((y_subj_true != i) & (y_subj_pred == i))
            fn = np.sum((y_subj_true == i) & (y_subj_pred != i))
            tp = np.sum((y_subj_true == i) & (y_subj_pred == i))
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            npvs.append(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
        
        metrics.update({
            'specificity': float(np.mean(specificities)),
            'specificity_macro': float(np.mean(specificities)),
            'negative_predictive_value': float(np.mean(npvs)),
            'npv_macro': float(np.mean(npvs)),
            'specificity_per_class': [float(s) for s in specificities],
            'npv_per_class': [float(n) for n in npvs]
        })

    # 3. Log no WandB (Métrica Clínica Completa)
    if log_to_wandb and wandb.run is not None:
        wandb_log = {
            f"rep_{repetition_number}/epoch": epoch_number,
            f"rep_{repetition_number}/f1_subj": metrics['f1_score'],
            f"rep_{repetition_number}/acc_subj": metrics['accuracy'],
            f"rep_{repetition_number}/mcc_subj": metrics['matthews_correlation_coefficient'],
            f"rep_{repetition_number}/kappa_subj": metrics['cohen_kappa'],
            f"rep_{repetition_number}/loss": val_loss,
        }
        # Adiciona métricas por classe ao WandB
        for i, name in enumerate(class_names):
            wandb_log[f"rep_{repetition_number}/f1_{name}"] = metrics['f1_per_class'][i]
            
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