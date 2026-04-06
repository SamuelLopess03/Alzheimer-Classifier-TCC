import os
import numpy as np
from typing import Dict, Optional, List, Tuple
from ..visualization.plots import (
    plot_confusion_matrix, 
    plot_roc_curve,
    close_figure
)

def print_test_metrics_summary(test_metrics: Dict, model_type: str, is_multiclass: bool):
    print(f"\nResultados Finais (PACIENTES - {model_type}):")
    print(f"{'-' * 60}")
    print(f"  F1-Score:       {test_metrics['f1_score'] * 100:.2f}%")
    print(f"  Acurácia:       {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Precisão:       {test_metrics['precision'] * 100:.2f}%")
    print(f"  Recall/Sensib:  {test_metrics['recall'] * 100:.2f}%")
    print(f"  Balanced Acc:   {test_metrics.get('balanced_accuracy', 0) * 100:.2f}%")
    print(f"  MCC:            {test_metrics.get('matthews_correlation_coefficient', 0):.4f}")

    if not is_multiclass:
        print(f"  Especificidade: {test_metrics['specificity'] * 100:.2f}%")
    else:
        print(f"  F1 (Macro):     {test_metrics.get('f1_macro', 0) * 100:.2f}%")
    print(f"{'-' * 60}\n")

def generate_visual_reports(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    test_metrics: Dict,
    class_names: List[str], 
    is_multiclass: bool,
    save_path: str, 
    model_type: str,
) -> Tuple:
    print(f"\n{'-' * 60}")
    print("GERANDO VISUALIZAÇÕES: CONFUSION MATRIX + ROC-AUC")
    print(f"{'-' * 60}\n")

    cm = np.array(test_metrics['confusion_matrix'])
    fig_cm = plot_confusion_matrix(
        cm=cm,
        metrics=test_metrics,
        class_names=class_names,
        is_multiclass=is_multiclass
    )

    cm_path = os.path.join(save_path, f"confusion_matrix_{model_type.lower().replace(' ', '_')}.png")
    fig_cm.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion Matrix salva em: {cm_path}\n")

    fig_roc = plot_roc_curve(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        class_names=class_names,
        is_multiclass=is_multiclass
    )

    roc_path = os.path.join(save_path, f"roc_curve_{model_type.lower().replace(' ', '_')}.png")
    fig_roc.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"ROC Curve salva em: {roc_path}\n")

    return fig_cm, fig_roc

def close_visual_reports(fig_cm, fig_roc):
    close_figure(fig_cm)
    close_figure(fig_roc)
