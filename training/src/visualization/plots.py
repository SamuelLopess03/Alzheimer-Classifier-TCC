import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from matplotlib.figure import Figure

from training.src.evaluation import calculate_roc_metrics

def plot_binary_confusion_matrix(
        cm: np.ndarray,
        metrics: Dict,
        class_names: Optional[List[str]] = None,
        figsize: tuple = (8, 6)
) -> Figure:
    if class_names is None:
        class_names = ['Demented', 'NonDemented']

    fig, ax = plt.subplots(figsize=figsize)

    tp = metrics['true_positive']
    fn = metrics['false_negative']
    fp = metrics['false_positive']
    tn = metrics['true_negative']

    tp_pct = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    fn_pct = fn / (tp + fn) * 100 if (tp + fn) > 0 else 0
    fp_pct = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    tn_pct = tn / (fp + tn) * 100 if (fp + tn) > 0 else 0

    annotations = np.array([
        [f'{tp}\n(TP)\n{tp_pct:.1f}%', f'{fn}\n(FN)\n{fn_pct:.1f}%'],
        [f'{fp}\n(FP)\n{fp_pct:.1f}%', f'{tn}\n(TN)\n{tn_pct:.1f}%']
    ])

    sns.heatmap(
        cm,
        annot=annotations,
        fmt='',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Número de Predições'},
        linewidths=2,
        linecolor='white',
        square=True
    )

    ax.set_xlabel('Predito', fontsize=13, fontweight='bold')
    ax.set_ylabel('Real', fontsize=13, fontweight='bold')
    ax.set_title(
        'Matriz de Confusão - Classificação Binária',
        fontsize=15,
        fontweight='bold',
        pad=20
    )

    text_str = (
        f"Sensitivity (Recall): {metrics['recall'] * 100:.2f}%\n"
        f"Specificity: {metrics['specificity'] * 100:.2f}%\n"
        f"Total Samples: {tp + tn + fp + fn}"
    )

    fig.text(
        0.1, 0.05,
        text_str,
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()

    return fig

def plot_binary_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
        figsize: tuple = (8, 6)
) -> Figure:
    if class_names is None:
        class_names = ['Demented', 'NonDemented']

    roc_metrics = calculate_roc_metrics(y_true, y_pred_proba)

    fpr, tpr = roc_metrics['fpr'], roc_metrics['tpr']
    roc_auc = roc_metrics['auc_roc']
    optimal_threshold = roc_metrics['optimal_threshold']
    optimal_idx = roc_metrics['optimal_idx']

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        fpr, tpr,
        color='darkorange',
        lw=3,
        label=f'ROC Curve (AUC = {roc_auc:.3f})'
    )

    ax.plot(
        [0, 1], [0, 1],
        color='navy',
        lw=2,
        linestyle='--',
        label='Chance (AUC = 0.500)'
    )

    ax.plot(
        fpr[optimal_idx], tpr[optimal_idx],
        'ro',
        markersize=10,
        label=f'Optimal Threshold = {optimal_threshold:.3f}'
    )

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR / Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Curva ROC - Classificação Binária\n({class_names[0]} vs {class_names[1]})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')

    text_str = (
        f'Classe Positiva: {class_names[0]}\n'
        f'Classe Negativa: {class_names[1]}\n'
        f'AUC-ROC: {roc_auc:.4f}\n'
        f'Optimal TPR: {tpr[optimal_idx]:.3f}\n'
        f'Optimal FPR: {fpr[optimal_idx]:.3f}'
    )

    ax.text(
        0.98, 0.22,
        text_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
    )

    plt.tight_layout()

    return fig

def close_figure(fig: Figure):
    plt.close(fig)