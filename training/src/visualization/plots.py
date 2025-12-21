import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from matplotlib.figure import Figure

from ..evaluation import calculate_roc_metrics

def plot_confusion_matrix(
        cm: np.ndarray,
        metrics: Dict,
        class_names: List[str],
        is_multiclass: bool = False,
        figsize: Optional[tuple] = None
) -> Figure:
    if is_multiclass:
        return plot_multiclass_confusion_matrix(cm, metrics, class_names, figsize)
    else:
        return plot_binary_confusion_matrix(cm, metrics, class_names, figsize)

def plot_binary_confusion_matrix(
        cm: np.ndarray,
        metrics: Dict,
        class_names: Optional[List[str]] = None,
        figsize: Optional[tuple] = None
) -> Figure:
    if class_names is None:
        class_names = ['Demented', 'Non Demented']

    if figsize is None:
        figsize = (8, 6)

    fig, ax = plt.subplots(figsize=figsize)

    tp = metrics.get('true_positive', cm[0][0])
    fn = metrics.get('false_negative', cm[0][1])
    fp = metrics.get('false_positive', cm[1][0])
    tn = metrics.get('true_negative', cm[1][1])

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
        f"Sensitivity (Recall): {metrics.get('recall', 0) * 100:.2f}%\n"
        f"Specificity: {metrics.get('specificity', 0) * 100:.2f}%\n"
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

def plot_multiclass_confusion_matrix(
        cm: np.ndarray,
        metrics: Dict,
        class_names: List[str],
        figsize: Optional[tuple] = None,
        normalize: bool = False
) -> Figure:
    if figsize is None:
        figsize = (10, 8)

    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_plot = np.nan_to_num(cm_plot)  # Substituir NaN por 0
        cbar_label = 'Proporção de Predições'
    else:
        cm_plot = cm
        cbar_label = 'Número de Predições'

    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            total = cm[i, :].sum()
            pct = (count / total * 100) if total > 0 else 0

            if normalize:
                annotations[i, j] = f'{cm_plot[i, j]:.1%}\n({count})'
            else:
                annotations[i, j] = f'{count}\n({pct:.1f}%)'

    sns.heatmap(
        cm_plot,
        annot=annotations,
        fmt='',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': cbar_label},
        linewidths=1,
        linecolor='white',
        square=True
    )

    ax.set_xlabel('Predito', fontsize=13, fontweight='bold')
    ax.set_ylabel('Real', fontsize=13, fontweight='bold')
    ax.set_title(
        'Matriz de Confusão - Classificação Multiclasse',
        fontsize=15,
        fontweight='bold',
        pad=20
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    total_samples = cm.sum()
    accuracy = metrics.get('accuracy', 0)
    balanced_acc = metrics.get('balanced_accuracy', 0)

    text_str = (
        f"Accuracy: {accuracy * 100:.2f}%\n"
        f"Balanced Accuracy: {balanced_acc * 100:.2f}%\n"
        f"Total Samples: {int(total_samples)}"
    )

    fig.text(
        0.1, 0.02,
        text_str,
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()

    return fig

def plot_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: List[str],
        is_multiclass: bool = False,
        figsize: Optional[tuple] = None
) -> Figure:
    if is_multiclass:
        return plot_multiclass_roc_curve(y_true, y_pred_proba, class_names, figsize)
    else:
        return plot_binary_roc_curve(y_true, y_pred_proba, class_names, figsize)

def plot_binary_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
        figsize: Optional[tuple] = None
) -> Figure:
    if class_names is None:
        class_names = ['Demented', 'Non Demented']

    if figsize is None:
        figsize = (8, 6)

    roc_metrics = calculate_roc_metrics(y_true, y_pred_proba, is_multiclass=False)

    fpr = roc_metrics['fpr']
    tpr = roc_metrics['tpr']
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

def plot_multiclass_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: List[str],
        figsize: Optional[tuple] = None
) -> Figure:
    if figsize is None:
        figsize = (10, 8)

    num_classes = len(class_names)

    roc_metrics = calculate_roc_metrics(y_true, y_pred_proba, is_multiclass=True)

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, num_classes))

    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        class_metrics = roc_metrics[f'class_{i}']
        fpr = class_metrics['fpr']
        tpr = class_metrics['tpr']
        auc_score = class_metrics['auc_roc']

        ax.plot(
            fpr, tpr,
            color=color,
            lw=2.5,
            label=f'{class_name} (AUC = {auc_score:.3f})'
        )

    ax.plot(
        [0, 1], [0, 1],
        color='navy',
        lw=2,
        linestyle='--',
        label='Chance (AUC = 0.500)'
    )

    if 'macro_auc' in roc_metrics:
        macro_auc = roc_metrics['macro_auc']
        ax.plot(
            [], [],  # Linha invisível apenas para a legenda
            color='black',
            lw=3,
            linestyle=':',
            label=f'Macro-average (AUC = {macro_auc:.3f})'
        )

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Curvas ROC - Classificação Multiclasse (One-vs-Rest)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')

    text_str = f'Número de Classes: {num_classes}\n'
    if 'macro_auc' in roc_metrics:
        text_str += f'Macro AUC: {roc_metrics["macro_auc"]:.4f}\n'
    if 'weighted_auc' in roc_metrics:
        text_str += f'Weighted AUC: {roc_metrics["weighted_auc"]:.4f}'

    ax.text(
        0.98, 0.02,
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
    if fig is not None:
        plt.close(fig)