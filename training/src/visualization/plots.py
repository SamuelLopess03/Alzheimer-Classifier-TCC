import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from matplotlib.figure import Figure

def _plot_base_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    xlabel: str = 'Predito',
    ylabel: str = 'Real',
    figsize: tuple = (8, 6),
    cmap: str = 'Blues',
    cbar_label: str = 'Número de Predições',
    extra_text: Optional[str] = None
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            total = cm[i, :].sum()
            pct = (count / total * 100) if total > 0 else 0
            annotations[i, j] = f'{count}\n({pct:.1f}%)'

    sns.heatmap(
        cm,
        annot=annotations,
        fmt='',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': cbar_label},
        linewidths=1.5,
        linecolor='white',
        square=True
    )

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    if len(class_names) > 3:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    if extra_text:
        fig.text(
            0.1, 0.02,
            extra_text,
            ha='left',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        plt.subplots_adjust(bottom=0.18)

    plt.tight_layout()
    return fig

def plot_confusion_matrix(
    cm: np.ndarray,
    metrics: Dict,
    class_names: List[str],
    is_multiclass: bool = False,
    figsize: Optional[tuple] = None
) -> Figure:
    if is_multiclass:
        title = 'Matriz de Confusão - Classificação Multiclasse'
        if figsize is None: figsize = (10, 8)
        
        extra_text = (
            f"Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%\n"
            f"Balanced Acc: {metrics.get('balanced_accuracy', 0) * 100:.2f}%\n"
            f"Total Samples: {int(cm.sum())}"
        )
    else:
        title = 'Matriz de Confusão - Classificação Binária'
        if figsize is None: figsize = (8, 6)
        
        extra_text = (
            f"Sensitivity: {metrics.get('recall', 0) * 100:.2f}%\n"
            f"Specificity: {metrics.get('specificity', 0) * 100:.2f}%\n"
            f"Total Samples: {int(cm.sum())}"
        )

    return _plot_base_confusion_matrix(
        cm=cm, 
        class_names=class_names, 
        title=title, 
        figsize=figsize,
        extra_text=extra_text
    )

def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    is_multiclass: bool = False,
    figsize: Optional[tuple] = None
) -> Figure:
    if figsize is None:
        figsize = (8, 6) if not is_multiclass else (10, 8)
        
    from ..evaluation import calculate_roc_metrics
    roc_metrics = calculate_roc_metrics(y_true, y_pred_proba, is_multiclass=is_multiclass)
    fig, ax = plt.subplots(figsize=figsize)

    if not is_multiclass:
        fpr, tpr = roc_metrics['fpr'], roc_metrics['tpr']
        roc_auc = roc_metrics['auc_roc']
        optimal_idx = roc_metrics['optimal_idx']
        
        ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, label=f'Threshold: {roc_metrics["optimal_threshold"]:.3f}')
    else:
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(class_names)))
        for i, (color, name) in enumerate(zip(colors, class_names)):
            m = roc_metrics[f'class_{i}']
            ax.plot(m['fpr'], m['tpr'], color=color, lw=2, label=f'{name} (AUC={m["auc_roc"]:.3f})')
        
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (0.5)')
    ax.set_xlabel('FPR', fontweight='bold')
    ax.set_ylabel('TPR', fontweight='bold')
    ax.set_title(f'Curva ROC - {"Multiclasse" if is_multiclass else "Binária"}', fontweight='bold', pad=20)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def close_figure(fig: Figure):
    if fig is not None:
        plt.close(fig)