from .plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    close_figure
)

from .wandb_logger import (
    init_wandb_run,
    create_per_class_metrics_table,
    summarize_wandb_repetitions,
    finish_wandb_run,
    # Funções especializadas por fase
    log_search_metrics,
    log_final_training_metrics,
    log_inference_results
)

from .terminal import (
    print_class_distribution,
    print_search_summary,
    print_detailed_metrics
)

__all__ = [
    # Plots Functions
    "plot_confusion_matrix",
    "plot_roc_curve",
    "close_figure",

    # Wandb Logger Functions
    "init_wandb_run",
    "create_per_class_metrics_table",
    "summarize_wandb_repetitions",
    "finish_wandb_run",
    # Funções especializadas por fase
    "log_search_metrics",
    "log_final_training_metrics",
    "log_inference_results",

    # Terminal Functions
    "print_class_distribution",
    "print_search_summary",
    "print_detailed_metrics"
]
