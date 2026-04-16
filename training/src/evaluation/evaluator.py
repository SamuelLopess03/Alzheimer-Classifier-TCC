import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..training import validation_epoch
from ..visualization.wandb_logger import log_inference_results
from ..data.dataset_wrappers import StaticPreprocessedDataset
from ..data.subject_manager import get_subject_ids_from_dataset
from src.utils.config import load_hyperparameters_config

from .metrics import evaluate_performance
from .reporter import print_test_metrics_summary, generate_visual_reports, close_visual_reports
from .gradcam import generate_gradcam_visualizations

def _print_evaluation_header(model_type, arch_name, n_classes, gradcam):
    print(f"\n{'=' * 60}")
    print(f"AVALIAÇÃO NO DATASET DE TESTE ({model_type})")
    print(f"{'=' * 60}")
    print(f"  Modelo:   {arch_name.upper()}")
    print(f"  Classes:  {n_classes}")
    print(f"  GradCAM:  {'Sim' if gradcam else 'Não'}")
    print(f"{'=' * 60}\n")

def _prepare_test_dataloader(test_dataset, architecture_name, batch_size=32) -> DataLoader:
    hardware_config = load_hyperparameters_config().get('hardware', {})
    test_preprocessed = StaticPreprocessedDataset(
        subset_dataset=test_dataset,
        architecture_name=architecture_name
    )
    return DataLoader(
        test_preprocessed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=hardware_config.get('num_workers', 2),
        pin_memory=hardware_config.get('pin_memory', True)
    )

def evaluate_on_test_set(
    model: nn.Module, 
    test_loader: DataLoader,
    criterion: nn.Module, 
    device: torch.device,
    class_names: list,
    is_multiclass: bool,
    test_subject_ids: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    y_true, y_pred, y_pred_proba, test_loss = validation_epoch(
        model=model,
        val_loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=True
    )

    test_metrics = evaluate_performance(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_pred_proba,
        subject_ids=test_subject_ids,
        class_names=class_names,
        val_loss=test_loss,
        is_multiclass=is_multiclass
    )

    return y_true, y_pred, y_pred_proba, test_metrics

def evaluate_model(
        model: nn.Module,
        test_dataset,
        device: torch.device,
        generate_gradcam: bool = True,
        gradcam_samples: int = 10,
        is_multiclass: bool = False,
        criterion: Optional[nn.Module] = None,
        save_path: str = "shared/models"
) -> Dict:
    arch_name      = getattr(model, 'architecture_name', 'desconhecida')
    hyperparameters = getattr(model, 'hyperparameters', {'batch_size': 32})
    class_names    = getattr(model, 'class_names', [])

    model_type_str = "Multiclasse" if is_multiclass else "Binário"
    gradcam_output_path = os.path.join(save_path, "gradcam")
    os.makedirs(gradcam_output_path, exist_ok=True)

    _print_evaluation_header(model_type_str, arch_name, len(class_names), generate_gradcam)

    test_loader = _prepare_test_dataloader(
        test_dataset, arch_name, hyperparameters.get('batch_size', 32)
    )
    test_subject_ids = get_subject_ids_from_dataset(test_dataset)

    y_true, y_pred, y_pred_proba, test_metrics = evaluate_on_test_set(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names or None,
        is_multiclass=is_multiclass,
        test_subject_ids=test_subject_ids
    )

    print_test_metrics_summary(test_metrics, model_type_str, is_multiclass)

    fig_cm, fig_roc = generate_visual_reports(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        test_metrics=test_metrics,
        class_names=class_names or None,
        is_multiclass=is_multiclass,
        save_path=save_path,
        model_type=model_type_str,
    )

    log_inference_results(
        test_metrics=test_metrics,
        class_names=class_names or [],
        is_multiclass=is_multiclass,
        cm_fig=fig_cm,
        roc_fig=fig_roc
    )

    close_visual_reports(fig_cm, fig_roc)

    if generate_gradcam:
        try:
            generate_gradcam_visualizations(
                model=model,
                test_loader=test_loader,
                device=device,
                class_names=class_names or None,
                save_path=gradcam_output_path,
                architecture_name=arch_name,
                samples_per_class=gradcam_samples
            )
        except Exception as e:
            print(f"Aviso: Grad-CAM falhou — {str(e)}\n")

    print(f"\n{'=' * 60}")
    print(f"AVALIAÇÃO CONCLUÍDA ({model_type_str})")
    print(f"{'=' * 60}\n")

    return {
        'test_metrics': test_metrics,
        'gradcam_path': gradcam_output_path if generate_gradcam else None
    }