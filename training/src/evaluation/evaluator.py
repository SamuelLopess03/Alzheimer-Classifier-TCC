import os
from typing import Dict, Optional, Tuple, List, Union
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

def _print_evaluation_header(is_ensemble, model_type_str, arch_name, n_classes, gradcam):
    title = "AVALIAÇÃO ENSEMBLE" if is_ensemble else "AVALIAÇÃO DE MODELO"
    print(f"\n{'=' * 60}")
    print(f"{title} NO TEST SET ({model_type_str})")
    print(f"{'=' * 60}")
    print(f"  Modelo(s): {arch_name.upper()}")
    print(f"  Classes:   {n_classes}")
    print(f"  GradCAM:   {'Sim' if gradcam else 'Não'}")
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

def _get_predictions(
    models: List[nn.Module], 
    test_loader: DataLoader, 
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_probas = []
    y_true = None
    
    for i, model in enumerate(models):
        if len(models) > 1:
            print(f"  Avaliando modelo {i+1}/{len(models)}...")
        
        y_true_fold, _, y_prob_fold, _ = validation_epoch(
            model=model,
            val_loader=test_loader,
            criterion=nn.CrossEntropyLoss(), 
            device=device,
            use_amp=True
        )
        all_probas.append(y_prob_fold)
        y_true = y_true_fold
        
    ensemble_proba = np.mean(all_probas, axis=0)
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    
    return y_true, ensemble_pred, ensemble_proba

def _run_shared_evaluation_flow(
    models: List[nn.Module],
    test_dataset,
    device: torch.device,
    is_multiclass: bool,
    generate_gradcam: bool,
    gradcam_samples: int,
    save_path: str
) -> Dict:
    main_model = models[0]
    arch_name = getattr(main_model, 'architecture_name', 'desconhecida')
    class_names = getattr(main_model, 'class_names', [])
    hyperparams = getattr(main_model, 'hyperparameters', {'batch_size': 32})
    model_type_str = "Multiclasse" if is_multiclass else "Binário"
    is_ensemble = len(models) > 1

    _print_evaluation_header(is_ensemble, model_type_str, arch_name, len(class_names), generate_gradcam)

    test_loader = _prepare_test_dataloader(test_dataset, arch_name, hyperparams.get('batch_size', 32))
    test_subject_ids = get_subject_ids_from_dataset(test_dataset)

    y_true, y_pred, y_proba = _get_predictions(models, test_loader, device)
    
    test_metrics = evaluate_performance(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_proba,
        subject_ids=test_subject_ids,
        class_names=class_names,
        val_loss=0.0, 
        is_multiclass=is_multiclass
    )

    print_test_metrics_summary(test_metrics, model_type_str, is_multiclass)
    
    fig_cm, fig_roc = generate_visual_reports(
        y_true=y_true, y_pred_proba=y_proba, test_metrics=test_metrics,
        class_names=class_names or None, is_multiclass=is_multiclass,
        save_path=save_path, model_type=model_type_str,
        subject_ids=test_subject_ids
    )

    log_inference_results(test_metrics, class_names or [], is_multiclass, fig_cm, fig_roc)
    close_visual_reports(fig_cm, fig_roc)

    gradcam_path = None
    if generate_gradcam:
        gradcam_path = os.path.join(save_path, "gradcam")
        os.makedirs(gradcam_path, exist_ok=True)
        try:
            generate_gradcam_visualizations(
                models=models, test_loader=test_loader, device=device,
                class_names=class_names or None, save_path=gradcam_path,
                architecture_name=arch_name, samples_per_class=gradcam_samples,
                y_true=y_true, y_pred=y_pred
            )
        except Exception as e:
            print(f"Aviso: Grad-CAM falhou - {str(e)}\n")

    print(f"\n{'=' * 60}")
    print(f"AVALIAÇÃO CONCLUÍDA ({model_type_str})")
    print(f"{'=' * 60}\n")

    return {'test_metrics': test_metrics, 'gradcam_path': gradcam_path}

def evaluate_model(model: nn.Module, **kwargs) -> Dict:
    return _run_shared_evaluation_flow(models=[model], **kwargs)

def evaluate_ensemble(models: List[nn.Module], **kwargs) -> Dict:
    return _run_shared_evaluation_flow(models=models, **kwargs)