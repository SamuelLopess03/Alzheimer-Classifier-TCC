import os
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..training import validation_epoch
from ..visualization import (
    finish_wandb_run,
    initialize_wandb_tracking
)
from ..data import StaticPreprocessedDataset, get_subject_ids_from_dataset
from ..utils import load_hyperparameters_config

from .metrics import evaluate_performance
from .reporter import print_test_metrics_summary, generate_visual_reports
from .gradcam import generate_gradcam_visualizations, log_gradcam_to_wandb

def _print_evaluation_header(model_type, checkpoint, params, classes, gradcam):
    print(f"\n{'=' * 60}")
    print(f"AVALIAÇÃO NO DATASET DE TESTE ({model_type})")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Modelo:     {params.get('architecture_name', 'N/A')}")
    print(f"  Classes:    {classes}")
    print(f"  GradCAM:    {'Sim' if gradcam else 'Não'}")
    print(f"{'=' * 60}\n")

def _prepare_test_dataloader(test_dataset, hyperparameters) -> DataLoader:
    hyperparams_config = load_hyperparameters_config()
    hardware_config = hyperparams_config.get('hardware', {})
    
    print("Criando dataset de teste (configuração estática)...\n")
    test_preprocessed = StaticPreprocessedDataset(
        subset_dataset=test_dataset,
        architecture_name=hyperparameters['architecture_name']
    )

    return DataLoader(
        test_preprocessed,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        num_workers=hardware_config.get('num_workers', 2),
        pin_memory=hardware_config.get('pin_memory', True)
    )

def evaluate_on_test_set(
    model: nn.Module, 
    test_loader: DataLoader,
    criterion: nn.Module, 
    device: torch.device,
    checkpoint_file: str, 
    class_names: list,
    is_multiclass: bool,
    test_subject_ids: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

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
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        test_dataset,
        training_results: Dict,
        hyperparameters: dict,
        device: torch.device,
        generate_gradcam: bool = True,
        gradcam_samples: int = 10
) -> Dict:
    checkpoint_file = training_results['checkpoint_path']
    model_type = training_results.get('model_type', 'Binário')
    class_names = training_results.get('class_names', [])
    is_multiclass = training_results.get('is_multiclass', False)
    save_path = training_results.get('save_path', '.')

    gradcam_path = os.path.join(os.path.dirname(save_path), "outputs", "gradcam")
    os.makedirs(gradcam_path, exist_ok=True)

    _print_evaluation_header(model_type, checkpoint_file, hyperparameters, class_names, generate_gradcam)

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_file}")

    test_loader = _prepare_test_dataloader(test_dataset, hyperparameters)
    test_subject_ids = get_subject_ids_from_dataset(test_dataset)

    wandb_enabled, run = initialize_wandb_tracking(
        training_results, hyperparameters, optimizer, criterion,
        hyperparameters.get('use_gradient_clipping', True),
        hyperparameters.get('max_grad_norm', 1.0)
    )

    y_true, y_pred, y_pred_proba, test_metrics = evaluate_on_test_set(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        checkpoint_file=checkpoint_file,
        class_names=class_names,
        is_multiclass=is_multiclass,
        test_subject_ids=test_subject_ids
    )

    print_test_metrics_summary(test_metrics, model_type, is_multiclass)
    
    generate_visual_reports(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        test_metrics=test_metrics,
        class_names=class_names,
        is_multiclass=is_multiclass,
        save_path=save_path,
        model_type=model_type,
        wandb_enabled=wandb_enabled,
        run=run
    )

    if generate_gradcam:
        try:
            generate_gradcam_visualizations(
                model=model,
                test_loader=test_loader,
                device=device,
                class_names=class_names,
                save_path=gradcam_path,
                architecture_name=hyperparameters['architecture_name'],
                num_samples=gradcam_samples,
                samples_per_class=max(2, gradcam_samples // len(class_names))
            )
            
            if wandb_enabled:
                log_gradcam_to_wandb(gradcam_path, model_type)

        except Exception as e:
            print(f"Erro ao gerar Grad-CAM: {str(e)}\n")

    if wandb_enabled and run is not None:
        finish_wandb_run(quiet=False)

    print(f"\n{'=' * 60}")
    print(f"AVALIAÇÃO CONCLUÍDA ({model_type})")
    print(f"{'=' * 60}\n")

    return {
        'test_metrics': test_metrics,
        'checkpoint_path': checkpoint_file,
        'gradcam_path': gradcam_path if generate_gradcam else None
    }