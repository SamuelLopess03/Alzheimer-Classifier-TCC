import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import numpy as np
from typing import Dict, Tuple, Optional

from .engine import train_epoch, validation_epoch
from .factory import get_training_config, create_scheduler
from .checkpoints import save_best_checkpoint
from ..data.dataset_wrappers import DynamicAugmentationDataset, StaticPreprocessedDataset
from ..data.subject_manager import get_subject_ids_from_dataset
from ..evaluation import evaluate_performance
from ..visualization.wandb_logger import log_final_training_metrics
from ..visualization.terminal import print_epoch_log
from src.utils.config import load_hyperparameters_config
from src.utils.hardware import get_pytorch_device

def run_training_process(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_split,
        val_split,
        device: torch.device,
        hyperparams: Dict,
        architecture_name: str,
        is_multiclass: bool = False,
        is_final_training: bool = False,
        fold_number: int = 1
) -> Dict:
    config = get_training_config(is_multiclass)
    hyperparams_config = load_hyperparameters_config()
    
    model_config = config['model']
    training_config = config['training']
    checkpoint_config = config['checkpoint']

    class_names = model_config['class_names']
    num_epochs = training_config['epochs']
    early_stopping_patience = training_config['patience']

    hardware_config = hyperparams_config.get('hardware', {})
    num_workers = hardware_config.get('num_workers', 2)
    pin_memory = hardware_config.get('pin_memory', True)
    mixed_precision = hardware_config.get('mixed_precision', True)
    
    model_type_str = "MULTICLASSE" if is_multiclass else "BINÁRIO"
    mode_str = "FINAL" if is_final_training else "BUSCA"
    
    print(f"\n{'-' * 60}")
    print(f"INICIANDO TREINAMENTO {mode_str} ({model_type_str})")
    print(f"Arquitetura: {architecture_name.upper()} | Fold: {fold_number}")
    print(f"{'-' * 60}")

    scaler = GradScaler(device=device.type) if mixed_precision and device.type == 'cuda' else None
    
    scheduler_config = training_config.get('scheduler', {'type': 'cosine', 'params': {}})
    scheduler = create_scheduler(optimizer, scheduler_config, num_epochs)

    print("Configurando Datasets...")
    val_dataset = StaticPreprocessedDataset(subset_dataset=val_split, architecture_name=architecture_name)
    val_subject_ids = get_subject_ids_from_dataset(val_dataset)
    val_loader = DataLoader(
        val_dataset, batch_size=hyperparams['batch_size'], shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    checkpoint_file = None
    if is_final_training:
        save_path = os.path.join(os.path.dirname(__file__), str(checkpoint_config['save_path']))
        os.makedirs(save_path, exist_ok=True)
        checkpoint_file = os.path.join(save_path, "best_model.pth")
        print(f"Checkpoint será salvo em: {checkpoint_file}")

    best_f1_score = 0.0
    best_metrics = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    for epoch in range(num_epochs):
        if hasattr(train_split, 'resample'):
            train_split.resample()

        train_dataset = DynamicAugmentationDataset(subset_dataset=train_split, architecture_name=architecture_name)
        train_loader = DataLoader(
            train_dataset, batch_size=hyperparams['batch_size'], shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True
        )

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            apply_clipping=True, max_grad_norm=1.0, use_amp=mixed_precision, scaler=scaler
        )

        y_true, y_pred, y_prob, val_loss = validation_epoch(
            model, val_loader, criterion, device, use_amp=mixed_precision
        )

        metrics = evaluate_performance(
            y_true=y_true, y_pred=y_pred, y_prob=y_prob, 
            subject_ids=val_subject_ids, class_names=class_names,
            val_loss=val_loss, fold_number=fold_number,
            epoch_number=epoch + 1, is_multiclass=is_multiclass
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(metrics['f1_macro'])

        if config['logging']['wandb']['enabled'] and is_final_training:
            log_final_training_metrics(
                metrics=metrics,
                epoch=epoch + 1,
                fold_number=fold_number,
                class_names=class_names,
                train_loss=train_loss,
                learning_rate=optimizer.param_groups[0]['lr']
            )

        is_epoch_best = metrics['f1_macro'] > best_f1_score
        
        print_epoch_log(
            epoch=epoch + 1, 
            num_epochs=num_epochs, 
            train_loss=train_loss, 
            metrics=metrics, 
            patience=patience_counter,
            is_best=is_epoch_best
        )

        if is_epoch_best:
            best_f1_score = metrics['f1_macro']
            best_metrics = metrics
            patience_counter = 0
            
            if is_final_training:
                save_best_checkpoint(
                    model, optimizer, epoch, best_f1_score, metrics,
                    config, checkpoint_file, architecture_name, hyperparams
                )
        else:
            patience_counter += 1

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics['f1_score'])
            else:
                scheduler.step()

        if patience_counter >= early_stopping_patience:
            print(f"\n[INFO] Early stopping no epoch {epoch + 1}")
            break

    print(f"\n{'-' * 60}")
    print(f"TREINAMENTO CONCLUÍDO | Melhor F1: {best_f1_score * 100:.2f}%")
    print(f"{'-' * 60}\n")

    return {
        'best_f1_score': best_f1_score,
        'best_metrics': best_metrics,
        'history': history,
        'checkpoint_path': checkpoint_file
    }