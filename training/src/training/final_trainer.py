import os
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from .trainer import get_training_config, train_epoch, validation_epoch
from ..evaluation import evaluate_performance
from ..data import DynamicAugmentationDataset, StaticPreprocessedDataset
from ..utils import load_hyperparameters_config, get_subject_ids_from_dataset

def setup_training_environment(is_multiclass: bool) -> Dict:
    config = get_training_config(is_multiclass)

    model_config = config['model']
    training_config = config['training']
    checkpoint_config = config['checkpoint']

    save_path = os.path.join(os.path.dirname(__file__), str(checkpoint_config['save_path']))
    os.makedirs(save_path, exist_ok=True)

    model_type = "Multiclasse" if is_multiclass else "Binário"
    checkpoint_file = os.path.join(save_path, f"best_model.pth")

    return {
        'config': config,
        'model_config': model_config,
        'training_config': training_config,
        'checkpoint_config': checkpoint_config,
        'save_path': save_path,
        'model_type': model_type,
        'checkpoint_file': checkpoint_file,
        'class_names': model_config['class_names'],
        'num_epochs': training_config['epochs'],
        'early_stopping_patience': training_config['patience'],
        'monitor_metric': checkpoint_config['monitor'],
        'wandb_enabled': config['logging']['wandb']['enabled']
    }

def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_config: Dict
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    scheduler_type = scheduler_config['type']
    params = scheduler_config['params']

    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params['step_size'],
            gamma=params['gamma']
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get('T_max', 50),
            eta_min=params.get('eta_min', 0)
        )
    elif scheduler_type == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=params['mode'],
            factor=params['factor'],
            patience=params['patience'],
            min_lr=params['min_lr']
        )
    else:
        print(f"\nScheduler tipo '{scheduler_type}' não reconhecido. Treinando sem scheduler.\n")
        return None

def setup_scheduler(optimizer: torch.optim.Optimizer, training_config: dict) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    scheduler = None
    if 'scheduler' in training_config:
        scheduler = create_scheduler(optimizer, training_config['scheduler'])
        if scheduler:
            print(f"Scheduler configurado: {training_config['scheduler']['type']}\n")
    return scheduler

def train_single_epoch(model: nn.Module, train_loader: DataLoader,
                       criterion: nn.Module, optimizer: torch.optim.Optimizer,
                       device: torch.device, use_gradient_clipping: bool,
                       max_grad_norm: float, scaler: GradScaler = None) -> Tuple[float, float]:
    return train_epoch(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        apply_clipping=use_gradient_clipping,
        max_grad_norm=max_grad_norm,
        use_amp=True,
        scaler=scaler
    )

def validate_single_epoch(model: nn.Module, val_loader: DataLoader,
                          criterion: nn.Module, device: torch.device,
                          class_names: list, is_multiclass: bool,
                          val_subject_ids: list,
                          train_loss: float, log_to_wandb: bool = False) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict]:
    y_true, y_pred, y_pred_proba, val_loss = validation_epoch(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        use_amp=True
    )

    metrics = evaluate_performance(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_pred_proba,
        subject_ids=val_subject_ids,
        class_names=class_names,
        val_loss=val_loss,
        log_to_wandb=log_to_wandb,
        is_multiclass=is_multiclass
    )

    return y_true, y_pred, y_pred_proba, val_loss, metrics

def update_learning_rate(scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                         metrics: Dict, optimizer: torch.optim.Optimizer):
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metrics['f1_score'])
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}\n")

def save_best_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, val_f1: float, metrics: Dict,
                         config: Dict, checkpoint_file: str):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
        'metrics': metrics,
        'config': config
    }, checkpoint_file)

def print_epoch_metrics(epoch: int, num_epochs: int, train_loss: float,
                        train_acc: float, metrics: Dict):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 60)
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
    print(f"Val Loss:   {metrics['val_loss']:.4f} | F1 (Subj): {metrics['f1_score'] * 100:.2f}%")
    print(f"Acurácia (Subj): {metrics['accuracy'] * 100:.2f}%")

def train_final_model(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_split,
        val_split,
        hyperparameters: dict,
        device: torch.device,
        is_multiclass: bool = False,
        use_gradient_clipping: bool = True,
        max_grad_norm: float = 1.0,
) -> Dict:
    env_config = setup_training_environment(is_multiclass)
    scheduler = setup_scheduler(optimizer, env_config['training_config'])

    print(f"\nConfigurações:")
    print(f"   Épocas: {env_config['num_epochs']}")
    print(f"   Patience: {env_config['early_stopping_patience']}")
    print(f"   Monitor: {env_config['monitor_metric']}")
    print(f"   Checkpoint: {env_config['checkpoint_file']}\n")

    hyperparams_config = load_hyperparameters_config()

    hardware_config = hyperparams_config.get('hardware', {})
    num_workers = hardware_config.get('num_workers', 2)
    pin_memory = hardware_config.get('pin_memory', True)
    mixed_precision = hardware_config.get('mixed_precision', True)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_f1_scores = []

    scaler = GradScaler(device='cuda') if mixed_precision and device.type == 'cuda' else None

    print("Criando dataset de validação (preprocessing estático)...\n")
    val_dataset = StaticPreprocessedDataset(
        subset_dataset=val_split,
        architecture_name=hyperparameters['architecture_name']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_subject_ids = get_subject_ids_from_dataset(val_split)

    for epoch in range(env_config['num_epochs']):
        # Se for um Subset dinâmico (SubjectSamplingSubset), re-amostra as fatias para esta época
        if hasattr(train_split, 'resample'):
            train_split.resample()

        train_dataset = DynamicAugmentationDataset(
            subset_dataset=train_split,
            architecture_name=hyperparameters['architecture_name']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparameters['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        train_loss, train_acc = train_single_epoch(
            model, train_loader, criterion, optimizer, device,
            use_gradient_clipping, max_grad_norm, scaler=scaler
        )

        y_true, y_pred, y_pred_proba, val_loss, metrics = validate_single_epoch(
            model, val_loader, criterion, device,
            env_config['class_names'], is_multiclass, 
            val_subject_ids, train_loss, log_to_wandb=False
        )

        metrics['val_loss'] = val_loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_scores.append(metrics['f1_score'])

        print_epoch_metrics(epoch, env_config['num_epochs'], train_loss, train_acc, metrics)

        if metrics['f1_score'] > best_val_f1:
            best_val_f1 = metrics['f1_score']
            best_epoch = epoch + 1
            patience_counter = 0

            save_best_checkpoint(
                model, optimizer, epoch, best_val_f1, metrics,
                env_config['config'], env_config['checkpoint_file']
            )

            print(f"\nMelhor modelo salvo (F1: {best_val_f1 * 100:.2f}%)")
        else:
            patience_counter += 1
            print(f"\nPatience: {patience_counter}/{env_config['early_stopping_patience']}")

        update_learning_rate(scheduler, metrics, optimizer)

        if patience_counter >= env_config['early_stopping_patience']:
            print(f"\nEarly stopping ativado no epoch {epoch + 1}")
            break

    print(f"\n{'-' * 60}")
    print(f"TREINAMENTO FINAL CONCLUÍDO ({env_config['model_type']})")
    print(f"{'-' * 60}")
    print(f"  Melhor Epoch: {best_epoch}")
    print(f"  Melhor F1-Score (val): {best_val_f1 * 100:.2f}%")
    print(f"  Checkpoint salvo em: {env_config['checkpoint_file']}")
    print(f"{'-' * 60}\n")

    results = {
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'train_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_f1_scores': val_f1_scores
        },
        'checkpoint_path': env_config['checkpoint_file'],
        'model_type': env_config['model_type'],
        'class_names': env_config['class_names'],
        'save_path': env_config['save_path'],
        'is_multiclass': is_multiclass,
        'wandb_enabled': env_config['wandb_enabled'],
    }

    return results
