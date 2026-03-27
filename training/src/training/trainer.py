import os
import platform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import lr_scheduler
import numpy as np
from typing import Dict, Tuple, Optional

from ..data import DynamicAugmentationDataset, StaticPreprocessedDataset
from ..evaluation import evaluate_performance
from ..utils import (
    load_binary_config, load_multiclass_config, load_hyperparameters_config,
    get_subject_ids_from_dataset
)

def get_training_config(is_multiclass: bool = False) -> Dict:
    if is_multiclass:
        return load_multiclass_config()
    else:
        return load_binary_config()

def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_config: Dict,
        num_epochs: int
) -> Optional[lr_scheduler.LRScheduler]:
    scheduler_type = scheduler_config.get('type', 'cosine')
    params = scheduler_config.get('params', {})

    if scheduler_type == 'step':
        step_size = int(params.get('step_size', 10))
        gamma = float(params.get('gamma', 0.1))
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        print(f"Scheduler: StepLR (step_size={step_size}, gamma={gamma})\n")

    elif scheduler_type == 'cosine':
        eta_min = float(params.get('eta_min', 1e-7))
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=eta_min
        )
        print(f"Scheduler: CosineAnnealingLR (T_max={num_epochs}, eta_min={eta_min})\n")

    elif scheduler_type == 'reduce_on_plateau':
        factor = float(params.get('factor', 0.5))
        patience = int(params.get('patience', 5))
        min_lr = float(params.get('min_lr', 1e-7))
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
        print(f"Scheduler: ReduceLROnPlateau (factor={factor}, patience={patience}, min_lr={min_lr})\n")

    else:
        print(f"Scheduler desconhecido: {scheduler_type}. Nenhum scheduler será usado.\n")
        return None

    return scheduler

def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        apply_clipping: bool = True,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        scaler: GradScaler = None
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        try:
            inputs: torch.Tensor = inputs.to(device, non_blocking=True)
            labels: torch.Tensor = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nNaN/Inf loss detectado no batch {batch_idx}")
                print(f"   Loss value: {loss.item()}\n")
                continue

            if scaler is not None:
                scaler.scale(loss).backward()

                if apply_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=max_grad_norm
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                if apply_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=max_grad_norm
                    )

                optimizer.step()

            running_loss += loss.item()

            predicted: torch.Tensor
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nCUDA Out of Memory no batch {batch_idx}")
                print("   Limpando cache e continuando...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    avg_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / total_samples

    return avg_loss, train_accuracy

def validation_epoch(
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        use_amp: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()

    val_loss = 0.0

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            try:
                inputs: torch.Tensor = inputs.to(device, non_blocking=True)
                labels: torch.Tensor = labels.to(device, non_blocking=True)

                if use_amp and device.type == 'cuda':
                    with autocast(device_type='cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nNaN/Inf loss no batch {batch_idx} de validação")
                    print(f"   Loss value: {loss.item()}")
                    continue

                val_loss += loss.item()

                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.append(predicted)
                all_labels.append(labels)
                all_probabilities.append(probabilities)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nCUDA Out of Memory no batch {batch_idx} de validação")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    avg_loss = val_loss / len(val_loader)

    y_pred = torch.cat(all_predictions).cpu().numpy()
    y_true = torch.cat(all_labels).cpu().numpy()
    y_pred_proba = torch.cat(all_probabilities).cpu().numpy()

    return y_true, y_pred, y_pred_proba, avg_loss

def train_holdout_model(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_split,
        val_split,
        device: torch.device,
        hyperparams: Dict,
        architecture_name: str = None,
        use_gradient_clipping: bool = True,
        max_grad_norm: float = 1.0,
        repetition_number: int = 1,
        is_multiclass: bool = False
) -> Dict:
    config = get_training_config(is_multiclass)
    hyperparams_config = load_hyperparameters_config()

    model_config = config['model']
    training_config = config['training']

    class_names = model_config['class_names']
    num_epochs = training_config['epochs']
    early_stopping_patience = training_config['patience']

    hardware_config = hyperparams_config.get('hardware', {})
    num_workers = hardware_config.get('num_workers', 2)
    pin_memory = hardware_config.get('pin_memory', True)
    mixed_precision = hardware_config.get('mixed_precision', True)

    model_type = "Multiclasse" if is_multiclass else "Binário"

    print(f"{'-' * 60}")
    print(f"INICIANDO TREINAMENTO - Repetição {repetition_number} ({model_type})")
    print(f"{'-' * 60}\n")
    print(f"Configurações de Treinamento:")
    print(f"   Épocas: {num_epochs}")
    print(f"   Patience: {early_stopping_patience}")
    print(f"   Batch Size: {hyperparams['batch_size']}")
    print(f"   Learning Rate: {hyperparams['learning_rate']}")
    print(f"   Optimizer: {hyperparams['optimizer']}\n")

    best_f1_score = 0.0
    best_metrics = None
    patience_counter = 0

    scaler = GradScaler(device='cuda') if mixed_precision and device.type == 'cuda' else None

    scheduler_config = training_config.get('scheduler', {'type': 'cosine', 'params': {}})
    scheduler = create_scheduler(optimizer, scheduler_config, num_epochs)

    print("Criando dataset de validação (preprocessing estático)...\n")
    val_dataset = StaticPreprocessedDataset(
        subset_dataset=val_split,
        architecture_name=architecture_name
    )
    val_subject_ids = get_subject_ids_from_dataset(val_dataset)

    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    for epoch in range(num_epochs):
        print(f"\n{'-' * 60}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'-' * 60}\n")

        if hasattr(train_split, 'resample'):
            train_split.resample()

        train_dataset = DynamicAugmentationDataset(
            subset_dataset=train_split,
            architecture_name=architecture_name
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparams['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        train_loss, train_accuracy = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            apply_clipping=use_gradient_clipping,
            max_grad_norm=max_grad_norm,
            use_amp=mixed_precision,
            scaler=scaler
        )

        y_true, y_pred, y_pred_proba, val_loss = validation_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=mixed_precision
        )

        # Avaliação CLÍNICA (Apenas Sujeito)
        metrics = evaluate_performance(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_pred_proba,
            subject_ids=val_subject_ids,
            class_names=class_names,
            val_loss=val_loss,
            repetition_number=repetition_number,
            epoch_number=epoch + 1,
            log_to_wandb=config['logging']['wandb']['enabled'],
            is_multiclass=is_multiclass
        )

        print(f"\nResultados do Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy * 100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        print(f"\nANÁLISE POR SUJEITO (N={len(set(val_subject_ids))}):")
        print(f"  F1-Score: {metrics['f1_score'] * 100:.2f}% | Acurácia: {metrics['accuracy'] * 100:.2f}%")
        
        if is_multiclass:
             print(f"  F1 (Macro): {metrics.get('f1_macro', 0) * 100:.2f}%")
        else:
             print(f"  Recall/Sensib: {metrics['recall'] * 100:.2f}%")

        # SELEÇÃO DO MELHOR MODELO: Baseada no F1-Score do Sujeito
        if metrics['f1_score'] > best_f1_score:
            best_f1_score = metrics['f1_score']
            best_metrics = metrics
            patience_counter = 0
            print(f"\nNovo melhor F1-Score (Sujeito): {best_f1_score * 100:.2f}%!")
        else:
            patience_counter += 1
            print(f"\nPatience: {patience_counter}/{early_stopping_patience} (Melhor F1: {best_f1_score * 100:.2f}%)")

        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            current_lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
            print(f"  LRs Atuais: {current_lrs}")

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping ativado no epoch {epoch + 1}")
            break

    result = {
        'best_f1_score': best_f1_score,
        'best_metrics': best_metrics,
    }

    print(f"\n{'-' * 60}")
    print(f"TREINAMENTO CONCLUÍDO - Repetição {repetition_number} ({model_type})")
    print(f"{'-' * 60}")
    print(f"  Melhor F1-Score: {best_f1_score * 100:.2f}%")
    print(f"  Acurácia: {best_metrics['accuracy'] * 100:.2f}%")
    if is_multiclass:
        print(f"  F1 Macro: {best_metrics.get('f1_macro', 0) * 100:.2f}%")
    print(f"{'-' * 60}\n")

    return result