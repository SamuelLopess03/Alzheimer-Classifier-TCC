import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from typing import Dict, Tuple, Optional
import os

from training.src.data import DynamicAugmentationDataset, StaticPreprocessedDataset
from training.src.evaluation import calculate_metrics_model
from training.src.utils import load_binary_config, load_multiclass_config, load_hyperparameters_config

def get_training_config(is_multiclass: bool = False) -> Dict:
    if is_multiclass:
        return load_multiclass_config()
    else:
        return load_binary_config()

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

def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        apply_clipping: bool = True,
        max_grad_norm: float = 1.0,
        use_amp: bool = True
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    scaler = GradScaler(device='cuda') if use_amp and device.type == 'cuda' else None

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        try:
            inputs: torch.Tensor = inputs.to(device)
            labels: torch.Tensor = labels.to(device)

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

    print("Criando dataset de validação (preprocessing estático)...\n")
    val_dataset = StaticPreprocessedDataset(
        subset_dataset=val_split,
        architecture_name=architecture_name
    )

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

        print("Criando dataset de treino (augmentation dinâmica)...\n")
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
            use_amp=mixed_precision
        )

        y_true, y_pred, y_pred_proba, val_loss = validation_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=mixed_precision
        )

        metrics = calculate_metrics_model(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            val_loss=val_loss,
            train_loss=train_loss,
            repetition_number=repetition_number,
            epoch_number=epoch + 1,
            log_to_wandb=config['logging']['wandb']['enabled']
        )

        print(f"\nResultados do Epoch {epoch + 1} - Modelo {model_type}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy * 100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {metrics['accuracy'] * 100:.2f}%")
        print(f"  Balanced Acc: {metrics['balanced_accuracy'] * 100:.2f}%")
        print(f"  F1-Score:     {metrics['f1_score'] * 100:.2f}%")
        print(f"  Sensitivity:  {metrics['recall'] * 100:.2f}%")
        print(f"  Specificity:  {metrics['specificity'] * 100:.2f}%")

        if metrics['f1_score'] > best_f1_score:
            best_f1_score = metrics['f1_score']
            best_metrics = metrics
            patience_counter = 0

            print(f"\nNovo melhor F1-Score: {best_f1_score * 100:.2f}%!")
        else:
            patience_counter += 1
            print(f"\nPatience: {patience_counter}/{early_stopping_patience}")

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
    print(f"  Balanced Acc: {best_metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"{'-' * 60}\n")

    return result

def train_final_model(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        is_multiclass: bool = False,
        use_gradient_clipping: bool = True,
        max_grad_norm: float = 1.0
) -> Dict:
    config = get_training_config(is_multiclass)

    model_config = config['model']
    training_config = config['training']
    checkpoint_config = config['checkpoint']

    class_names = model_config['class_names']
    num_epochs = training_config['epochs']
    early_stopping_patience = training_config['patience']
    monitor_metric = checkpoint_config['monitor']
    save_path = checkpoint_config['save_path']

    os.makedirs(save_path, exist_ok=True)

    model_type = "Multiclasse" if is_multiclass else "Binário"
    checkpoint_file = os.path.join(save_path, f"best_model_{model_type.lower()}.pth")

    scheduler = None
    if 'scheduler' in training_config:
        scheduler = create_scheduler(optimizer, training_config['scheduler'])
        if scheduler:
            print(f"Scheduler configurado: {training_config['scheduler']['type']}\n")

    print(f"\n{'-' * 60}")
    print(f"TREINAMENTO FINAL DO MODELO ({model_type})")
    print(f"{'-' * 60}")
    print(f"Configurações:")
    print(f"   Épocas: {num_epochs}")
    print(f"   Patience: {early_stopping_patience}")
    print(f"   Monitor: {monitor_metric}")
    print(f"   Checkpoint: {checkpoint_file}\n")

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            apply_clipping=use_gradient_clipping,
            max_grad_norm=max_grad_norm,
            use_amp=True
        )

        y_true, y_pred, y_pred_proba, val_loss = validation_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=True
        )

        metrics = calculate_metrics_model(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            val_loss=val_loss,
            train_loss=train_loss,
            log_to_wandb=config['logging']['wandb']['enabled']
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_scores.append(metrics['f1_score'])

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {metrics['accuracy'] * 100:.2f}%")
        print(f"Val F1: {metrics['f1_score'] * 100:.2f}% | Balanced Acc: {metrics['balanced_accuracy'] * 100:.2f}%")

        if metrics['f1_score'] > best_val_f1:
            best_val_f1 = metrics['f1_score']
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'metrics': metrics,
                'config': config
            }, checkpoint_file)

            print(f"\nMelhor modelo salvo (F1: {best_val_f1 * 100:.2f}%)")
        else:
            patience_counter += 1
            print(f"\nPatience: {patience_counter}/{early_stopping_patience}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics['f1_score'])
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.2e}")

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping ativado no epoch {epoch + 1}")
            break

    print(f"\n{'-' * 60}")
    print(f"AVALIAÇÃO NO TEST SET ({model_type})")
    print(f"{'-' * 60}\n")

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    y_true, y_pred, y_pred_proba, test_loss = validation_epoch(
        model=model,
        val_loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=True
    )

    test_metrics = calculate_metrics_model(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        val_loss=test_loss,
        train_loss=0.0,
        log_to_wandb=False
    )

    print(f"\nResultados Finais (Test Set - {model_type}):")
    print(f"  Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Balanced Acc: {test_metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  F1-Score: {test_metrics['f1_score'] * 100:.2f}%")
    print(f"  Sensitivity: {test_metrics['recall'] * 100:.2f}%")
    print(f"  Specificity: {test_metrics['specificity'] * 100:.2f}%")
    print(f"  Precision: {test_metrics['precision'] * 100:.2f}%\n")

    results = {
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'train_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_f1_scores': val_f1_scores
        },
        'checkpoint_path': checkpoint_file
    }

    return results