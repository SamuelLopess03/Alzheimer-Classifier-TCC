import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from typing import Dict, Tuple, Optional

from training.src.data import DynamicAugmentationDataset, StaticPreprocessedDataset
from training.src.evaluation import calculate_binary_metrics

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
        class_names: list,
        hyperparams: Dict,
        num_epochs: int,
        early_stopping_patience: int,
        architecture_name: str = None,
        use_gradient_clipping: bool = True,
        max_grad_norm: float = 1.0,
        repetition_number: int = 1
) -> Dict:
    print(f"{'-' * 60}")
    print(f"INICIANDO TREINAMENTO - Repetição {repetition_number}")
    print(f"{'-' * 60}\n")

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
        num_workers=2,
        pin_memory=torch.cuda.is_available()
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
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
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
            use_amp=True
        )

        y_true, y_pred, y_pred_proba, val_loss = validation_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=True
        )

        metrics = calculate_binary_metrics(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            val_loss=val_loss,
            train_loss=train_loss,
            repetition_number=repetition_number,
            epoch_number=epoch + 1,
            log_to_wandb=True
        )

        print(f"\nResultados do Epoch {epoch + 1}:")
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
    print(f"TREINAMENTO CONCLUÍDO - Repetição {repetition_number}")
    print(f"{'-' * 60}")
    print(f"  Melhor F1-Score: {best_f1_score * 100:.2f}%")
    print(f"  Balanced Acc: {best_metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"{'-' * 60}\n")

    return result

def train_final_model(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        class_names: list,
        num_epochs: int,
        save_path: str,
        early_stopping_patience: int = 10
) -> Dict:
    print(f"\n{'-' * 60}")
    print("TREINAMENTO FINAL DO MODELO")
    print(f"{'-' * 60}\n")

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
            model, train_loader, criterion, optimizer, device
        )

        y_true, y_pred, y_pred_proba, val_loss = validation_epoch(
            model, val_loader, criterion, device
        )

        metrics = calculate_binary_metrics(
            y_true, y_pred, class_names, val_loss, train_loss,
            log_to_wandb=False
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_scores.append(metrics['f1_score'])

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val F1: {metrics['f1_score'] * 100:.2f}% | Val Acc: {metrics['accuracy'] * 100:.2f}%\n")

        if metrics['f1_score'] > best_val_f1:
            best_val_f1 = metrics['f1_score']
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'metrics': metrics
            }, save_path)

            print(f"Melhor modelo salvo (F1: {best_val_f1 * 100:.2f}%)\n")
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step()
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}\n")

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping no epoch {epoch + 1}")
            break

    print(f"\n{'-' * 60}")
    print("AVALIAÇÃO NO TEST SET")
    print(f"{'-' * 60}\n")

    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    y_true, y_pred, y_pred_proba, test_loss = validation_epoch(
        model, test_loader, criterion, device
    )

    test_metrics = calculate_binary_metrics(
        y_true, y_pred, class_names, test_loss, 0.0,
        log_to_wandb=False
    )

    print(f"\nResultados Finais (Test Set):")
    print(f"  Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Balanced Acc: {test_metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  F1-Score: {test_metrics['f1_score'] * 100:.2f}%")
    print(f"  Sensitivity: {test_metrics['recall'] * 100:.2f}%")
    print(f"  Specificity: {test_metrics['specificity'] * 100:.2f}%\n")

    results = {
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'train_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_f1_scores': val_f1_scores
        }
    }

    return results