import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from typing import Tuple

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

            if scaler is not None and use_amp:
                with autocast(device_type=device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[AVISO] NaN/Inf loss detectado no batch {batch_idx}. Pulando...")
                continue

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                if apply_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if apply_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[AVISO] CUDA Out of Memory no batch {batch_idx}. Limpando cache...")
                torch.cuda.empty_cache()
                continue
            raise e

    avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    train_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

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
                    continue

                val_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.append(predicted)
                all_labels.append(labels)
                all_probabilities.append(probabilities)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise e

    avg_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

    y_pred = torch.cat(all_predictions).cpu().numpy()
    y_true = torch.cat(all_labels).cpu().numpy()
    y_pred_proba = torch.cat(all_probabilities).cpu().numpy()

    return y_true, y_pred, y_pred_proba, avg_loss
