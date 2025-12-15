import os

import wandb
from typing import Dict, Tuple, Optional, cast, Any, Union, Sequence
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from pathlib import Path

from .trainer import get_training_config, train_epoch, validation_epoch
from training.src.visualization import (
    init_wandb_run, finish_wandb_run,
    plot_confusion_matrix, plot_roc_curve,
    log_confusion_matrix_figure, log_roc_curve_figure,
    close_figure
)
from training.src.evaluation import calculate_metrics_model

def setup_training_environment(is_multiclass: bool) -> Dict:
    config = get_training_config(is_multiclass)

    model_config = config['model']
    training_config = config['training']
    checkpoint_config = config['checkpoint']

    save_path = checkpoint_config['save_path']
    os.makedirs(save_path, exist_ok=True)

    gradcam_path = Path("../../../shared/outputs/gradcam")
    gradcam_path.mkdir(parents=True, exist_ok=True)

    model_type = "Multiclasse" if is_multiclass else "Binário"
    checkpoint_file = os.path.join(save_path, f"best_model.pth")

    return {
        'config': config,
        'model_config': model_config,
        'training_config': training_config,
        'checkpoint_config': checkpoint_config,
        'save_path': save_path,
        'gradcam_path': str(gradcam_path),
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

def initialize_wandb_tracking(env_config: Dict, hyperparameters: dict,
                              optimizer: torch.optim.Optimizer,
                              criterion: nn.Module,
                              use_gradient_clipping: bool,
                              max_grad_norm: float) -> Tuple[bool, Optional[object]]:
    if not env_config['wandb_enabled']:
        return False, None

    config = env_config['config']
    architecture_name = hyperparameters['architecture_name']
    wandb_project = config['logging']['wandb'].get('project', 'final_training')
    wandb_entity = config['logging']['wandb'].get('entity', None)

    run_name = f"{architecture_name}_final_training_{env_config['model_type'].lower()}"
    wandb_dir = os.path.join(env_config['save_path'], 'wandb_logs')

    run = init_wandb_run(
        project_name=wandb_project,
        run_name=run_name,
        config={
            "model_type": env_config['model_type'],
            "num_classes": len(env_config['class_names']),
            "class_names": env_config['class_names'],
            "num_epochs": env_config['num_epochs'],
            "patience": env_config['early_stopping_patience'],
            "monitor_metric": env_config['monitor_metric'],
            "optimizer": optimizer.__class__.__name__,
            "criterion": criterion.__class__.__name__,
            "use_gradient_clipping": use_gradient_clipping,
            "max_grad_norm": max_grad_norm if use_gradient_clipping else None,
            "scheduler": env_config['training_config'].get('scheduler', {}).get('type', 'None'),
            **hyperparameters
        },
        entity=wandb_entity,
        tags=["final_training", architecture_name, env_config['model_type'].lower()],
        group=f"{architecture_name}_final_{env_config['model_type'].lower()}",
        save_code=True,
        directory=wandb_dir
    )

    if run is None:
        print("Falha ao inicializar W&B. Continuando sem logging.\n")
        return False, None

    return True, run

def print_training_header(env_config: Dict):
    print(f"\n{'=' * 80}")
    print(f"TREINAMENTO FINAL DO MODELO ({env_config['model_type']})")
    print(f"{'=' * 80}")
    print(f"Configurações:")
    print(f"   Épocas: {env_config['num_epochs']}")
    print(f"   Patience: {env_config['early_stopping_patience']}")
    print(f"   Monitor: {env_config['monitor_metric']}")
    print(f"   Checkpoint: {env_config['checkpoint_file']}")
    print(f"   Grad-CAM: {env_config['gradcam_path']}\n")

def train_single_epoch(model: nn.Module, train_loader: DataLoader,
                       criterion: nn.Module, optimizer: torch.optim.Optimizer,
                       device: torch.device, use_gradient_clipping: bool,
                       max_grad_norm: float) -> Tuple[float, float]:
    return train_epoch(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        apply_clipping=use_gradient_clipping,
        max_grad_norm=max_grad_norm,
        use_amp=True
    )

def validate_single_epoch(model: nn.Module, val_loader: DataLoader,
                          criterion: nn.Module, device: torch.device,
                          class_names: list, is_multiclass: bool,
                          train_loss: float, log_to_wandb: bool = False) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict]:
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
    print(f"Val Loss: {metrics['val_loss']:.4f} | Val Acc: {metrics['accuracy'] * 100:.2f}%")
    print(f"Val F1: {metrics['f1_score'] * 100:.2f}% | Balanced Acc: {metrics['balanced_accuracy'] * 100:.2f}%")

def get_target_layer(model: nn.Module, architecture_name: str) -> nn.Module:
    arch_lower = architecture_name.lower()

    if 'resnext' in arch_lower:
        return cast(Any, model.layer4[-1])

    elif 'efficientnet' in arch_lower:
        return cast(Any, model.features[-1])

    elif 'densenet' in arch_lower:
        return model.features.denseblock4

    elif 'vit' in arch_lower:
        block = cast(Any, model.blocks[-1])
        return block.norm1

    elif 'swin' in arch_lower:
        layer = cast(Any, model.layers[-1])
        block = layer.blocks[-1]
        return block.norm1

    elif 'convnext' in arch_lower:
        return cast(Any, model.stages[-1])

    else:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module

        return list(model.modules())[-1]

def denormalize_images(
    img: np.ndarray,
    mean: Union[Sequence[float], np.ndarray],
    std: Union[Sequence[float], np.ndarray]
) -> np.ndarray:
    mean = np.array(mean)
    std = np.array(std)

    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)

    return img_uint8

def generate_gradcam_visualizations(model: nn.Module,
                                    test_loader: DataLoader,
                                    device: torch.device,
                                    class_names: list,
                                    save_path: str,
                                    architecture_name: str,
                                    num_samples: int = 10,
                                    samples_per_class: int = 2) -> None:
    print(f"\n{'-' * 60}")
    print("GERANDO VISUALIZAÇÕES GRAD-CAM")
    print(f"{'-' * 60}\n")

    model.eval()

    target_layer = get_target_layer(model, architecture_name)
    print(f"Camada alvo para Grad-CAM: {target_layer.__class__.__name__}\n")

    cam = GradCAM(model=model, target_layers=[target_layer])

    class_samples = {i: [] for i in range(len(class_names))}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.cpu().numpy()

            for img, label in zip(images, labels):
                if len(class_samples[label]) < samples_per_class:
                    class_samples[label].append(img)

            if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                break

    sample_count = 0
    for class_idx, samples in class_samples.items():
        class_name = class_names[class_idx]

        for idx, img_tensor in enumerate(samples):
            if sample_count >= num_samples:
                break

            img_tensor = img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                pred_class = output.argmax(dim=1).item()
                pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()

            grayscale_cam = cam(input_tensor=img_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]

            img_np = img_tensor.cpu().squeeze(0).numpy()

            img_uint8 = denormalize_images(img_np, mean=[0.449], std=[0.226])

            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

            img_rgb = img_rgb.astype(np.float32) / 255.0

            visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

            filename = f"gradcam_true_{class_name}_pred_{class_names[pred_class]}_conf_{pred_prob:.2f}_sample_{idx}.png"
            save_file = os.path.join(save_path, filename)

            cv2.imwrite(save_file, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            sample_count += 1
            print(f"Grad-CAM salvo: {filename}")

    print(f"\nTotal de {sample_count} visualizações Grad-CAM geradas em: {save_path}\n")

def evaluate_on_test_set(model: nn.Module, test_loader: DataLoader,
                         criterion: nn.Module, device: torch.device,
                         checkpoint_file: str, class_names: list,
                         is_multiclass: bool, wandb_enabled: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    checkpoint = torch.load(checkpoint_file, map_location=device)
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
        log_to_wandb=wandb_enabled,
        is_multiclass=is_multiclass
    )

    return y_true, y_pred, y_pred_proba, test_metrics

def print_test_metrics(test_metrics: Dict, model_type: str, is_multiclass: bool):
    print(f"\nResultados Finais (Test Set - {model_type}):")
    print(f"{'-' * 60}")
    print(f"  Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Balanced Acc: {test_metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  F1-Score: {test_metrics['f1_score'] * 100:.2f}%")

    if not is_multiclass:
        print(f"  Sensitivity: {test_metrics['recall'] * 100:.2f}%")
        print(f"  Specificity: {test_metrics['specificity'] * 100:.2f}%")
        print(f"  Precision: {test_metrics['precision'] * 100:.2f}%")
        print(f"  NPV: {test_metrics['negative_predictive_value'] * 100:.2f}%")
    else:
        print(f"  F1 (Macro): {test_metrics.get('f1_macro', 0) * 100:.2f}%")
        print(f"  Precision (Weighted): {test_metrics['precision'] * 100:.2f}%")
        print(f"  Recall (Weighted): {test_metrics['recall'] * 100:.2f}%")

    print(f"  MCC: {test_metrics['matthews_correlation_coefficient']:.4f}")
    print(f"  Cohen's Kappa: {test_metrics['cohen_kappa']:.4f}")
    print(f"{'-' * 60}\n")

def generate_and_save_visualizations(y_true: np.ndarray, y_pred_proba: np.ndarray, test_metrics: Dict,
                                     class_names: list, is_multiclass: bool,
                                     save_path: str, model_type: str,
                                     wandb_enabled: bool, run: Optional[object]):
    print(f"\n{'-' * 60}")
    print("GERANDO VISUALIZAÇÕES MATRIZ DE CONFUSÃO E CURVA AUC-ROC")
    print(f"{'-' * 60}\n")

    cm = np.array(test_metrics['confusion_matrix'])
    fig_cm = plot_confusion_matrix(
        cm=cm,
        metrics=test_metrics,
        class_names=class_names,
        is_multiclass=is_multiclass
    )

    cm_path = os.path.join(save_path, f"confusion_matrix_{model_type.lower().replace(' ', '_')}.png")
    fig_cm.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion Matrix salva em: {cm_path}\n")

    if wandb_enabled and run is not None:
        log_confusion_matrix_figure(fig_cm, key=f"{model_type.lower()}/confusion_matrix")

    close_figure(fig_cm)

    fig_roc = plot_roc_curve(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        class_names=class_names,
        is_multiclass=is_multiclass
    )

    roc_path = os.path.join(save_path, f"roc_curve_{model_type.lower().replace(' ', '_')}.png")
    fig_roc.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"ROC Curve salva em: {roc_path}\n")

    if wandb_enabled and run is not None:
        log_roc_curve_figure(fig_roc, key=f"{model_type.lower()}/roc_curve")

    close_figure(fig_roc)

def train_final_model(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        hyperparameters: dict,
        device: torch.device,
        is_multiclass: bool = False,
        use_gradient_clipping: bool = True,
        max_grad_norm: float = 1.0,
        generate_gradcam: bool = True,
        gradcam_samples: int = 10
) -> Dict:
    env_config = setup_training_environment(is_multiclass)
    scheduler = setup_scheduler(optimizer, env_config['training_config'])
    print_training_header(env_config)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_f1_scores = []

    for epoch in range(env_config['num_epochs']):
        train_loss, train_acc = train_single_epoch(
            model, train_loader, criterion, optimizer, device,
            use_gradient_clipping, max_grad_norm
        )

        y_true, y_pred, y_pred_proba, val_loss, metrics = validate_single_epoch(
            model, val_loader, criterion, device,
            env_config['class_names'], is_multiclass, train_loss, log_to_wandb=False
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
    print(f"AVALIAÇÃO NO DATASET DE TESTE ({env_config['model_type']})")
    print(f"{'-' * 60}\n")

    wandb_enabled, run = initialize_wandb_tracking(
        env_config, hyperparameters, optimizer, criterion,
        use_gradient_clipping, max_grad_norm
    )

    y_true, y_pred, y_pred_proba, test_metrics = evaluate_on_test_set(
        model, test_loader, criterion, device,
        env_config['checkpoint_file'], env_config['class_names'],
        is_multiclass, wandb_enabled
    )

    print_test_metrics(test_metrics, env_config['model_type'], is_multiclass)

    generate_and_save_visualizations(
        y_true, y_pred_proba, test_metrics,
        env_config['class_names'], is_multiclass,
        env_config['save_path'], env_config['model_type'],
        wandb_enabled, run
    )

    if generate_gradcam:
        try:
            generate_gradcam_visualizations(
                model=model,
                test_loader=test_loader,
                device=device,
                class_names=env_config['class_names'],
                save_path=env_config['gradcam_path'],
                architecture_name=hyperparameters['architecture_name'],
                num_samples=gradcam_samples,
                samples_per_class=max(2, gradcam_samples // len(env_config['class_names']))
            )

            if wandb_enabled and run is not None:
                gradcam_images = []
                for img_file in Path(env_config['gradcam_path']).glob("*.png"):
                    gradcam_images.append(wandb.Image(str(img_file), caption=img_file.stem))

                if gradcam_images:
                    wandb.log({f"{env_config['model_type'].lower()}/gradcam": gradcam_images})
                    print(f"Grad-CAM enviado para W&B\n")

        except Exception as e:
            print(f"Erro ao gerar Grad-CAM: {str(e)}\n")

    if wandb_enabled and run is not None:
        finish_wandb_run(quiet=False)

    results = {
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'train_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_f1_scores': val_f1_scores
        },
        'checkpoint_path': env_config['checkpoint_file'],
        'gradcam_path': env_config['gradcam_path'] if generate_gradcam else None
    }

    print(f"{'=' * 80}")
    print(f"TREINAMENTO FINAL CONCLUÍDO ({env_config['model_type']})")
    print(f"{'=' * 80}\n")

    return results