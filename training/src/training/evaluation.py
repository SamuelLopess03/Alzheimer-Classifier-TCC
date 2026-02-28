import os
from typing import Dict, Tuple, Optional, cast, Any, Union, Sequence

import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from pathlib import Path

from .trainer import validation_epoch
from ..visualization import (
    init_wandb_run, finish_wandb_run,
    plot_confusion_matrix, plot_roc_curve,
    log_confusion_matrix_figure, log_roc_curve_figure,
    close_figure
)
from ..evaluation import calculate_metrics_model
from ..data import StaticPreprocessedDataset
from ..utils import load_hyperparameters_config

def initialize_wandb_tracking(training_results: Dict, hyperparameters: dict,
                              optimizer: torch.optim.Optimizer,
                              criterion: nn.Module,
                              use_gradient_clipping: bool,
                              max_grad_norm: float) -> Tuple[bool, Optional[object]]:
    if not training_results.get('wandb_enabled', False):
        return False, None

    config = training_results.get('config', {})
    architecture_name = hyperparameters['architecture_name']
    model_type = training_results.get('model_type', 'Binário')
    class_names = training_results.get('class_names', [])
    save_path = training_results.get('save_path', '.')

    logging_config = config.get('logging', {}).get('wandb', {})
    wandb_project = logging_config.get('project', 'final_training')
    wandb_entity = logging_config.get('entity', None)

    run_name = f"{architecture_name}_evaluation_{model_type.lower()}"
    wandb_dir = os.path.join(save_path, 'wandb_logs')

    run = init_wandb_run(
        project_name=wandb_project,
        run_name=run_name,
        config={
            "model_type": model_type,
            "num_classes": len(class_names),
            "class_names": class_names,
            "optimizer": optimizer.__class__.__name__,
            "criterion": criterion.__class__.__name__,
            "use_gradient_clipping": use_gradient_clipping,
            "max_grad_norm": max_grad_norm if use_gradient_clipping else None,
            **hyperparameters
        },
        entity=wandb_entity,
        tags=["evaluation", architecture_name, model_type.lower()],
        group=f"{architecture_name}_eval_{model_type.lower()}",
        save_code=False,
        directory=wandb_dir
    )

    if run is None:
        print("Falha ao inicializar W&B. Continuando sem logging.\n")
        return False, None

    return True, run

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

    print(f"\n{'=' * 60}")
    print(f"AVALIAÇÃO NO DATASET DE TESTE ({model_type})")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {checkpoint_file}")
    print(f"  Modelo: {hyperparameters.get('architecture_name', 'N/A')}")
    print(f"  Classes: {class_names}")
    print(f"  GradCAM: {'Sim' if generate_gradcam else 'Não'}")
    print(f"{'=' * 60}\n")

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_file}\n"
            f"Execute primeiro o treinamento final (Fase 2)."
        )

    hyperparams_config = load_hyperparameters_config()
    hardware_config = hyperparams_config.get('hardware', {})
    num_workers = hardware_config.get('num_workers', 2)
    pin_memory = hardware_config.get('pin_memory', True)

    print("Criando dataset de teste (preprocessing estático)...\n")
    test_preprocessed = StaticPreprocessedDataset(
        subset_dataset=test_dataset,
        architecture_name=hyperparameters['architecture_name']
    )

    test_loader = DataLoader(
        test_preprocessed,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    wandb_enabled, run = initialize_wandb_tracking(
        training_results, hyperparameters, optimizer, criterion,
        hyperparameters.get('use_gradient_clipping', True),
        hyperparameters.get('max_grad_norm', 1.0)
    )

    y_true, y_pred, y_pred_proba, test_metrics = evaluate_on_test_set(
        model, test_loader, criterion, device,
        checkpoint_file, class_names,
        is_multiclass, wandb_enabled
    )

    print_test_metrics(test_metrics, model_type, is_multiclass)

    generate_and_save_visualizations(
        y_true, y_pred_proba, test_metrics,
        class_names, is_multiclass,
        save_path, model_type,
        wandb_enabled, run
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

            if wandb_enabled and run is not None:
                gradcam_images = []
                for img_file in Path(gradcam_path).glob("*.png"):
                    gradcam_images.append(wandb.Image(str(img_file), caption=img_file.stem))

                if gradcam_images:
                    wandb.log({f"{model_type.lower()}/gradcam": gradcam_images})
                    print(f"Grad-CAM enviado para W&B\n")

        except Exception as e:
            print(f"Erro ao gerar Grad-CAM: {str(e)}\n")

    if wandb_enabled and run is not None:
        finish_wandb_run(quiet=False)

    results = {
        'test_metrics': test_metrics,
        'checkpoint_path': checkpoint_file,
        'gradcam_path': gradcam_path if generate_gradcam else None
    }

    print(f"\n{'=' * 60}")
    print(f"AVALIAÇÃO CONCLUÍDA ({model_type})")
    print(f"{'=' * 60}\n")

    return results
