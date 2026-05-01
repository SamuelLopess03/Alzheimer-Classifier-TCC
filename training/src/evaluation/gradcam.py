import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import Dict, List, Optional, Union

from ..models import get_target_layer
from ..data.preprocessing import denormalize_images
from ..data.subject_manager import get_central_slices_per_class, resolve_dataset_chain
from src.utils.config import load_hyperparameters_config

def _collect_class_samples(
    test_loader: DataLoader, 
    device: torch.device, 
    num_classes: int, 
    samples_per_class: int
) -> Dict[int, List[torch.Tensor]]:
    wrapper_dataset = getattr(test_loader, 'dataset', None)
    
    base_ds, base_indices = resolve_dataset_chain(wrapper_dataset)

    if base_ds and base_indices is None:
        base_indices = np.arange(len(base_ds))

    if base_ds and hasattr(base_ds, 'samples') and base_indices is not None:
        class_indices = get_central_slices_per_class(base_ds, num_classes, samples_per_class, indices=base_indices)
        
        base_to_wrapper = {base_idx: i for i, base_idx in enumerate(base_indices)}
        
        return {
            class_idx: [wrapper_dataset[base_to_wrapper[idx]][0].to(device) for idx in indices]
            for class_idx, indices in class_indices.items()
        }

    print("[AVISO] Dataset raiz incompatível (sem 'samples'). O Grad-CAM não gerará imagens, pois o limite por paciente não pode ser garantido.")
    return {i: [] for i in range(num_classes)}

def run_single_gradcam(
    model: nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    save_path: str,
    target_layer_path: Optional[str] = None
) -> str:
    """Gera Grad-CAM para um único modelo."""
    return run_ensemble_gradcam(
        models=[model],
        img_tensor=img_tensor,
        device=device,
        class_names=class_names,
        save_path=save_path,
        target_layer_path=target_layer_path
    )

def run_ensemble_gradcam(
    models: List[nn.Module],
    img_tensor: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    save_path: str,
    target_layer_path: Optional[str] = None
) -> str:
    """Gera o mapa consensual do Ensemble (média dos mapas de calor)."""
    for m in models: m.eval()
    
    img_batch = img_tensor.unsqueeze(0).to(device)
    all_grayscale_cams = []

    # Configuração da arquitetura baseada no primeiro modelo
    if target_layer_path is None:
        arch_name = getattr(models[0], 'architecture_name', 'resnet50').lower()
        hyperparams_config = load_hyperparameters_config()
        arch_cfg = hyperparams_config['model_config'].get(arch_name)
        target_layer_path = arch_cfg.get('gradcam_target_layer') if arch_cfg else None

    # Acumula os mapas de calor de todos os modelos
    for model in models:
        target_layer = get_target_layer(model, target_layer_path)
        cam = GradCAM(model=model, target_layers=[target_layer])
        
        # O Grad-CAM gera um mapa de 0 a 1 para a entrada
        grayscale_cam = cam(input_tensor=img_batch, targets=None)[0, :]
        all_grayscale_cams.append(grayscale_cam)

    # Média dos mapas de calor (Consenso do Ensemble)
    ensemble_grayscale_cam = np.mean(all_grayscale_cams, axis=0)

    # Preparação da imagem original para overlay
    img_np = img_tensor.cpu().numpy()
    if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
        img_np = np.transpose(img_np, (1, 2, 0))
    if img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)

    img_uint8 = denormalize_images(img_np, mean=[0.449], std=[0.226])
    
    if img_uint8.ndim == 2:
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_uint8

    img_rgb = img_rgb.astype(np.float32) / 255.0
    
    # Renderiza o mapa consensual sobre a imagem
    visualization = show_cam_on_image(img_rgb, ensemble_grayscale_cam, use_rgb=True)
    
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    return save_path

def generate_gradcam_visualizations(
    models: Union[nn.Module, List[nn.Module]],
    test_loader: DataLoader,
    device: torch.device,
    class_names: list,
    save_path: str,
    architecture_name: str,
    samples_per_class: int = 10
) -> None:
    if not isinstance(models, list):
        models = [models]

    print(f"\n{'-' * 60}")
    print(f"GERANDO VISUALIZAÇÕES GRAD-CAM (ENSEMBLE: {len(models)} modelos)")
    print(f"{'-' * 60}\n")

    for m in models: m.architecture_name = architecture_name
    
    class_samples = _collect_class_samples(test_loader, device, len(class_names), samples_per_class)
    
    for class_idx, samples in class_samples.items():
        class_name = class_names[class_idx]
        class_save_path = os.path.join(save_path, class_name)
        os.makedirs(class_save_path, exist_ok=True)
        
        for idx, img_tensor in enumerate(samples):
            filename = f"gradcam_ensemble_{class_name}_sample_{idx}.png"
            dest = os.path.join(class_save_path, filename)
            
            run_ensemble_gradcam(
                models=models,
                img_tensor=img_tensor,
                device=device,
                class_names=class_names,
                save_path=dest
            )

    print(f"\nVisualizações Grad-CAM geradas em: {save_path}\n")

def log_gradcam_to_wandb(gradcam_path: str, model_type: str):
    if wandb.run is None:
        return

    gradcam_images = []
    for img_file in Path(gradcam_path).rglob("*.png"):
        gradcam_images.append(wandb.Image(str(img_file), caption=img_file.stem))

    if gradcam_images:
        wandb.log({f"{model_type.lower()}/gradcam": gradcam_images})
        print(f"Grad-CAM enviado para W&B\n")