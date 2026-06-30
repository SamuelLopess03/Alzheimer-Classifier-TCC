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
from ..data.preprocessing import denormalize_images, MedicalImagePreprocessor
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

def generate_ensemble_gradcam_image(
    models: List[nn.Module],
    img_tensor: torch.Tensor,
    device: torch.device,
    target_layer_path: Optional[str] = None
) -> np.ndarray:
    for m in models: m.eval()
    
    img_batch = img_tensor.unsqueeze(0).to(device)
    all_grayscale_cams = []

    if target_layer_path is None:
        arch_name = getattr(models[0], 'architecture_name', 'resnet50').lower()
        hyperparams_config = load_hyperparameters_config()
        arch_cfg = hyperparams_config['model_config'].get(arch_name)
        target_layer_path = arch_cfg.get('gradcam_target_layer') if arch_cfg else None

    for model in models:
        target_layer = get_target_layer(model, target_layer_path)
        cam = GradCAM(model=model, target_layers=[target_layer])
        
        grayscale_cam = cam(input_tensor=img_batch, targets=None)[0, :]
        all_grayscale_cams.append(grayscale_cam)

    ensemble_grayscale_cam = np.mean(all_grayscale_cams, axis=0)

    img_np = img_tensor.cpu().numpy()
    if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
        img_np = np.transpose(img_np, (1, 2, 0))
    if img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)

    preprocessor = MedicalImagePreprocessor(arch_name)
    mean, std = preprocessor.get_normalization_params()
    img_uint8 = denormalize_images(img_np, mean=mean, std=std)
    
    if img_uint8.ndim == 2:
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_uint8

    img_rgb = img_rgb.astype(np.float32) / 255.0
    
    return show_cam_on_image(img_rgb, ensemble_grayscale_cam, use_rgb=True)

def run_ensemble_gradcam(
    models: List[nn.Module],
    img_tensor: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    save_path: str,
    target_layer_path: Optional[str] = None,
    predicted_class_name: Optional[str] = None,
    confidence: Optional[float] = None,
    binary_confidence: Optional[float] = None
) -> str:
    visualization = generate_ensemble_gradcam_image(models, img_tensor, device, target_layer_path)
    
    if predicted_class_name is not None:
        has_two_lines = binary_confidence is not None and confidence is not None
        border_height = 60 if has_two_lines else 40
        
        h, w, c = visualization.shape
        new_img = np.zeros((h + border_height, w, c), dtype=np.uint8)
        new_img[:h, :, :] = visualization
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        color_dim = (180, 180, 180)
        thickness = 1
        
        if has_two_lines:
            line1 = f"Pred: {predicted_class_name}"
            scale1 = 0.44
            size1 = cv2.getTextSize(line1, font, scale1, thickness)[0]
            x1 = (w - size1[0]) // 2
            y1 = h + 20
            cv2.putText(new_img, line1, (x1, y1), font, scale1, color, thickness, cv2.LINE_AA)
            
            line2 = f"Demented: {binary_confidence*100:.1f}%  |  Class: {confidence*100:.1f}%"
            scale2 = 0.38
            size2 = cv2.getTextSize(line2, font, scale2, thickness)[0]
            x2 = (w - size2[0]) // 2
            y2 = h + 48
            cv2.putText(new_img, line2, (x2, y2), font, scale2, color_dim, thickness, cv2.LINE_AA)
        else:
            text = f"Pred: {predicted_class_name}"
            if confidence is not None:
                text += f" ({confidence*100:.1f}%)"
            scale = 0.42
            size = cv2.getTextSize(text, font, scale, thickness)[0]
            x = (w - size[0]) // 2
            y = h + (border_height + size[1]) // 2
            cv2.putText(new_img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        
        visualization = new_img

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