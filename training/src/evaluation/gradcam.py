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
from typing import Dict, List, Optional

from ..models import get_target_layer
from ..data import denormalize_images
from ..utils import load_hyperparameters_config

def _collect_class_samples(
    test_loader: DataLoader, 
    device: torch.device, 
    num_classes: int, 
    samples_per_class: int
) -> Dict[int, List[torch.Tensor]]:
    class_samples = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.cpu().numpy()

            for img, label in zip(images, labels):
                if len(class_samples[label]) < samples_per_class:
                    class_samples[label].append(img)

            if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                break
                
    return class_samples

def run_single_gradcam(
    model: nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    save_path: str,
    target_layer_path: Optional[str] = None
) -> str:
    model.eval()
    
    if target_layer_path is None:
        arch_name = getattr(model, 'architecture_name', 'resnet50').lower()
        hyperparams_config = load_hyperparameters_config()
        arch_cfg = hyperparams_config['model_config'].get(arch_name)
        target_layer_path = arch_cfg.get('gradcam_target_layer') if arch_cfg else None

    target_layer = get_target_layer(model, target_layer_path)
    cam = GradCAM(model=model, target_layers=[target_layer])

    img_batch = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_batch)
        pred_class = output.argmax(dim=1).item()
        pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()

    grayscale_cam = cam(input_tensor=img_batch, targets=None)[0, :]

    img_np = img_tensor.cpu().numpy()
    img_uint8 = denormalize_images(img_np, mean=[0.449], std=[0.226])
    
    if img_uint8.ndim == 2:
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_uint8

    img_rgb = img_rgb.astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
    
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    return save_path

def generate_gradcam_visualizations(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list,
    save_path: str,
    architecture_name: str,
    num_samples: int = 10,
    samples_per_class: int = 2
) -> None:
    print(f"\n{'-' * 60}")
    print("GERANDO VISUALIZAÇÕES GRAD-CAM")
    print(f"{'-' * 60}\n")

    model.architecture_name = architecture_name
    
    class_samples = _collect_class_samples(test_loader, device, len(class_names), samples_per_class)
    
    sample_count = 0
    for class_idx, samples in class_samples.items():
        class_name = class_names[class_idx]
        for idx, img_tensor in enumerate(samples):
            if sample_count >= num_samples:
                break
            
            filename = f"gradcam_true_{class_name}_sample_{idx}.png"
            dest = os.path.join(save_path, filename)
            
            run_single_gradcam(
                model=model,
                img_tensor=img_tensor,
                device=device,
                class_names=class_names,
                save_path=dest
            )
            sample_count += 1

    print(f"\nVisualizações Grad-CAM geradas em: {save_path}\n")

def log_gradcam_to_wandb(gradcam_path: str, model_type: str):
    if wandb.run is None:
        return

    gradcam_images = []
    for img_file in Path(gradcam_path).glob("*.png"):
        gradcam_images.append(wandb.Image(str(img_file), caption=img_file.stem))

    if gradcam_images:
        wandb.log({f"{model_type.lower()}/gradcam": gradcam_images})
        print(f"Grad-CAM enviado para W&B\n")