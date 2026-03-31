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

def _process_visualizations(
    model: nn.Module,
    cam: GradCAM,
    class_samples: Dict[int, List[torch.Tensor]],
    class_names: List[str],
    device: torch.device,
    save_path: str,
    num_samples: int
):
    sample_count = 0
    
    for class_idx, samples in class_samples.items():
        class_name = class_names[class_idx]

        for idx, img_tensor in enumerate(samples):
            if sample_count >= num_samples:
                return

            _process_single_gradcam(
                model=model,
                cam=cam,
                img_tensor=img_tensor,
                device=device,
                class_names=class_names,
                class_name=class_name,
                save_path=save_path,
                sample_idx=idx
            )
            sample_count += 1

def _process_single_gradcam(
    model: nn.Module,
    cam: GradCAM,
    img_tensor: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    class_name: str,
    save_path: str,
    sample_idx: int
):
    img_batch = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_batch)
        pred_class = output.argmax(dim=1).item()
        pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()

    grayscale_cam = cam(input_tensor=img_batch, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    img_np = img_tensor.cpu().squeeze(0).numpy()
    img_uint8 = denormalize_images(img_np, mean=[0.449], std=[0.226])
    
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0

    visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
    
    filename = f"gradcam_true_{class_name}_pred_{class_names[pred_class]}_conf_{pred_prob:.2f}_sample_{sample_idx}.png"
    save_file = os.path.join(save_path, filename)
    
    cv2.imwrite(save_file, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM salvo: {filename}")

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

    model.eval()
    
    hyperparams_config = load_hyperparameters_config()
    arch_cfg = hyperparams_config['model_config'].get(architecture_name.lower())
    target_layer_path = arch_cfg.get('gradcam_target_layer') if arch_cfg else None

    target_layer = get_target_layer(model, target_layer_path)
    print(f"Camada alvo para Grad-CAM: {target_layer.__class__.__name__}\n")

    cam = GradCAM(model=model, target_layers=[target_layer])

    class_samples = _collect_class_samples(test_loader, device, len(class_names), samples_per_class)

    _process_visualizations(
        model=model,
        cam=cam,
        class_samples=class_samples,
        class_names=class_names,
        device=device,
        save_path=save_path,
        num_samples=num_samples
    )

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