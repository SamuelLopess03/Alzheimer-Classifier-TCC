import numpy as np
import cv2
import torch
from PIL import Image
from typing import Dict, Tuple, Optional

class MedicalImagePreprocessor:
    def __init__(self, architecture_name: str):
        self.architecture_name = architecture_name.lower()
        self.config = self._get_architecture_config()

    def _get_architecture_config(self) -> Dict:
        configs = {
            "resnext50_32x4d": {
                "image_size": 224,
                "channels": 1,
                "mean": [0.449],  # Mean para imagens grayscale de RM
                "std": [0.226],
                "description": "ResNeXt-50 optimized for 224x224"
            },
            "convnext_tiny": {
                "image_size": 224,
                "channels": 1,
                "mean": [0.449],
                "std": [0.226],
                "description": "ConvNeXt Tiny optimized for 224x224"
            },
            "efficientnetv2_s": {
                "image_size": 384,
                "channels": 1,
                "mean": [0.449],
                "std": [0.226],
                "description": "EfficientNetV2-S optimized for 384x384"
            },
            "densenet121": {
                "image_size": 224,
                "channels": 1,
                "mean": [0.449],
                "std": [0.226],
                "description": "DenseNet-121 optimized for 224x224"
            },
            "vit_b_16": {
                "image_size": 224,
                "channels": 1,
                "mean": [0.449],
                "std": [0.226],
                "description": "Vision Transformer Base with 16x16 patches"
            },
            "swin_v2_tiny": {
                "image_size": 256,
                "channels": 1,
                "mean": [0.449],
                "std": [0.226],
                "description": "Swin Transformer V2 Tiny optimized for 256x256"
            }
        }

        if self.architecture_name not in configs:
            print(f"\nArquitetura '{self.architecture_name}' não reconhecida. Usando config padrão (ResNeXt50).\n")
            return configs["resnext50_32x4d"]

        return configs[self.architecture_name]

    def get_image_size(self) -> int:
        return self.config["image_size"]

    def get_channels(self) -> int:
        return self.config["channels"]

    def get_normalization_params(self) -> Tuple[list, list]:
        return self.config["mean"], self.config["std"]

    def print_config(self):
        print(f"\n{'-' * 60}")
        print(f"PREPROCESSOR CONFIG: {self.architecture_name.upper()}")
        print(f"{'=' * 60}")
        print(f"  Image Size: {self.config['image_size']}x{self.config['image_size']}")
        print(f"  Channels: {self.config['channels']}")
        print(f"  Mean: {self.config['mean']}")
        print(f"  Std: {self.config['std']}")
        print(f"  Description: {self.config['description']}")
        print(f"{'-' * 60}\n")

def convert_pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)

def convert_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.numpy()

    # Se for channels-first, converter para channels-last
    if array.ndim == 3 and array.shape[0] in [1, 3]:
        array = np.transpose(array, (1, 2, 0))

    return array

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        if image.shape[2] == 1:
            return image.squeeze(-1)
        elif image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Formato de imagem não suportado: shape={image.shape}")
    else:
        raise ValueError(f"Dimensões de imagem inválidas: {image.ndim}D")

def normalize_image(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    return image

def prepare_image_for_augmentation(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = convert_pil_to_numpy(image)
    elif isinstance(image, torch.Tensor):
        image = convert_tensor_to_numpy(image)

    image = convert_to_grayscale(image)

    image = normalize_image(image)

    return image

def validate_image_format(
        image: np.ndarray,
        expected_channels: int = 1
) -> Tuple[bool, Optional[str]]:
    if image.ndim not in [2, 3]:
        return False, f"Dimensões inválidas: {image.ndim}D (esperado 2D ou 3D)"

    if image.ndim == 3:
        if image.shape[2] != expected_channels:
            return False, f"Canais inválidos: {image.shape[2]} (esperado {expected_channels})"

    if image.dtype not in [np.uint8, np.float32, np.float64]:
        return False, f"Tipo inválido: {image.dtype} (esperado uint8, float32, ou float64)"

    if image.dtype == np.uint8:
        if image.min() < 0 or image.max() > 255:
            return False, f"Range inválido para uint8: [{image.min()}, {image.max()}]"
    elif image.dtype in [np.float32, np.float64]:
        if 1.0 < image.max() <= 255:
            pass
        elif image.min() < 0 or image.max() > 1.0:
            return False, f"Range inválido para float: [{image.min():.2f}, {image.max():.2f}]"

    return True, None