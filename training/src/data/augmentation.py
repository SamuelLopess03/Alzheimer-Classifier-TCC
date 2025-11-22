import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, Subset
from typing import List
from collections import Counter

from .preprocessing import MedicalImagePreprocessor, prepare_image_for_augmentation

def get_alzheimer_grayscale_augmentation(
        architecture_name: str,
        dataset_size: int,
        is_training: bool = True
) -> alb.Compose:
    preprocessor = MedicalImagePreprocessor(architecture_name)
    config = preprocessor.config

    is_transformer = architecture_name.lower() in ['vit_b_16', 'swin_v2_tiny']

    if is_training:
        if dataset_size < 20000:
            print(f"\nAugmentação Pesada: Dataset pequeno (<20k) - {'Transformer' if is_transformer else 'CNN'}\n")

            if is_transformer:
                # Transformers: augmentation mais leve para preservar estrutura
                augmentations = [
                    alb.Resize(config["image_size"], config["image_size"]),

                    # Geométricas
                    alb.HorizontalFlip(p=0.4),
                    alb.Rotate(limit=3, p=0.3),

                    # Intensidade
                    alb.CLAHE(clip_limit=1.5, tile_grid_size=(6, 6), p=0.5),
                    alb.RandomBrightnessContrast(
                        brightness_limit=0.08,
                        contrast_limit=0.1,
                        p=0.4
                    ),

                    # Normalização e conversão para tensor
                    alb.Normalize(mean=config["mean"], std=config["std"]),
                    ToTensorV2()
                ]
            else:
                # CNNs: augmentation pesada
                augmentations = [
                    alb.Resize(config["image_size"], config["image_size"]),

                    # Geométricas
                    alb.HorizontalFlip(p=0.5),
                    alb.Rotate(limit=5, p=0.4),
                    alb.Affine(
                        scale=(0.95, 1.05),
                        translate_percent=(-0.05, 0.05),
                        rotate=(-3, 3),
                        p=0.3
                    ),

                    # Intensidade
                    alb.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.7),
                    alb.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.15,
                        p=0.6
                    ),
                    alb.RandomGamma(gamma_limit=(90, 110), p=0.5),

                    # Blur/Sharpen
                    alb.OneOf([
                        alb.GaussianBlur(blur_limit=(1, 3), p=1.0),
                        alb.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=1.0),
                    ], p=0.3),

                    # Normalização e conversão para tensor
                    alb.Normalize(mean=config["mean"], std=config["std"]),
                    ToTensorV2()
                ]

        elif dataset_size < 50000:
            print(f"\nAugmentação Moderada: Dataset médio (<50k)\n")
            augmentations = [
                alb.Resize(config["image_size"], config["image_size"]),

                # Geométricas moderadas
                alb.HorizontalFlip(p=0.5),
                alb.Rotate(limit=3, p=0.3),

                # Intensidade moderada
                alb.CLAHE(clip_limit=1.5, tile_grid_size=(6, 6), p=0.5),
                alb.RandomBrightnessContrast(
                    brightness_limit=0.08,
                    contrast_limit=0.1,
                    p=0.4
                ),
                alb.RandomGamma(gamma_limit=(95, 105), p=0.3),

                # Blur leve
                alb.GaussianBlur(blur_limit=(1, 2), p=0.1),

                # Normalização e conversão para tensor
                alb.Normalize(mean=config["mean"], std=config["std"]),
                ToTensorV2()
            ]

        else:
            print(f"\nAugmentação Leve: Dataset grande (≥50k)\n")
            augmentations = [
                alb.Resize(config["image_size"], config["image_size"]),

                # Geométricas leves
                alb.HorizontalFlip(p=0.3),
                alb.Rotate(limit=2, p=0.2),

                # Intensidade leve
                alb.CLAHE(clip_limit=1.2, tile_grid_size=(8, 8), p=0.3),
                alb.RandomBrightnessContrast(
                    brightness_limit=0.05,
                    contrast_limit=0.05,
                    p=0.2
                ),

                # Normalização e conversão para tensor
                alb.Normalize(mean=config["mean"], std=config["std"]),
                ToTensorV2()
            ]

    else:
        # Validação/Teste: apenas preprocessing
        augmentations = [
            alb.Resize(config["image_size"], config["image_size"]),
            alb.Normalize(mean=config["mean"], std=config["std"]),
            ToTensorV2()
        ]

    return alb.Compose(augmentations)

def create_synthetic_augmentation_for_minority(
        architecture_name: str
) -> alb.Compose:
    is_transformer = architecture_name.lower() in ['vit_b_16', 'swin_v2_tiny']

    if is_transformer:
        # Transformers: augmentations geométricas mais sutis
        augmentations = [
            alb.VerticalFlip(p=0.3),

            alb.Affine(
                translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                scale=(0.9, 1.1),
                rotate=(-7, 7),
                p=0.5
            ),

            # Intensidade
            alb.RandomToneCurve(scale=0.15, p=0.4),
            alb.ColorJitter(
                brightness=0.1,
                contrast=0.15,
                saturation=0.0,
                hue=0.0,
                p=0.4
            ),

            # Ruído
            alb.GaussNoise(std_range=(0.05, 0.15), mean_range=(0.0, 0.0), p=0.4),
        ]
    else:
        # CNNs: augmentations mais agressivas
        augmentations = [
            alb.VerticalFlip(p=0.3),

            alb.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.85, 1.15),
                rotate=(-10, 10),
                p=0.6
            ),

            alb.ElasticTransform(
                alpha=1,
                sigma=20,
                p=0.3
            ),

            # Intensidade
            alb.RandomToneCurve(scale=0.2, p=0.5),
            alb.Equalize(p=0.3),
            alb.ColorJitter(
                brightness=0.15,
                contrast=0.2,
                saturation=0.0,
                hue=0.0,
                p=0.5
            ),

            # Ruído
            alb.GaussNoise(std_range=(0.05, 0.15), mean_range=(0.0, 0.0), p=0.4),
            alb.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.3),
                p=0.3
            ),
        ]

    return alb.Compose(augmentations)

class DynamicAugmentationDataset(Dataset):
    def __init__(self, subset_dataset: Subset, architecture_name: str):
        self.subset_dataset = subset_dataset
        self.architecture_name = architecture_name

        self.transform = get_alzheimer_grayscale_augmentation(
            architecture_name=architecture_name,
            dataset_size=len(subset_dataset),
            is_training=True
        )

    def __len__(self):
        return len(self.subset_dataset)

    def __getitem__(self, idx):
        image, label = self.subset_dataset[idx]

        image = prepare_image_for_augmentation(image, ensure_uint8=True)

        transformed = self.transform(image=image)
        processed_image = transformed['image']

        return processed_image, label

class StaticPreprocessedDataset(Dataset):
    def __init__(self, subset_dataset: Subset, architecture_name: str):
        self.subset_dataset = subset_dataset
        self.architecture_name = architecture_name

        self.transform = get_alzheimer_grayscale_augmentation(
            architecture_name=architecture_name,
            dataset_size=len(subset_dataset),
            is_training=False
        )

        print(f"\nPré-processando {len(subset_dataset)} imagens (estático)...\n")
        self.preprocessed_data = []
        self.labels = []

        for idx in range(len(subset_dataset)):
            image, label = subset_dataset[idx]
            image = prepare_image_for_augmentation(image, ensure_uint8=True)

            transformed = self.transform(image=image)
            processed_image = transformed['image']

            self.preprocessed_data.append(processed_image)
            self.labels.append(label)

        print(f"Pré-processamento estático concluído!\n")

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        return self.preprocessed_data[idx], self.labels[idx]

class SyntheticAugmentedDataset(Dataset):
    def __init__(
            self,
            original_dataset: Subset,
            synthetic_indices: List[int],
            augmentation_transform: alb.Compose,
            original_transform: alb.Compose = None
    ):
        self.original_dataset = original_dataset
        self.synthetic_indices = synthetic_indices
        self.augmentation_transform = augmentation_transform
        self.original_transform = original_transform

        self.num_synthetic_copies = len(synthetic_indices)

    def __len__(self):
        return len(self.original_dataset) + self.num_synthetic_copies

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]

        synthetic_idx = idx - len(self.original_dataset)
        original_idx = self.synthetic_indices[synthetic_idx]

        base_dataset = self.original_dataset
        while hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset

        if hasattr(base_dataset, 'data') and hasattr(base_dataset, 'targets'):
            image = base_dataset.data[original_idx]
            label = base_dataset.targets[original_idx]
        else:
            image, label = base_dataset[original_idx]

        image = prepare_image_for_augmentation(image, ensure_uint8=True)

        augmented = self.augmentation_transform(image=image)
        image = augmented['image']

        return image, label

def augment_minority_class(
        train_split: Subset,
        target_strategy: str = 'balance',
        target_ratio: float = 0.6,
        architecture_name: str = 'resnext50_32x4d',
        minority_class: int = 0
) -> Subset:
    print(f"\n{'-' * 60}")
    print("AUGMENTAÇÃO DA CLASSE MINORITÁRIA")
    print(f"{'-' * 60}\n")

    if hasattr(train_split.dataset, 'targets'):
        all_labels = np.array(train_split.dataset.targets)
    elif hasattr(train_split.dataset, 'labels'):
        all_labels = np.array(train_split.dataset.labels)
    else:
        all_labels = []
        for i in range(len(train_split)):
            _, label = train_split.dataset[i]
            all_labels.append(label)
        all_labels = np.array(all_labels)

    train_labels = all_labels[train_split.indices]

    class_counts = Counter(train_labels)
    minority_count = class_counts[minority_class]
    majority_class = 1 - minority_class
    majority_count = class_counts[majority_class]

    if target_strategy == 'balance':
        target_minority_count = majority_count
        print(f"\nEstratégia: Balance (igualar classes)\n")
    elif target_strategy == 'ratio':
        target_minority_count = int(majority_count * target_ratio)
        print(f"\nEstratégia: Ratio {target_ratio}:1\n")
    elif target_strategy == 'proportional':
        target_minority_count = int(minority_count * 1.5)
        print(f"\nEstratégia: Proportional (1.5x)\n")
    else:
        raise ValueError(f"\nEstratégia desconhecida: {target_strategy}\n")

    num_synthetic = max(0, target_minority_count - minority_count)

    if num_synthetic == 0:
        print(f"\nNenhuma amostra sintética necessária.\n")
        return train_split

    print(f"\nGerando {num_synthetic} amostras sintéticas...\n")

    minority_indices_in_split = [
        train_split.indices[i]
        for i in range(len(train_split))
        if train_labels[i] == minority_class
    ]

    np.random.seed(42)
    synthetic_source_indices = np.random.choice(
        minority_indices_in_split,
        size=num_synthetic,
        replace=True
    )

    synthetic_transform = create_synthetic_augmentation_for_minority(
        architecture_name=architecture_name
    )

    augmented_dataset = SyntheticAugmentedDataset(
        original_dataset=train_split,
        synthetic_indices=synthetic_source_indices.tolist(),
        augmentation_transform=synthetic_transform,
        original_transform=None
    )

    new_indices = list(range(len(augmented_dataset)))
    augmented_split = Subset(augmented_dataset, new_indices)

    print(f"\nDistribuição Final:")
    print(f"   Classe {minority_class}: {target_minority_count} amostras")
    print(f"   Classe {majority_class}: {majority_count} amostras")
    print(f"   Total: {len(augmented_split)} amostras")
    print(f"   Novo Ratio: 1:{majority_count / target_minority_count:.2f}")

    print(f"\n{'-' * 60}\n")

    return augmented_split