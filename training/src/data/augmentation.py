import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, Subset
from typing import List, Optional, Dict
from collections import Counter

from .preprocessing import MedicalImagePreprocessor, prepare_image_for_augmentation
from ..utils import load_augmentation_config

def get_alzheimer_grayscale_augmentation(
        architecture_name: str,
        dataset_size: int,
        is_training: bool = True
) -> alb.Compose:
    preprocessor = MedicalImagePreprocessor(architecture_name)
    config = preprocessor.config
    aug_config = load_augmentation_config()

    is_transformer = architecture_name.lower() in ['vit_b_16', 'swin_v2_tiny']

    if is_training:
        thresholds = aug_config['dataset_size_thresholds']

        if dataset_size < thresholds['small']:
            print(
                f"\nAugmentação Pesada: Dataset pequeno (<{thresholds['small']}) - {'Transformer' if is_transformer else 'CNN'}\n")

            if is_transformer:
                cfg = aug_config['train_heavy_transformer']

                tile_grid_size = (int(cfg['clahe']['tile_grid_size'][0]),
                                  int(cfg['clahe']['tile_grid_size'][1]))

                augmentations = [
                    alb.Resize(config["image_size"], config["image_size"]),
                    alb.HorizontalFlip(p=cfg['horizontal_flip']['probability']),
                    alb.Rotate(limit=cfg['rotation']['limit'], p=cfg['rotation']['probability']),
                    alb.CLAHE(
                        clip_limit=cfg['clahe']['clip_limit'],
                        tile_grid_size=tile_grid_size,
                        p=cfg['clahe']['probability']
                    ),
                    alb.RandomBrightnessContrast(
                        brightness_limit=cfg['brightness_contrast']['brightness_limit'],
                        contrast_limit=cfg['brightness_contrast']['contrast_limit'],
                        p=cfg['brightness_contrast']['probability']
                    ),
                    alb.Normalize(mean=config["mean"], std=config["std"]),
                    ToTensorV2()
                ]
            else:
                cfg = aug_config['train_heavy_cnn']

                # Tipagens Necessárias para Evitar os Avisos do Type Checker
                scale = (float(cfg['affine']['scale'][0]), float(cfg['affine']['scale'][1]))
                translate_percent = (float(cfg['affine']['translate_percent'][0]), float(cfg['affine']['translate_percent'][1]))
                rotate = (float(cfg['affine']['rotate'][0]), float(cfg['affine']['rotate'][1]))
                tile_grid_size = (int(cfg['clahe']['tile_grid_size'][0]), int(cfg['clahe']['tile_grid_size'][1]))
                gamma_limit = (float(cfg['random_gamma']['gamma_limit'][0]), float(cfg['random_gamma']['gamma_limit'][1]))
                blur_limit = (int(cfg['blur_sharpen']['gaussian_blur']['blur_limit'][0]),
                              int(cfg['blur_sharpen']['gaussian_blur']['blur_limit'][1]))
                sharpen_alpha = (float(cfg['blur_sharpen']['sharpen']['alpha'][0]), float(cfg['blur_sharpen']['sharpen']['alpha'][1]))
                sharpen_lightness = (float(cfg['blur_sharpen']['sharpen']['lightness'][0]),
                                     float(cfg['blur_sharpen']['sharpen']['lightness'][1]))

                augmentations = [
                    alb.Resize(config["image_size"], config["image_size"]),
                    alb.HorizontalFlip(p=cfg['horizontal_flip']['probability']),
                    alb.Rotate(limit=cfg['rotation']['limit'], p=cfg['rotation']['probability']),
                    alb.Affine(
                        scale=scale,
                        translate_percent=translate_percent,
                        rotate=rotate,
                        p=cfg['affine']['probability']
                    ),
                    alb.CLAHE(
                        clip_limit=cfg['clahe']['clip_limit'],
                        tile_grid_size=tile_grid_size,
                        p=cfg['clahe']['probability']
                    ),
                    alb.RandomBrightnessContrast(
                        brightness_limit=cfg['brightness_contrast']['brightness_limit'],
                        contrast_limit=cfg['brightness_contrast']['contrast_limit'],
                        p=cfg['brightness_contrast']['probability']
                    ),
                    alb.RandomGamma(
                        gamma_limit=gamma_limit,
                        p=cfg['random_gamma']['probability']
                    ),
                    alb.OneOf([
                        alb.GaussianBlur(blur_limit=blur_limit, p=1.0),
                        alb.Sharpen(
                            alpha=sharpen_alpha,
                            lightness=sharpen_lightness,
                            p=1.0
                        ),
                    ], p=cfg['blur_sharpen']['probability']),
                    alb.Normalize(mean=config["mean"], std=config["std"]),
                    ToTensorV2()
                ]

        elif dataset_size < thresholds['medium']:
            print(f"\nAugmentação Moderada: Dataset médio (<{thresholds['medium']})\n")

            cfg = aug_config['train_moderate']

            tile_grid_size = (int(cfg['clahe']['tile_grid_size'][0]),
                              int(cfg['clahe']['tile_grid_size'][1]))
            gamma_limit = (float(cfg['random_gamma']['gamma_limit'][0]), float(cfg['random_gamma']['gamma_limit'][1]))
            blur_limit = (int(cfg['gaussian_blur']['blur_limit'][0]),
                          int(cfg['gaussian_blur']['blur_limit'][1]))

            augmentations = [
                alb.Resize(config["image_size"], config["image_size"]),
                alb.HorizontalFlip(p=cfg['horizontal_flip']['probability']),
                alb.Rotate(limit=cfg['rotation']['limit'], p=cfg['rotation']['probability']),
                alb.CLAHE(
                    clip_limit=cfg['clahe']['clip_limit'],
                    tile_grid_size=tile_grid_size,
                    p=cfg['clahe']['probability']
                ),
                alb.RandomBrightnessContrast(
                    brightness_limit=cfg['brightness_contrast']['brightness_limit'],
                    contrast_limit=cfg['brightness_contrast']['contrast_limit'],
                    p=cfg['brightness_contrast']['probability']
                ),
                alb.RandomGamma(
                    gamma_limit=gamma_limit,
                    p=cfg['random_gamma']['probability']
                ),
                alb.GaussianBlur(
                    blur_limit=blur_limit,
                    p=cfg['gaussian_blur']['probability']
                ),
                alb.Normalize(mean=config["mean"], std=config["std"]),
                ToTensorV2()
            ]

        else:
            print(f"\nAugmentação Leve: Dataset grande (≥{thresholds['large']})\n")

            cfg = aug_config['train_light']

            tile_grid_size = (int(cfg['clahe']['tile_grid_size'][0]),
                              int(cfg['clahe']['tile_grid_size'][1]))

            augmentations = [
                alb.Resize(config["image_size"], config["image_size"]),
                alb.HorizontalFlip(p=cfg['horizontal_flip']['probability']),
                alb.Rotate(limit=cfg['rotation']['limit'], p=cfg['rotation']['probability']),
                alb.CLAHE(
                    clip_limit=cfg['clahe']['clip_limit'],
                    tile_grid_size=tile_grid_size,
                    p=cfg['clahe']['probability']
                ),
                alb.RandomBrightnessContrast(
                    brightness_limit=cfg['brightness_contrast']['brightness_limit'],
                    contrast_limit=cfg['brightness_contrast']['contrast_limit'],
                    p=cfg['brightness_contrast']['probability']
                ),
                alb.Normalize(mean=config["mean"], std=config["std"]),
                ToTensorV2()
            ]

    else:
        augmentations = [
            alb.Resize(config["image_size"], config["image_size"]),
            alb.Normalize(mean=config["mean"], std=config["std"]),
            ToTensorV2()
        ]

    return alb.Compose(augmentations)

def create_synthetic_augmentation_for_minority(
        architecture_name: str
) -> alb.Compose:
    aug_config = load_augmentation_config()
    is_transformer = architecture_name.lower() in ['vit_b_16', 'swin_v2_tiny']

    if is_transformer:
        cfg = aug_config['synthetic_transformer']

        # Tipagens Necessárias para Evitar os Avisos do Type Checker
        translate_percent_x = (float(cfg['affine']['translate_percent']['x'][0]),
                               float(cfg['affine']['translate_percent']['x'][1]))
        translate_percent_y = (float(cfg['affine']['translate_percent']['y'][0]),
                               float(cfg['affine']['translate_percent']['y'][1]))
        scale = (float(cfg['affine']['scale'][0]), float(cfg['affine']['scale'][1]))
        rotate = (float(cfg['affine']['rotate'][0]), float(cfg['affine']['rotate'][1]))
        std_range = (float(cfg['gauss_noise']['std_range'][0]), float(cfg['gauss_noise']['std_range'][1]))
        mean_range = (float(cfg['gauss_noise']['mean_range'][0]), float(cfg['gauss_noise']['mean_range'][1]))

        augmentations = [
            alb.VerticalFlip(p=cfg['vertical_flip']['probability']),
            alb.Affine(
                translate_percent={
                    "x": translate_percent_x,
                    "y": translate_percent_y
                },
                scale=scale,
                rotate=rotate,
                p=cfg['affine']['probability']
            ),
            # Transformações de intensidade para grayscale
            alb.Posterize(
                num_bits=6,
                p=0.2
            ),
            # Distorções geométricas
            alb.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.3
            ),
            # Ruído
            alb.GaussNoise(
                std_range=std_range,
                mean_range=mean_range,
                p=cfg['gauss_noise']['probability']
            ),
            alb.MultiplicativeNoise(
                multiplier=(0.9, 1.1),
                per_channel=False,
                p=0.2
            ),
        ]
    else:
        cfg = aug_config['synthetic_cnn']

        # Tipagens Necessárias para Evitar os Avisos do Type Checker
        translate_percent_x = (float(cfg['affine']['translate_percent']['x'][0]),
                               float(cfg['affine']['translate_percent']['x'][1]))
        translate_percent_y = (float(cfg['affine']['translate_percent']['y'][0]),
                               float(cfg['affine']['translate_percent']['y'][1]))
        scale = (float(cfg['affine']['scale'][0]), float(cfg['affine']['scale'][1]))
        rotate = (float(cfg['affine']['rotate'][0]), float(cfg['affine']['rotate'][1]))
        std_range = (float(cfg['gauss_noise']['std_range'][0]), float(cfg['gauss_noise']['std_range'][1]))
        mean_range = (float(cfg['gauss_noise']['mean_range'][0]), float(cfg['gauss_noise']['mean_range'][1]))

        augmentations = [
            alb.VerticalFlip(p=cfg['vertical_flip']['probability']),
            alb.Affine(
                translate_percent={
                    "x": translate_percent_x,
                    "y": translate_percent_y
                },
                scale=scale,
                rotate=rotate,
                p=cfg['affine']['probability']
            ),
            # Transformações geométricas avançadas
            alb.ElasticTransform(
                alpha=cfg['elastic_transform']['alpha'],
                sigma=cfg['elastic_transform']['sigma'],
                p=cfg['elastic_transform']['probability']
            ),
            alb.Perspective(
                scale=(0.05, 0.1),
                p=0.2
            ),
            # Equalização de histograma
            alb.Equalize(p=cfg['equalize']['probability']),
            # Efeitos de textura e bordas
            alb.Emboss(
                alpha=(0.2, 0.5),
                strength=(0.2, 0.7),
                p=0.2
            ),
            # Blur especializado
            alb.MedianBlur(
                blur_limit=5,
                p=0.2
            ),
            alb.MotionBlur(
                blur_limit=7,
                p=0.15
            ),
            # Ruídos
            alb.GaussNoise(
                std_range=std_range,
                mean_range=mean_range,
                p=cfg['gauss_noise']['probability']
            ),
            alb.MultiplicativeNoise(
                multiplier=(0.9, 1.1),
                per_channel=False,
                p=0.2
            )
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

        image = prepare_image_for_augmentation(image)

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
            image = prepare_image_for_augmentation(image)

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

        image = prepare_image_for_augmentation(image)

        augmented = self.augmentation_transform(image=image)
        image = augmented['image']

        return image, label

def augment_minority_class(
        train_split: Subset,
        target_strategy: Optional[str] = None,
        target_ratio: Optional[float] = None,
        architecture_name: str = 'resnext50_32x4d',
        minority_classes: Optional[List[int]] = None,
        custom_targets: Optional[Dict[int, int]] = None,
        target_percentage: Optional[Dict[int, float]] = None
) -> Subset:
    aug_config = load_augmentation_config()
    minority_config = aug_config['minority_augmentation']

    if minority_classes is None:
        minority_classes = [0]

    if target_strategy is None:
        target_strategy = list(minority_config["strategies"].keys())[1]

    if target_ratio is None and target_strategy == 'ratio':
        target_ratio = minority_config['strategies']['ratio']['default_ratio']

    print(f"\n{'-' * 60}")
    print("AUGMENTAÇÃO DAS CLASSES MINORITÁRIAS")
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

    all_classes = set(class_counts.keys())
    majority_classes = all_classes - set(minority_classes)

    if len(majority_classes) > 0:
        majority_count = max([class_counts[c] for c in majority_classes])
    else:
        majority_count = max(class_counts.values())

    targets_per_class = {}

    for minority_class in minority_classes:
        minority_count = class_counts[minority_class]

        if target_strategy == 'balance':
            target_minority_count = majority_count

        elif target_strategy == 'ratio':
            target_minority_count = int(majority_count * target_ratio)

        elif target_strategy == 'proportional':
            multiplier = minority_config['strategies']['proportional']['multiplier']
            target_minority_count = int(minority_count * multiplier)

        elif target_strategy == 'custom':
            if custom_targets is None or minority_class not in custom_targets:
                raise ValueError(
                    f"Estratégia 'custom' requer dicionário custom_targets "
                    f"com target para classe {minority_class}"
                )
            target_minority_count = custom_targets[minority_class]

        elif target_strategy == 'percentage':
            if target_percentage is None or minority_class not in target_percentage:
                raise ValueError(
                    f"Estratégia 'percentage' requer dicionário target_percentage "
                    f"com percentual para classe {minority_class}"
                )
            percentage = target_percentage[minority_class]
            if not 0 < percentage <= 1:
                raise ValueError(f"Percentual deve estar entre 0 e 1, recebido: {percentage}")

            current_total = sum(class_counts.values())
            target_minority_count = int(
                (percentage * current_total) / (1 - percentage)
            )

        else:
            raise ValueError(f"\nEstratégia desconhecida: {target_strategy}\n")

        targets_per_class[minority_class] = target_minority_count

    print(f"\nEstratégia: {target_strategy}")
    if target_strategy == 'ratio':
        print(f"Ratio: {target_ratio}:1")
    elif target_strategy == 'custom':
        print(f"Targets customizados: {custom_targets}")
    elif target_strategy == 'percentage':
        print(f"Percentuais alvo: {target_percentage}")
    print()

    all_synthetic_indices = []
    base_seed = minority_config['random_seed']['base']
    use_offset = minority_config['random_seed']['per_class_offset']

    for minority_class in minority_classes:
        minority_count = class_counts[minority_class]
        target_minority_count = targets_per_class[minority_class]
        num_synthetic = max(0, target_minority_count - minority_count)

        if num_synthetic > 0:
            print(f"Classe {minority_class}: gerando {num_synthetic} amostras sintéticas")

            minority_indices_in_split = [
                train_split.indices[i]
                for i in range(len(train_split))
                if train_labels[i] == minority_class
            ]

            seed = base_seed + minority_class if use_offset else base_seed
            np.random.seed(seed)
            synthetic_source_indices = np.random.choice(
                minority_indices_in_split,
                size=num_synthetic,
                replace=True
            )

            all_synthetic_indices.extend(synthetic_source_indices.tolist())

    if len(all_synthetic_indices) == 0:
        print(f"\nNenhuma amostra sintética necessária.\n")
        return train_split

    print(f"\nTotal de amostras sintéticas: {len(all_synthetic_indices)}\n")

    synthetic_transform = create_synthetic_augmentation_for_minority(
        architecture_name=architecture_name
    )

    augmented_dataset = SyntheticAugmentedDataset(
        original_dataset=train_split,
        synthetic_indices=all_synthetic_indices,
        augmentation_transform=synthetic_transform,
        original_transform=None
    )

    new_indices = list(range(len(augmented_dataset)))
    augmented_split = Subset(augmented_dataset, new_indices)

    total_final = len(augmented_split)
    print(f"Distribuição Final:")
    for class_idx in sorted(class_counts.keys()):
        if class_idx in minority_classes:
            final_count = targets_per_class[class_idx]
        else:
            final_count = class_counts[class_idx]
        percentage = (final_count / total_final) * 100
        print(f"   Classe {class_idx}: {final_count} amostras ({percentage:.1f}%)")
    print(f"   Total: {total_final} amostras\n")

    print(f"{'-' * 60}\n")

    return augmented_split