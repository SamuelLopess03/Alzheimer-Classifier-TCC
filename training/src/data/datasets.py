import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from typing import List, Optional, Dict
from collections import Counter

from .preprocessing import prepare_image_for_augmentation
from .utils import count_unique_subjects, resolve_subset_labels
from .augmentation import get_alzheimer_grayscale_augmentation, create_synthetic_augmentation_for_minority
from ..utils import load_augmentation_config

class DynamicAugmentationDataset(Dataset):
    def __init__(self, subset_dataset: Subset, architecture_name: str):
        self.subset_dataset = subset_dataset
        self.architecture_name = architecture_name

        try:
            size_metric = count_unique_subjects(subset_dataset)
        except:
            size_metric = None

        if not size_metric:
            size_metric = len(subset_dataset)

        self.transform = get_alzheimer_grayscale_augmentation(
            architecture_name=architecture_name,
            dataset_size=size_metric,
            is_training=True
        )
        
        self.synthetic_transform = create_synthetic_augmentation_for_minority(
            architecture_name=architecture_name
        )

    def __len__(self):
        return len(self.subset_dataset)

    def _is_idx_synthetic(self, idx: int) -> bool:
        base_dataset = self.subset_dataset
        while hasattr(base_dataset, 'dataset'):
            if hasattr(base_dataset, 'indices'):
                idx = base_dataset.indices[idx]
            base_dataset = base_dataset.dataset
        if hasattr(base_dataset, 'is_synthetic'):
            return base_dataset.is_synthetic(idx)
        return False

    def __getitem__(self, idx):
        is_synthetic = self._is_idx_synthetic(idx)
        
        if is_synthetic:
            image, label = self.subset_dataset[idx]
        else:
            image, label = self.subset_dataset[idx]
            image = prepare_image_for_augmentation(image)
            transformed = self.transform(image=image)
            image = transformed['image']

        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

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
            self.preprocessed_data.append(transformed['image'])
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
            augmentation_transform: torch.nn.Module,
            original_transform: torch.nn.Module = None
    ):
        self.original_dataset = original_dataset
        self.synthetic_indices = synthetic_indices
        self.augmentation_transform = augmentation_transform
        self.original_transform = original_transform
        self.num_synthetic_copies = len(synthetic_indices)

    def __len__(self):
        return len(self.original_dataset) + self.num_synthetic_copies

    def is_synthetic(self, idx: int) -> bool:
        return idx >= len(self.original_dataset)

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
        return augmented['image'], label

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
    min_cfg = aug_config['minority_augmentation']
    minority_classes = minority_classes or [0]
    target_strategy = target_strategy or list(min_cfg["strategies"].keys())[1]
    target_ratio = target_ratio or min_cfg['strategies']['ratio']['default_ratio']

    print(f"\n{'-' * 60}\nBALANCER: Iniciando aumentação ({target_strategy})\n{'-' * 60}")

    train_labels = resolve_subset_labels(train_split)
    class_counts = Counter(train_labels)

    targets = _calculate_target_counts(
        class_counts, minority_classes, target_strategy, min_cfg, 
        target_ratio, custom_targets, target_percentage
    )
    
    synthetic_indices = _sample_synthetic_indices(
        train_split, train_labels, targets, class_counts, min_cfg
    )

    if not synthetic_indices:
        print("\nNenhum balanceamento necessário.\n")
        return train_split

    synthetic_transform = create_synthetic_augmentation_for_minority(architecture_name)
    
    augmented_ds = SyntheticAugmentedDataset(
        original_dataset=train_split,
        synthetic_indices=synthetic_indices,
        augmentation_transform=synthetic_transform
    )

    final_split = Subset(augmented_ds, list(range(len(augmented_ds))))
    _print_augmentation_summary(class_counts, targets, len(final_split))

    return final_split

def _calculate_target_counts(class_counts, minority_classes, strategy, config, ratio, custom, percentage):
    all_classes = set(class_counts.keys())
    majority_classes = all_classes - set(minority_classes)
    majority_count = max([class_counts[c] for c in majority_classes]) if majority_classes else max(class_counts.values())
    
    targets = {}
    for cl in minority_classes:
        count = class_counts[cl]
        if strategy == 'balance': targets[cl] = majority_count
        elif strategy == 'ratio': targets[cl] = int(majority_count * (ratio or 1.0))
        elif strategy == 'proportional': targets[cl] = int(count * config['strategies']['proportional']['multiplier'])
        elif strategy == 'custom': targets[cl] = custom[cl]
        elif strategy == 'percentage':
            p = percentage[cl]
            targets[cl] = int((p * sum(class_counts.values())) / (1 - p))

    return targets

def _sample_synthetic_indices(split, labels, targets, class_counts, config):
    synthetic_indices = []
    base_seed = config['random_seed']['base']
    
    for cl, target_count in targets.items():
        num_new = max(0, target_count - class_counts[cl])
        if num_new > 0:
            print(f"   Classe {cl}: gerando {num_new} amostras sintéticas")
            indices_in_split = [split.indices[i] for i, l in enumerate(labels) if l == cl]
            seed = base_seed + cl if config['random_seed']['per_class_offset'] else base_seed
            rng = np.random.default_rng(seed)
            sampled = rng.choice(indices_in_split, size=num_new, replace=True)
            synthetic_indices.extend(sampled.tolist())

    return synthetic_indices

def _print_augmentation_summary(class_counts, targets, total):
    print(f"\nDistribuição Final ({total} amostras):")
    for cl in sorted(class_counts.keys()):
        count = targets.get(cl, class_counts[cl])
        print(f"   Classe {cl}: {count} ({100*count/total:.1f}%)")
    print(f"{'-' * 60}\n")
