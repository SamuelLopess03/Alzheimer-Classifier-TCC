import os
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from typing import List, Optional, Dict
from collections import Counter, defaultdict

from .preprocessing import prepare_image_for_augmentation
from .augmentation import get_alzheimer_grayscale_augmentation, create_synthetic_augmentation_for_minority
from .subject_manager import extract_subject_id, extract_slice_index, count_unique_subjects, resolve_subset_labels
from src.utils.config import load_augmentation_config

class DynamicAugmentationDataset(Dataset):
    def __init__(self, subset_dataset: Subset, architecture_name: str):
        self.subset_dataset = subset_dataset
        self.architecture_name = architecture_name

        try:
            size_metric = count_unique_subjects(subset_dataset)
        except Exception:
            size_metric = None

        if not size_metric:
            size_metric = len(subset_dataset)

        num_slices = len(subset_dataset)
        self.transform = get_alzheimer_grayscale_augmentation(
            architecture_name=architecture_name,
            dataset_size=size_metric,
            is_training=True,
            num_slices=num_slices
        )
        
        self.synthetic_transform = create_synthetic_augmentation_for_minority(
            architecture_name=architecture_name
        )

    def __len__(self) -> int:
        return len(self.subset_dataset)

    def _is_idx_synthetic(self, idx: int) -> bool:
        base_dataset = self.subset_dataset

        while hasattr(base_dataset, 'dataset'):
            if hasattr(base_dataset, 'is_synthetic'):
                if base_dataset.is_synthetic(idx):
                    return True
            
            if hasattr(base_dataset, 'indices'):
                idx = base_dataset.indices[idx]
            base_dataset = base_dataset.dataset

        if hasattr(base_dataset, 'is_synthetic'):
            return base_dataset.is_synthetic(idx)

        return False

    def __getitem__(self, idx: int):
        is_synthetic = self._is_idx_synthetic(idx)
        image, label = self.subset_dataset[idx]

        image = prepare_image_for_augmentation(image)

        if is_synthetic:
            # Amostras sintéticas recebem o pipeline mais agressivo
            transformed = self.synthetic_transform(image=image)
        else:
            # Amostras reais recebem o pipeline de treino (dinâmico por tamanho)
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

        transform = get_alzheimer_grayscale_augmentation(
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
            transformed = transform(image=image)
            self.preprocessed_data.append(transformed['image'])
            self.labels.append(label)

        print("Pré-processamento estático concluído!\n")

    def __len__(self) -> int:
        return len(self.preprocessed_data)

    def __getitem__(self, idx: int):
        return self.preprocessed_data[idx], self.labels[idx]

class SyntheticAugmentedDataset(Dataset):
    def __init__(
            self,
            sampling_subset: 'SubjectSamplingSubset',
            augmentation_transform
    ):
        self.sampling_subset = sampling_subset
        self.augmentation_transform = augmentation_transform

    def __len__(self) -> int:
        return len(self.sampling_subset.indices) + len(self.sampling_subset.synthetic_indices)

    def is_synthetic(self, idx: int) -> bool:
        return idx >= len(self.sampling_subset.indices)

    def __getitem__(self, idx: int):
        if idx < len(self.sampling_subset.indices):
            return self.sampling_subset[idx]

        synthetic_idx = idx - len(self.sampling_subset.indices)
        original_img_idx = self.sampling_subset.synthetic_indices[synthetic_idx]
        
        base_dataset = self.sampling_subset.dataset
        image, label = base_dataset[original_img_idx]
        
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
    strategies_cfg = min_cfg.get("strategies", {})

    minority_classes = minority_classes or [0]
    target_strategy = target_strategy or list(strategies_cfg.keys())[1] if strategies_cfg else 'ratio'
    
    if target_ratio is None:
        ratio_cfg = strategies_cfg.get('ratio', {})
        target_ratio = ratio_cfg.get('default_ratio', 0.6) if isinstance(ratio_cfg, dict) else 0.6

    print(f"\n{'-' * 60}\nBALANCER: Iniciando aumentação ({target_strategy})\n{'-' * 60}")

    train_labels = resolve_subset_labels(train_split)
    class_counts = Counter(train_labels)

    targets = _calculate_target_counts(
        class_counts, minority_classes, target_strategy, min_cfg,
        target_ratio, custom_targets, target_percentage
    )
    
    synthetic_quotas = _calculate_subject_quotas(
        train_split, train_labels, targets, class_counts
    )

    if not any(q > 0 for q in synthetic_quotas.values()):
        print("\nNenhum balanceamento necessário.\n")
        return train_split

    train_split.set_synthetic_quotas(synthetic_quotas)
    
    synthetic_transform = create_synthetic_augmentation_for_minority(architecture_name)
    
    augmented_ds = SyntheticAugmentedDataset(
        sampling_subset=train_split,
        augmentation_transform=synthetic_transform
    )

    final_split = Subset(augmented_ds, list(range(len(augmented_ds))))
    _print_augmentation_summary(class_counts, targets, len(final_split))

    return final_split

def _calculate_target_counts(
    class_counts, minority_classes, strategy, config, ratio, custom, percentage
) -> Dict[int, int]:
    majority_classes = set(class_counts.keys()) - set(minority_classes)
    majority_count = max(class_counts[c] for c in majority_classes) if majority_classes else max(class_counts.values())

    targets = {}
    for cl in minority_classes:
        count = class_counts[cl]
        if strategy == 'balance':
            targets[cl] = majority_count
        elif strategy == 'ratio':
            targets[cl] = int(majority_count * (ratio or 1.0))
        elif strategy == 'proportional':
            prop_cfg = config['strategies'].get('proportional', {})
            multiplier = prop_cfg.get('multiplier', 1.5) if isinstance(prop_cfg, dict) else 1.5
            targets[cl] = int(count * multiplier)
        elif strategy == 'custom':
            targets[cl] = custom[cl]
        elif strategy == 'percentage':
            p = percentage[cl]
            targets[cl] = int((p * sum(class_counts.values())) / (1 - p))

    return targets

def _calculate_subject_quotas(
    split: 'SubjectSamplingSubset', labels, targets, class_counts
) -> Dict[str, int]:
    quotas = {}
    
    subjects_in_split = list(split.subject_indices.keys())
    
    for cl, target_count in targets.items():
        num_new = max(0, target_count - class_counts[cl])
        if num_new <= 0:
            continue
            
        print(f"   Classe {cl}: distribuindo {num_new} cotas sintéticas dinâmicas")
        
        minority_subjects = [s for s in subjects_in_split if split.subject_labels.get(s) == cl]
        
        if not minority_subjects:
            continue
            
        quota_per_subject = num_new // len(minority_subjects)
        remainder = num_new % len(minority_subjects)
        
        for i, s in enumerate(minority_subjects):
            quotas[s] = quota_per_subject + (1 if i < remainder else 0)

    return quotas

def _print_augmentation_summary(class_counts, targets, total):
    print(f"\nDistribuição Final ({total} amostras):")
    for cl in sorted(class_counts.keys()):
        count = targets.get(cl, class_counts[cl])
        print(f"   Classe {cl}: {count} ({100*count/total:.1f}%)")
    print(f"{'-' * 60}\n")

class SubjectSamplingSubset(Subset):
    def __init__(
        self,
        dataset,
        subject_indices: Dict[str, List[int]],
        max_slices: Optional[int],
        random_state: int = 42,
        strategy: str = 'random'
    ):
        self.subject_indices = subject_indices
        self.max_slices = max_slices
        self.strategy = strategy
        self.rng = np.random.default_rng(random_state)
        
        self.subject_labels = {sid: dataset.samples[indices[0]][1] for sid, indices in subject_indices.items()}
        self.synthetic_quotas = {}
        self.synthetic_indices = []
        
        super().__init__(dataset, [])
        self.resample()

    def set_synthetic_quotas(self, quotas: Dict[str, int]):
        self.synthetic_quotas = quotas
        
        self.resample()

    def resample(self):
        new_indices = []
        new_synthetic = []

        for sid, indices in self.subject_indices.items():
            if self.max_slices is not None and len(indices) > self.max_slices:
                if self.strategy == 'random':
                    sampled = self.rng.choice(indices, size=self.max_slices, replace=False)
                    new_indices.extend(sampled.tolist())
                elif self.strategy == 'middle':
                    sampled = self._select_middle_slices(indices, self.max_slices)
                    new_indices.extend(sampled)
            else:
                new_indices.extend(indices)
            
            quota = self.synthetic_quotas.get(sid, 0)
            if quota > 0:
                synthetic_sampled = self.rng.choice(indices, size=quota, replace=True)
                new_synthetic.extend(synthetic_sampled.tolist())
                
        self.indices = sorted(new_indices)
        self.synthetic_indices = new_synthetic

    def _select_middle_slices(self, indices: List[int], max_n: int) -> List[int]:
        if max_n is None or len(indices) <= max_n:
            return indices

        indexed = []
        for idx in indices:
            path, _ = self.dataset.samples[idx]
            slice_idx = extract_slice_index(os.path.basename(path))
            indexed.append((slice_idx, idx))

        indexed.sort()
        mid = len(indexed) // 2
        half = max_n // 2
        start = max(0, mid - half)
        end = min(len(indexed), start + max_n)
        start = max(0, end - max_n)

        return [item[1] for item in indexed[start:end]]

def _group_samples_by_subject(dataset) -> dict:
    subjects = defaultdict(lambda: {"indices": [], "label": None})
    for idx, (path, label) in enumerate(dataset.samples):
        subject_id = extract_subject_id(os.path.basename(path))
        subjects[subject_id]["indices"].append(idx)
        subjects[subject_id]["label"] = label
    return subjects

def _print_holdout_report(dataset, subjects, train_dataset, val_dataset, train_subject_ids, val_subject_ids):
    train_labels_list = [dataset.samples[i][1] for i in train_dataset.indices]
    val_labels_list   = [dataset.samples[i][1] for i in val_dataset.indices]
    train_counts = Counter(train_labels_list)
    val_counts   = Counter(val_labels_list)
    train_subj_per_class = Counter([subjects[s]["label"] for s in train_subject_ids])
    val_subj_per_class   = Counter([subjects[s]["label"] for s in val_subject_ids])

    print("Distribuição por Classe:")
    print("\n  TREINO:")
    total_train = sum(train_counts.values())
    for cls in sorted(train_counts.keys()):
        count = train_counts[cls]
        print(f"    Classe {cls}: {count:>5} fatias, {train_subj_per_class.get(cls, 0):>3} sujeitos ({count/total_train*100:>5.1f}%)")
    print(f"    Total:     {total_train:>5} fatias, {len(train_subject_ids):>3} sujeitos")

    print("\n  VALIDAÇÃO:")
    total_val = sum(val_counts.values())
    for cls in sorted(val_counts.keys()):
        count = val_counts[cls]
        print(f"    Classe {cls}: {count:>5} fatias, {val_subj_per_class.get(cls, 0):>3} sujeitos ({count/total_val*100:>5.1f}%)")
    print(f"    Total:     {total_val:>5} fatias, {len(val_subject_ids):>3} sujeitos")

    overlap = set(train_subject_ids) & set(val_subject_ids)
    if overlap:
        print(f"\n  ERRO CRÍTICO: {len(overlap)} sujeitos em ambos os splits!")
    else:
        print("\n  Zero vazamento entre treino e validação")

def create_stratified_holdout_split(
        dataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.3,
        random_state: int = 42,
        max_slices_per_subject: Optional[int] = None
) -> tuple:
    print(f"{'-' * 60}")
    print("CRIANDO HOLDOUT SPLIT POR SUJEITO (DINÂMICO)")
    print(f"{'-' * 60}\n")
    print(f"Configuração:")
    print(f"  Train Ratio: {train_ratio:.1%}")
    print(f"  Val Ratio:   {val_ratio:.1%}")
    print(f"  Random State: {random_state}")
    if max_slices_per_subject:
        print(f"  Max Slices/Subject: {max_slices_per_subject}\n")

    total_ratio = train_ratio + val_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio devem somar 1.0, mas somam {total_ratio:.4f}")

    subjects = _group_samples_by_subject(dataset)
    subject_ids = list(subjects.keys())
    subject_labels = [subjects[s]["label"] for s in subject_ids]
    print(f"Dataset total: {len(dataset)} fatias de {len(subject_ids)} sujeitos\n")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_subj_idx, val_subj_idx = next(splitter.split(subject_ids, subject_labels))
    train_subject_ids = [subject_ids[i] for i in train_subj_idx]
    val_subject_ids   = [subject_ids[i] for i in val_subj_idx]

    train_dataset = SubjectSamplingSubset(
        dataset=dataset,
        subject_indices={sid: subjects[sid]["indices"] for sid in train_subject_ids},
        max_slices=max_slices_per_subject,
        random_state=random_state,
        strategy='random'
    )
    val_dataset = SubjectSamplingSubset(
        dataset=dataset,
        subject_indices={sid: subjects[sid]["indices"] for sid in val_subject_ids},
        max_slices=max_slices_per_subject,
        random_state=random_state,
        strategy='middle'
    )

    _print_holdout_report(dataset, subjects, train_dataset, val_dataset, train_subject_ids, val_subject_ids)
    print(f"\n{'-' * 60}\n")
    return train_dataset, val_dataset