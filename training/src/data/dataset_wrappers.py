import os
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from typing import List, Optional, Dict, Tuple
from collections import Counter, defaultdict

from .preprocessing import prepare_image_for_augmentation
from .augmentation import get_alzheimer_grayscale_augmentation, create_synthetic_augmentation_for_minority
from .subject_manager import extract_subject_id, extract_slice_index, resolve_subset_labels
from src.utils.config import load_augmentation_config

class DynamicAugmentationDataset(Dataset):
    def __init__(self, subset_dataset: Subset, architecture_name: str):
        self.subset_dataset = subset_dataset
        self.architecture_name = architecture_name

        from .subject_manager import count_unique_subjects
        try:
            n_subjects = count_unique_subjects(subset_dataset)
        except:
            n_subjects = len(subset_dataset) # Fallback para fatias se falhar

        num_slices = len(subset_dataset)
        self.transform = get_alzheimer_grayscale_augmentation(
            architecture_name=architecture_name,
            dataset_size=n_subjects,
            is_training=True,
            num_slices=num_slices
        )

    def __len__(self) -> int:
        return len(self.subset_dataset)

    def __getitem__(self, idx: int):
        image, label = self.subset_dataset[idx]
        image = prepare_image_for_augmentation(image)
        transformed = self.transform(image=image)
        
        img_tensor = transformed['image']
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.as_tensor(img_tensor)
            
        return img_tensor, torch.tensor(label, dtype=torch.long)

class StaticPreprocessedDataset(Dataset):
    def __init__(self, subset_dataset: Subset, architecture_name: str):
        self.subset_dataset = subset_dataset
        
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

    def __len__(self) -> int:
        return len(self.preprocessed_data)

    def __getitem__(self, idx: int):
        return self.preprocessed_data[idx], self.labels[idx]

class SyntheticAugmentedDataset(Dataset):
    def __init__(self, sampling_subset: 'SubjectSamplingSubset', augmentation_transform):
        self.sampling_subset = sampling_subset
        self.augmentation_transform = augmentation_transform
        self.dataset = sampling_subset.dataset

        self.total_len = len(self.sampling_subset.indices) + len(self.sampling_subset.synthetic_indices)
        self.indices = list(range(self.total_len))

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int):
        if idx < len(self.sampling_subset.indices):
            return self.sampling_subset[idx]

        synthetic_idx = idx - len(self.sampling_subset.indices)
        original_img_idx = self.sampling_subset.synthetic_indices[synthetic_idx]
        
        image, label = self.dataset[original_img_idx]
        image = prepare_image_for_augmentation(image)
        augmented = self.augmentation_transform(image=image)
        
        return augmented['image'], label

class SubjectSamplingSubset(Subset):
    def __init__(self, dataset, subject_indices, max_slices, strategy='random'):
        self.subject_indices = subject_indices
        self.max_slices = max_slices
        self.strategy = strategy
        self.rng = np.random.default_rng()
        
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
        return [item[1] for item in indexed[start:end]]

def augment_minority_class(
    train_split: SubjectSamplingSubset,
    architecture_name: str,
    target_strategy: str = 'balance',
    minority_classes: List[int] = [],
    target_ratio: float = 0.6
) -> Dataset:
    if not minority_classes:
        return train_split

    print(f"[AUGMENT] Balanceando classes {minority_classes} via estratégia '{target_strategy}'")
    
    train_labels = resolve_subset_labels(train_split)
    class_counts = Counter(train_labels)
    
    majority_count = max(class_counts.values())
    targets = {}
    for cl in minority_classes:
        if target_strategy == 'balance':
            targets[cl] = majority_count
        elif target_strategy == 'ratio':
            targets[cl] = int(majority_count * target_ratio)
        else:
            targets[cl] = class_counts[cl]

    quotas = {}
    subjects_in_split = list(train_split.subject_indices.keys())
    
    for cl, target_count in targets.items():
        num_new = max(0, target_count - class_counts[cl])
        if num_new <= 0: continue
            
        minority_subjects = [s for s in subjects_in_split if train_split.subject_labels.get(s) == cl]
        if not minority_subjects: continue
            
        quota_per_subject = num_new // len(minority_subjects)
        remainder = num_new % len(minority_subjects)
        for i, s in enumerate(minority_subjects):
            quotas[s] = quota_per_subject + (1 if i < remainder else 0)

    train_split.set_synthetic_quotas(quotas)
    
    synthetic_transform = create_synthetic_augmentation_for_minority(architecture_name)
    return SyntheticAugmentedDataset(
        sampling_subset=train_split,
        augmentation_transform=synthetic_transform
    )

def _group_samples_by_subject(dataset) -> dict:
    subjects = defaultdict(lambda: {"indices": [], "label": None})
    for idx, (path, label) in enumerate(dataset.samples):
        subject_id = extract_subject_id(os.path.basename(path))
        subjects[subject_id]["indices"].append(idx)
        subjects[subject_id]["label"] = label
    return subjects

def _print_fold_report(fold_idx, dataset, subjects, train_dataset, val_dataset, train_subject_ids, val_subject_ids):
    train_labels_list = resolve_subset_labels(train_dataset)
    val_labels_list   = resolve_subset_labels(val_dataset)
    train_counts = Counter(train_labels_list)
    val_counts   = Counter(val_labels_list)
    
    print(f"\n  > [FOLD {fold_idx}/5] CONFIGURADO")
    print(f"    - Treino:    {len(train_dataset):>6} fatias | {len(train_subject_ids):>3} sujeitos")
    print(f"    - Validação: {len(val_dataset):>6} fatias | {len(val_subject_ids):>3} sujeitos")
    
    overlap = set(train_subject_ids) & set(val_subject_ids)
    if overlap: 
        print(f"    - [ERRO CRÍTICO] Vazamento de {len(overlap)} sujeitos!")
    else: 
        print(f"    - [OK] Estritamente separado por sujeito")

def create_kfold_splits(
    dataset,
    n_folds: int = 5,
    random_state: int = 42,
    max_slices_per_subject: Optional[int] = None,
    minority_classes: List[int] = None,
    architecture_name: str = None,
    minority_config: Dict = None
) -> List[Tuple]:
    print(f"{'=' * 60}\nCRIANDO {n_folds}-FOLD CV (POR SUJEITO)\n{'=' * 60}")
    
    subjects = _group_samples_by_subject(dataset)
    subject_ids = list(subjects.keys())
    subject_labels = [subjects[sid]["label"] for sid in subject_ids]
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_subj_idx, val_subj_idx) in enumerate(skf.split(subject_ids, subject_labels), 1):
        train_subject_ids = [subject_ids[i] for i in train_subj_idx]
        val_subject_ids   = [subject_ids[i] for i in val_subj_idx]
        
        train_dataset = SubjectSamplingSubset(
            dataset=dataset,
            subject_indices={sid: subjects[sid]["indices"] for sid in train_subject_ids},
            max_slices=max_slices_per_subject,
            random_state=random_state,
            strategy='random'
        )
        
        if minority_config and minority_config.get('enabled'):
            ratio_val = minority_config.get('ratio', 0.6)
            target_ratio = ratio_val if isinstance(ratio_val, (int, float)) else ratio_val.get('default_ratio', 0.6)

            train_dataset = augment_minority_class(
                train_split=train_dataset,
                architecture_name=architecture_name,
                target_strategy=minority_config.get('strategy', 'ratio'),
                minority_classes=minority_classes or [],
                target_ratio=target_ratio
            )
            
        val_dataset = SubjectSamplingSubset(
            dataset=dataset,
            subject_indices={sid: subjects[sid]["indices"] for sid in val_subject_ids},
            max_slices=max_slices_per_subject,
            random_state=random_state,
            strategy='middle'
        )
        
        _print_fold_report(fold_idx, dataset, subjects, train_dataset, val_dataset, train_subject_ids, val_subject_ids)
        folds.append((train_dataset, val_dataset))

    print(f"\n{'-' * 60}")
    print(f"K-FOLD CONCLUÍDO: {n_folds} Folds prontos para execução")
    print(f"{'-' * 60}\n")
    return folds