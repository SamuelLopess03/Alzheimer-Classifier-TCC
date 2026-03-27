import os
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter, defaultdict
from typing import Tuple, Optional, List, Dict

from .dataset import extract_subject_id, extract_slice_index

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
        super().__init__(dataset, [])
        self.resample()

    def resample(self):
        new_indices = []
        
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
        
        self.indices = sorted(new_indices)

    def _select_middle_slices(self, indices, max_n):
        if max_n is None or len(indices) <= max_n:
            return indices
            
        indexed_indices = []
        for idx in indices:
            path, _ = self.dataset.samples[idx]
            filename = os.path.basename(path)
            slice_idx = extract_slice_index(filename)
            indexed_indices.append((slice_idx, idx))
            
        indexed_indices.sort()
        mid = len(indexed_indices) // 2
        half = max_n // 2
        start = max(0, mid - half)
        end = start + max_n
        if end > len(indexed_indices):
            end = len(indexed_indices)
            start = max(0, end - max_n)

        return [item[1] for item in indexed_indices[start:end]]

def create_stratified_holdout_split(
        dataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.3,
        random_state: int = 42,
        max_slices_per_subject: Optional[int] = None
) -> Tuple[Subset, Subset]:
    print(f"{'-' * 60}")
    print(f"CRIANDO HOLDOUT SPLIT POR SUJEITO (DINÂMICO)")
    print(f"{'-' * 60}\n")

    print(f"Configuração:")
    print(f"  Train Ratio: {train_ratio:.1%}")
    print(f"  Val Ratio: {val_ratio:.1%}")
    print(f"  Random State: {random_state}")
    if max_slices_per_subject:
        print(f"  Max Slices/Subject: {max_slices_per_subject}\n")

    total_ratio = train_ratio + val_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio devem somar 1.0, "
            f"mas somam {total_ratio:.4f}"
        )

    print("Extraindo Subject IDs e agrupando por sujeito...\n")

    subjects = defaultdict(lambda: {"indices": [], "label": None})

    for idx, (path, label) in enumerate(dataset.samples):
        filename = os.path.basename(path)
        subject_id = extract_subject_id(filename)
        subjects[subject_id]["indices"].append(idx)
        subjects[subject_id]["label"] = label

    subject_ids = list(subjects.keys())
    subject_labels = [subjects[s]["label"] for s in subject_ids]

    print(f"Dataset total: {len(dataset)} fatias de {len(subject_ids)} sujeitos\n")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=random_state
    )

    train_subj_idx, val_subj_idx = next(
        splitter.split(subject_ids, subject_labels)
    )

    train_subject_ids = [subject_ids[i] for i in train_subj_idx]
    val_subject_ids = [subject_ids[i] for i in val_subj_idx]

    train_subject_indices = {sid: subjects[sid]["indices"] for sid in train_subject_ids}
    val_subject_indices = {sid: subjects[sid]["indices"] for sid in val_subject_ids}

    train_dataset = SubjectSamplingSubset(
        dataset=dataset,
        subject_indices=train_subject_indices,
        max_slices=max_slices_per_subject,
        random_state=random_state,
        strategy='random'
    )

    val_dataset = SubjectSamplingSubset(
        dataset=dataset,
        subject_indices=val_subject_indices,
        max_slices=max_slices_per_subject,
        random_state=random_state,
        strategy='middle'
    )

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    train_labels = [dataset.samples[i][1] for i in train_indices]
    val_labels = [dataset.samples[i][1] for i in val_indices]

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)

    print("Distribuição por Classe:")
    print("\n  TREINO:")
    total_train = sum(train_counts.values())
    train_subj_per_class = Counter([subjects[s]["label"] for s in train_subject_ids])
    for cls in sorted(train_counts.keys()):
        count = train_counts[cls]
        pct = count / total_train * 100
        n_subj = train_subj_per_class.get(cls, 0)
        print(f"    Classe {cls}: {count:>5} fatias, {n_subj:>3} sujeitos ({pct:>5.1f}%)")
    print(f"    Total:     {total_train:>5} fatias, {len(train_subject_ids):>3} sujeitos")

    print("\n  VALIDAÇÃO:")
    total_val = sum(val_counts.values())
    val_subj_per_class = Counter([subjects[s]["label"] for s in val_subject_ids])
    for cls in sorted(val_counts.keys()):
        count = val_counts[cls]
        pct = count / total_val * 100
        n_subj = val_subj_per_class.get(cls, 0)
        print(f"    Classe {cls}: {count:>5} fatias, {n_subj:>3} sujeitos ({pct:>5.1f}%)")
    print(f"    Total:     {total_val:>5} fatias, {len(val_subject_ids):>3} sujeitos")

    overlap = set(train_subject_ids) & set(val_subject_ids)
    if overlap:
        print(f"\n  ERRO CRÍTICO: {len(overlap)} sujeitos em ambos os splits!")
    else:
        print(f"\n  Zero vazamento entre treino e validação")

    print(f"\n{'-' * 60}\n")

    return train_dataset, val_dataset

def verify_split_stratification(
        train_subset: Subset,
        val_subset: Subset,
        tolerance: float = 0.05
) -> bool:
    base_dataset = train_subset.dataset
    while hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset

    if hasattr(base_dataset, 'targets'):
        all_labels = np.array(base_dataset.targets)
    elif hasattr(base_dataset, 'labels'):
        all_labels = np.array(base_dataset.labels)
    else:
        all_labels = np.array([base_dataset[i][1] for i in range(len(base_dataset))])

    train_labels = all_labels[train_subset.indices]
    val_labels = all_labels[val_subset.indices]

    train_props = Counter(train_labels)
    val_props = Counter(val_labels)

    train_total = len(train_labels)
    val_total = len(val_labels)

    is_valid = True
    for cls in set(train_labels) | set(val_labels):
        train_pct = train_props.get(cls, 0) / train_total
        val_pct = val_props.get(cls, 0) / val_total

        diff = abs(train_pct - val_pct)

        if diff > tolerance:
            print(f"Classe {cls}: diferença de {diff:.1%} excede tolerância de {tolerance:.1%}\n")
            is_valid = False

    if is_valid:
        print("Estratificação válida\n")

    return is_valid

def get_split_statistics(
        train_subset: Subset,
        val_subset: Subset,
        test_subset: Subset = None
) -> dict:
    base_dataset = train_subset.dataset
    while hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset

    if hasattr(base_dataset, 'targets'):
        all_labels = np.array(base_dataset.targets)
    elif hasattr(base_dataset, 'labels'):
        all_labels = np.array(base_dataset.labels)
    else:
        all_labels = np.array([base_dataset[i][1] for i in range(len(base_dataset))])

    train_labels = all_labels[train_subset.indices]
    val_labels = all_labels[val_subset.indices]

    train_subjects = set()
    val_subjects = set()

    if hasattr(base_dataset, 'samples'):
        for idx in train_subset.indices:
            path = base_dataset.samples[idx][0]
            train_subjects.add(extract_subject_id(os.path.basename(path)))
        for idx in val_subset.indices:
            path = base_dataset.samples[idx][0]
            val_subjects.add(extract_subject_id(os.path.basename(path)))

    stats = {
        'train_size': len(train_labels),
        'val_size': len(val_labels),
        'train_subjects': len(train_subjects),
        'val_subjects': len(val_subjects),
        'subject_overlap': len(train_subjects & val_subjects),
        'train_distribution': dict(Counter(train_labels)),
        'val_distribution': dict(Counter(val_labels)),
    }

    if test_subset is not None:
        test_labels = all_labels[test_subset.indices]
        test_subjects = set()
        if hasattr(base_dataset, 'samples'):
            for idx in test_subset.indices:
                path = base_dataset.samples[idx][0]
                test_subjects.add(extract_subject_id(os.path.basename(path)))
        stats['test_size'] = len(test_labels)
        stats['test_subjects'] = len(test_subjects)
        stats['test_distribution'] = dict(Counter(test_labels))

    return stats