import os
import re
import shutil
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from collections import defaultdict

def extract_subject_id(filename: str) -> str:
    match = re.search(r'(OAS\d+_\d+)', filename)
    return match.group(1) if match else "unknown"

def extract_slice_index(filename: str) -> int:
    match = re.search(r'_(\d+)\.(?:jpg|jpeg|png)$', filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def resolve_dataset_chain(ds: Any, indices: Optional[np.ndarray] = None) -> Tuple[Any, Optional[np.ndarray]]:
    from torch.utils.data import Subset
    curr = ds
    curr_indices = indices

    if isinstance(curr, Subset):
        new_indices = np.array(curr.indices)
        if curr_indices is not None:
            new_indices = new_indices[curr_indices]
        return resolve_dataset_chain(curr.dataset, new_indices)

    if hasattr(curr, 'original_dataset') and hasattr(curr, 'synthetic_indices'):
        orig_len = len(curr.original_dataset)
        base_ds, orig_base_indices = resolve_dataset_chain(curr.original_dataset, None)
        
        if curr_indices is None:
            curr_indices = np.arange(len(curr))
            
        resolved_base_indices = []
        for idx in curr_indices:
            if idx < orig_len:
                resolved_base_indices.append(orig_base_indices[idx])
            else:
                synth_idx = idx - orig_len
                resolved_base_indices.append(curr.synthetic_indices[synth_idx])
                
        return base_ds, np.array(resolved_base_indices)

    if hasattr(curr, 'subset_dataset'):
        return resolve_dataset_chain(curr.subset_dataset, curr_indices)
        
    return curr, curr_indices

def count_unique_subjects(dataset: Any, indices=None) -> int:
    base_ds, resolved_indices = resolve_dataset_chain(dataset, indices)

    subjects = set()
    if hasattr(base_ds, 'samples'):
        targets = (base_ds.samples[i][0] for i in resolved_indices) if resolved_indices is not None else (p for p, _ in base_ds.samples)
        for path in targets:
            subjects.add(extract_subject_id(os.path.basename(path)))
    
    return len(subjects)

def get_subject_ids_from_dataset(dataset: Any) -> List[str]:
    base_ds, resolved_indices = resolve_dataset_chain(dataset)
    subject_ids = []
    
    if hasattr(base_ds, 'samples'):
        targets = (base_ds.samples[i][0] for i in resolved_indices) if resolved_indices is not None else (p for p, _ in base_ds.samples)
        for path in targets:
            subject_ids.append(extract_subject_id(os.path.basename(path)))
            
    return subject_ids

def resolve_subset_labels(split: Any) -> np.ndarray:
    if hasattr(split.dataset, 'targets'):
        all_labels = np.array(split.dataset.targets)
    elif hasattr(split.dataset, 'labels'):
        all_labels = np.array(split.dataset.labels)
    else:
        all_labels = np.array([split.dataset[i][1] for i in range(len(split.dataset))])
    return all_labels[split.indices]

class DatasetMetadata:
    def __init__(self, root: str, classes: List[str]):
        self.root = root
        self.classes = classes
        self.samples = []
        for cls in classes:
            p = os.path.join(root, cls)
            if os.path.exists(p):
                self.samples.extend([(os.path.join(p, f), cls) for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    def __len__(self) -> int: return len(self.samples)

def split_dataset_train_test(
        dataset_path: str, classes: List[str], train_ratio: float,
        output_train_path: str, output_test_path: str, random_state: int = 42
) -> Tuple[DatasetMetadata, DatasetMetadata]:
    print(f"\nDividindo dataset: {int(train_ratio*100)}% Treino / {int((1-train_ratio)*100)}% Teste")

    subject_files = defaultdict(list)
    subjects_by_class = defaultdict(set)
    for cls in classes:
        p = os.path.join(dataset_path, cls)
        for img in [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg'))]:
            sid = extract_subject_id(img)
            subjects_by_class[cls].add(sid)
            subject_files[sid].append((cls, img))

    rng = np.random.RandomState(random_state)
    train_subjs, test_subjs = [], []
    for cls in classes:
        subjs = sorted(list(subjects_by_class[cls]))
        rng.shuffle(subjs)
        n_train = max(1, int(len(subjs) * train_ratio))
        train_subjs.extend(subjs[:n_train])
        test_subjs.extend(subjs[n_train:])

    for sid in train_subjs:
        for cls, img in subject_files[sid]:
            os.makedirs(os.path.join(output_train_path, cls), exist_ok=True)
            shutil.copy(os.path.join(dataset_path, cls, img), os.path.join(output_train_path, cls, img))
            
    for sid in test_subjs:
        for cls, img in subject_files[sid]:
            os.makedirs(os.path.join(output_test_path, cls), exist_ok=True)
            shutil.copy(os.path.join(dataset_path, cls, img), os.path.join(output_test_path, cls, img))

    return DatasetMetadata(output_train_path, classes), DatasetMetadata(output_test_path, classes)
