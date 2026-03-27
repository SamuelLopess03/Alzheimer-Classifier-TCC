import os
import re
import numpy as np
from torch.utils.data import Subset
from typing import Optional, List, Tuple

from ..utils.subject_utils import (
    extract_subject_id, 
    resolve_dataset_chain, 
    get_subject_ids_from_dataset
)

def count_unique_subjects(dataset, indices=None) -> int:
    base_ds, resolved_indices = resolve_dataset_chain(dataset, indices)
    
    if hasattr(base_ds, 'subject_indices') and isinstance(base_ds.subject_indices, dict):
        return len(base_ds.subject_indices)
        
    subjects = set()
    if hasattr(base_ds, 'samples'):
        if resolved_indices is not None:
            for idx in resolved_indices:
                try:
                    path = base_ds.samples[idx][0]
                    subjects.add(extract_subject_id(os.path.basename(path)))
                except: 
                    continue
        else:
            for path, _ in base_ds.samples:
                try:
                    subjects.add(extract_subject_id(os.path.basename(path)))
                except: 
                    continue
                
    return len(subjects) if subjects else 0

def resolve_subset_labels(split: Subset) -> np.ndarray:
    if hasattr(split.dataset, 'targets'):
        all_labels = np.array(split.dataset.targets)
    elif hasattr(split.dataset, 'labels'):
        all_labels = np.array(split.dataset.labels)
    else:
        all_labels = []
        for i in range(len(split)):
            _, label = split.dataset[i]
            all_labels.append(label)
        all_labels = np.array(all_labels)
    
    return all_labels[split.indices]

def get_subject_ids_from_dataset(ds) -> List[str]:
    base_ds, resolved_indices = resolve_dataset_chain(ds)
    
    subject_ids = []
    if hasattr(base_ds, 'samples'):
        if resolved_indices is not None:
            for idx in resolved_indices:
                path = base_ds.samples[idx][0]
                subject_ids.append(extract_subject_id(os.path.basename(path)))
        else:
            for path, _ in base_ds.samples:
                subject_ids.append(extract_subject_id(os.path.basename(path)))
    else:
        subject_ids = ["unknown"] * len(ds)
        
    return subject_ids
