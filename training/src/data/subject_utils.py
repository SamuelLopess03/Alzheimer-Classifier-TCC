import re
import os
import numpy as np
from typing import List, Tuple, Optional, Any

def extract_subject_id(filename: str) -> str:
    match = re.search(r'(OAS\d+_\d+)', filename)
    if match:
        return match.group(1)
    return "unknown"

def extract_slice_index(filename: str) -> int:
    match = re.search(r'_(\d+)\.(?:jpg|jpeg|png)$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

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
        base_ds, orig_base_indices = resolve_dataset_chain(curr.original_dataset, None)
        
        if curr_indices is None:
            curr_indices = np.arange(len(curr))
            
        resolved_base_indices = []
        orig_len = len(curr.original_dataset)
        
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

    if hasattr(base_ds, 'subject_indices') and isinstance(base_ds.subject_indices, dict):
        return len(base_ds.subject_indices)

    subjects = set()
    if hasattr(base_ds, 'samples'):
        targets = (
            (base_ds.samples[i][0] for i in resolved_indices)
            if resolved_indices is not None
            else (p for p, _ in base_ds.samples)
        )
        
        for path in targets:
            try:
                subjects.add(extract_subject_id(os.path.basename(path)))
            except Exception:
                continue
    
    return len(subjects) if subjects else 0

def resolve_subset_labels(split: Any) -> np.ndarray:
    if hasattr(split.dataset, 'targets'):
        all_labels = np.array(split.dataset.targets)
    elif hasattr(split.dataset, 'labels'):
        all_labels = np.array(split.dataset.labels)
    else:
        all_labels = np.array([split.dataset[i][1] for i in range(len(split.dataset))])
    return all_labels[split.indices]

def get_subject_ids_from_dataset(dataset: Any) -> List[str]:
    base_ds, resolved_indices = resolve_dataset_chain(dataset)
    
    subject_ids = []
    
    if hasattr(base_ds, 'samples'):
        targets = (
            (base_ds.samples[i][0] for i in resolved_indices)
            if resolved_indices is not None
            else (p for p, _ in base_ds.samples)
        )
        
        for path in targets:
            subject_ids.append(extract_subject_id(os.path.basename(path)))
            
    return subject_ids
