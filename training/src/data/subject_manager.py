import os
import re
import shutil
import numpy as np
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from collections import defaultdict

def extract_subject_id(filename: str) -> str:
    # 1. Tenta o padrão OASIS-1/2: OAS1_0001
    match_oas12 = re.search(r'(OAS[12]_\d{4})', filename, re.IGNORECASE)
    if match_oas12:
        return match_oas12.group(1).upper()
        
    # 2. Tenta o padrão OASIS-3: OAS30001
    match_oas3 = re.search(r'(OAS3\d{4,5})', filename, re.IGNORECASE)
    if match_oas3:
        return match_oas3.group(1).upper()
    
    # 3. Fallback genérico para qualquer padrão OAS
    match_generic = re.search(r'(OAS\d+(?:_\d+)?)', filename, re.IGNORECASE)
    return match_generic.group(1).upper() if match_generic else "unknown"
    
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

    if hasattr(curr, 'sampling_subset'):
        sampling_subset = curr.sampling_subset
        real_len = len(sampling_subset.indices)
        
        if curr_indices is None:
            curr_indices = np.arange(len(curr))
            
        resolved_indices = []
        for idx in curr_indices:
            if idx < real_len:
                resolved_indices.append(sampling_subset.indices[idx])
            else:
                synth_idx = idx - real_len
                resolved_indices.append(sampling_subset.synthetic_indices[synth_idx])
                
        return resolve_dataset_chain(sampling_subset.dataset, np.array(resolved_indices))

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
        output_train_path: str, output_test_path: str, random_state: int = 42,
        workers: int = None
) -> Tuple[DatasetMetadata, DatasetMetadata]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if workers is None:
        workers = min(32, max(4, (os.cpu_count() or 4) * 2))

    print(f"\nDividindo dataset: {int(train_ratio*100)}% Treino / {int((1-train_ratio)*100)}% Teste")

    subject_files = defaultdict(list)
    subjects_by_class = defaultdict(set)
    for cls in classes:
        p = os.path.join(dataset_path, cls)
        if not os.path.exists(p):
            continue
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

    for cls in classes:
        os.makedirs(os.path.join(output_train_path, cls), exist_ok=True)
        os.makedirs(os.path.join(output_test_path, cls), exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for sid in train_subjs:
            for cls, img in subject_files[sid]:
                src = os.path.join(dataset_path, cls, img)
                dst = os.path.join(output_train_path, cls, img)
                futures.append(pool.submit(shutil.copyfile, src, dst))

        for sid in test_subjs:
            for cls, img in subject_files[sid]:
                src = os.path.join(dataset_path, cls, img)
                dst = os.path.join(output_test_path, cls, img)
                futures.append(pool.submit(shutil.copyfile, src, dst))
                
        for future in as_completed(futures):
            future.result()

    train_count = sum(len(subject_files[sid]) for sid in train_subjs)
    test_count  = sum(len(subject_files[sid]) for sid in test_subjs)
    print(f"  Train: {train_count} imagens ({len(train_subjs)} sujeitos) | Test: {test_count} imagens ({len(test_subjs)} sujeitos)")

    return DatasetMetadata(output_train_path, classes), DatasetMetadata(output_test_path, classes)

def get_central_slices_per_class(
    dataset: Any, 
    num_classes: int, 
    samples_per_class: int,
    indices: Optional[np.ndarray] = None
) -> Dict[int, List[int]]:
    class_indices = {i: [] for i in range(num_classes)}
    
    if not hasattr(dataset, 'samples'):
        return class_indices

    for class_idx in range(num_classes):
        if indices is not None:
            class_items = [(i, dataset.samples[i][0]) for i in indices if dataset.samples[i][1] == class_idx]
        else:
            class_items = [(i, path) for i, (path, label) in enumerate(dataset.samples) if label == class_idx]
            
        if not class_items:
            continue
            
        pacientes_indices = {}
        for idx, path in class_items:
            sid = extract_subject_id(os.path.basename(path))
            if sid not in pacientes_indices:
                pacientes_indices[sid] = []
            pacientes_indices[sid].append(idx)
            
        if not pacientes_indices:
            continue
            
        paciente_escolhido = max(pacientes_indices.keys(), key=lambda sid: len(pacientes_indices[sid]))
        paciente_indices_lista = pacientes_indices[paciente_escolhido]
        
        paciente_indices_lista.sort(key=lambda idx: extract_slice_index(os.path.basename(dataset.samples[idx][0])))
        
        center_idx = len(paciente_indices_lista) // 2
        start_idx = max(0, center_idx - (samples_per_class // 2))
        end_idx = min(len(paciente_indices_lista), start_idx + samples_per_class)
        
        selected = paciente_indices_lista[start_idx:end_idx]
        
        class_indices[class_idx] = selected
        
    return class_indices
