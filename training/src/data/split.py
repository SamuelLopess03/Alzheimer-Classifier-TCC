import os
import shutil
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from collections import defaultdict

from .subject_utils import extract_subject_id

def fast_copy(src: str, dst: str):
    try:
        if hasattr(os, 'link'):
            os.link(src, dst)
        else:
            shutil.copy(src, dst)
    except (OSError, AttributeError):
        shutil.copy(src, dst)

class DatasetMetadata:
    def __init__(self, root: str, classes: List[str]):
        self.root = root
        self.classes = classes
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        for cls_name in self.classes:
            cls_path = os.path.join(self.root, cls_name)
            if os.path.exists(cls_path):
                for f in os.listdir(cls_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(cls_path, f), cls_name))
    
    def __len__(self) -> int:
        return len(self.samples)

def _map_subjects_and_files(
        dataset_path: str, 
        classes: List[str]
) -> Tuple[Dict[str, List[str]], Dict[str, List[Tuple[str, str]]], int]:
    subjects_by_class: Dict[str, List[str]] = {}
    subject_files: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    total_files = 0

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"  AVISO: Classe '{class_name}' não encontrada em {dataset_path}")
            subjects_by_class[class_name] = []
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        class_subjects = set()
        for img in images:
            subject_id = extract_subject_id(img)
            class_subjects.add(subject_id)
            subject_files[subject_id].append((class_name, img))

        subjects_by_class[class_name] = sorted(class_subjects)
        total_files += len(images)

    return subjects_by_class, subject_files, total_files

def _perform_subject_split_math(
        classes: List[str],
        subjects_by_class: Dict[str, List[str]], 
        subject_files: Dict[str, List[Tuple[str, str]]],
        train_ratio: float, 
        random_state: int
) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(random_state)
    train_subjects, test_subjects = [], []

    print("Dividindo sujeitos por classe:")
    for class_name in classes:
        class_subjs = subjects_by_class[class_name].copy()
        rng.shuffle(class_subjs)
        
        n_train = max(1, int(len(class_subjs) * train_ratio))
        if n_train == len(class_subjs) and len(class_subjs) > 1:
            n_train = len(class_subjs) - 1

        train_subjs = class_subjs[:n_train]
        test_subjs = class_subjs[n_train:]
        train_subjects.extend(train_subjs)
        test_subjects.extend(test_subjs)

        train_slices = sum(len(subject_files[s]) for s in train_subjs)
        test_slices = sum(len(subject_files[s]) for s in test_subjs)

        print(f"  {class_name}:")
        print(f"    Treino: {len(train_subjs)} sujeitos ({train_slices} fatias)")
        print(f"    Teste:  {len(test_subjs)} sujeitos ({test_slices} fatias)")
    print()

    return train_subjects, test_subjects

def _copy_split_files(
        dataset_path: str, 
        output_train_path: str, 
        output_test_path: str, 
        train_subjects: List[str], 
        test_subjects: List[str], 
        subject_files: Dict[str, List[Tuple[str, str]]], 
        classes: List[str]
):
    for class_name in classes:
        os.makedirs(os.path.join(output_train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(output_test_path, class_name), exist_ok=True)

    for subject in train_subjects:
        for class_name, img in subject_files[subject]:
            fast_copy(os.path.join(dataset_path, class_name, img), os.path.join(output_train_path, class_name, img))

    for subject in test_subjects:
        for class_name, img in subject_files[subject]:
            fast_copy(os.path.join(dataset_path, class_name, img), os.path.join(output_test_path, class_name, img))

def _verify_split_leakage(train_subjects: List[str], test_subjects: List[str]):
    overlap = set(train_subjects) & set(test_subjects)
    if overlap:
        print(f"\n  ERRO CRÍTICO: {len(overlap)} sujeitos em ambos os splits: {overlap}")
    else:
        print(f"\n  Verificação de integridade: zero vazamento de pacientes (sujeitos) entre splits")

def split_dataset_train_test(
        dataset_path: str,
        classes: List[str],
        train_ratio: float,
        output_train_path: str,
        output_test_path: str,
        random_state: int = 42,
        stratify: bool = True
) -> Tuple[DatasetMetadata, DatasetMetadata]:
    print("\n" + "-" * 60)
    print(f"INICIANDO DIVISÃO DO DATASET POR SUJEITO ({int(train_ratio * 100)}% TREINO / {int((1 - train_ratio) * 100)}% TESTE)")
    print("-" * 60 + "\n")

    subjects_by_class, subject_files, total_files = _map_subjects_and_files(dataset_path, classes)
    total_subjects = sum(len(s) for s in subjects_by_class.values())
    print(f"Dataset total: {total_files} fatias de {total_subjects} sujeitos\n")

    train_subjects, test_subjects = _perform_subject_split_math(
        classes, subjects_by_class, subject_files, train_ratio, random_state
    )

    _copy_split_files(
        dataset_path, output_train_path, output_test_path, 
        train_subjects, test_subjects, subject_files, classes
    )

    train_dataset = DatasetMetadata(root=output_train_path, classes=classes)
    test_dataset = DatasetMetadata(root=output_test_path, classes=classes)

    print(f"Resumo Final:")
    print(f"  Treino: {len(train_subjects)} sujeitos, {len(train_dataset)} fatias")
    print(f"  Teste:  {len(test_subjects)} sujeitos, {len(test_dataset)} fatias")
    print(f"  Classes: {train_dataset.classes}")

    _verify_split_leakage(train_subjects, test_subjects)

    print("\n" + "-" * 60)
    print("DIVISÃO DO DATASET POR SUJEITO CONCLUÍDA")
    print("-" * 60)

    return train_dataset, test_dataset
