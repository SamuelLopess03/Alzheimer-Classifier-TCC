import os
import re
import numpy as np
from PIL import Image
import zipfile
import shutil
from typing import Tuple, Optional, List, Dict
from collections import defaultdict
from torchvision import datasets

from .subject_utils import extract_subject_id, extract_slice_index

def find_dataset_classes(dataset_path: str) -> Optional[List[str]]:
    classes = []

    try:
        data_folder = dataset_path

        for item in os.listdir(data_folder):
            item_path = os.path.join(data_folder, item)
            if os.path.isdir(item_path) and has_images(item_path):
                classes.append(item)

        classes.sort()

    except Exception as e:
        print(f"\nErro ao buscar classes: {e}\n")
        return None

    return classes if classes else None

def has_images(folder_path: str, min_files: int = 1) -> bool:
    image_extensions = {'.jpg', '.jpeg'}
    image_count = 0

    try:
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_count += 1
                if image_count >= min_files:
                    return True
    except Exception as e:
        print(f"\nErro ao buscar images: {e}\n")
        return False

    return False

def download_kaggle_dataset(
        dataset_name: str,
        output_dir: str,
        kaggle_json_path: str
) -> Tuple[bool, Optional[List[str]]]:
    print(f"{'-' * 60}")
    print("INICIANDO DOWNLOAD DO DATASET")
    print(f"{'-' * 60}\n")

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    if kaggle_json_path and os.path.exists(kaggle_json_path):
        shutil.copy(kaggle_json_path, os.path.join(kaggle_dir, "kaggle.json"))
        os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
        print("Credenciais do Kaggle configuradas\n")
    else:
        kaggle_config = os.path.join(kaggle_dir, "kaggle.json")
        if not os.path.exists(kaggle_config):
            print("Erro: Arquivo kaggle.json não encontrado!\n")
            print("Configure suas credenciais em ~/.kaggle/kaggle.json\n")
            return False, None

    print(f"Baixando dataset '{dataset_name}' do Kaggle...\n")
    download_result = os.system(f"kaggle datasets download -d {dataset_name} -p {output_dir}")

    if download_result != 0:
        print("Erro no download. Verifique o nome do dataset e suas credenciais.\n")
        return False, None

    zip_filename = os.path.join(output_dir, dataset_name.split("/")[-1] + ".zip")
    if not os.path.exists(zip_filename):
        print("Arquivo zip não encontrado após download.\n")
        return False, None

    print(f"Extraindo arquivos para: {output_dir}\n")

    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Download e extração concluídos com sucesso\n")
    except Exception as e:
        print(f"Erro na extração: {e}\n")
        return False, None

    data_folder = os.path.join(output_dir, "Data")

    if os.path.exists(data_folder):
        for item in os.listdir(data_folder):
            src = os.path.join(data_folder, item)
            dst = os.path.join(output_dir, item)

            shutil.move(src, dst)

        shutil.rmtree(data_folder)

    classes = find_dataset_classes(output_dir)

    if classes:
        print(f"\nClasses encontradas ({len(classes)}):")
        for i, class_name in enumerate(classes, 1):
            print(f"   {i}. {class_name}")
    else:
        print("Nenhuma classe encontrada\n")

    try:
        os.remove(zip_filename)
        print(f"Arquivo zip removido: {zip_filename}\n")
    except Exception as e:
        print(f"Erro ao remover arquivo zip: {e}\n")
        pass

    print("-" * 60)
    print("DOWNLOAD DO DATASET FINALIZADO")
    print("-" * 60)

    return True, classes

def split_dataset_train_test(
        dataset_path: str,
        classes: List[str],
        train_ratio: float,
        output_train_path: str,
        output_test_path: str,
        random_state: int = 42,
        stratify: bool = True
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    print("\n" + "-" * 60)
    print(f"INICIANDO DIVISÃO DO DATASET POR SUJEITO ({int(train_ratio * 100)}% TREINO / {int((1 - train_ratio) * 100)}% TESTE)")
    print("-" * 60 + "\n")

    subjects_by_class: Dict[str, List[str]] = {}
    subject_files: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    total_files = 0

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"  AVISO: Classe '{class_name}' não encontrada em {dataset_path}")
            subjects_by_class[class_name] = []
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg'))]

        class_subjects = set()
        for img in images:
            subject_id = extract_subject_id(img)
            class_subjects.add(subject_id)
            subject_files[subject_id].append((class_name, img))

        subjects_by_class[class_name] = sorted(class_subjects)
        total_files += len(images)

    total_subjects = sum(len(s) for s in subjects_by_class.values())
    print(f"Dataset total: {total_files} fatias de {total_subjects} sujeitos\n")

    for class_name, subjs in subjects_by_class.items():
        n_slices = sum(len(subject_files[s]) for s in subjs)
        print(f"  {class_name}: {len(subjs)} sujeitos, {n_slices} fatias")
    print()

    rng = np.random.RandomState(random_state)

    train_subjects = []
    test_subjects = []

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

    train_count = 0
    test_count = 0

    for class_name in classes:
        os.makedirs(os.path.join(output_train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(output_test_path, class_name), exist_ok=True)

    for subject in train_subjects:
        for class_name, img in subject_files[subject]:
            src = os.path.join(dataset_path, class_name, img)
            dst = os.path.join(output_train_path, class_name, img)
            shutil.copy2(src, dst)
            train_count += 1

    for subject in test_subjects:
        for class_name, img in subject_files[subject]:
            src = os.path.join(dataset_path, class_name, img)
            dst = os.path.join(output_test_path, class_name, img)
            shutil.copy2(src, dst)
            test_count += 1

    train_dataset = datasets.ImageFolder(root=output_train_path, transform=None)
    test_dataset = datasets.ImageFolder(root=output_test_path, transform=None)

    print(f"Resumo Final:")
    print(f"  Treino: {len(train_subjects)} sujeitos, {len(train_dataset)} fatias")
    print(f"  Teste:  {len(test_subjects)} sujeitos, {len(test_dataset)} fatias")
    print(f"  Classes: {train_dataset.classes}")

    overlap = set(train_subjects) & set(test_subjects)
    if overlap:
        print(f"\n  ERRO CRÍTICO: {len(overlap)} sujeitos em ambos os splits: {overlap}")
    else:
        print(f"\n  Verificação de integridade: zero vazamento entre splits")

    print("\n" + "-" * 60)
    print("DIVISÃO DO DATASET POR SUJEITO CONCLUÍDA")
    print("-" * 60)

    return train_dataset, test_dataset
    
def validate_image_files(directory: str) -> Tuple[int, int, List[str]]:
    valid_count = 0
    corrupt_count = 0
    errors = []

    print(f"\nValidando imagens em: {directory}")

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, file)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                    valid_count += 1
                except Exception as e:
                    corrupt_count += 1
                    errors.append(f"{filepath}: {str(e)}")

    return valid_count, corrupt_count, errors

def resolve_subset_labels(split) -> np.ndarray:
    if hasattr(split.dataset, 'targets'):
        all_labels = np.array(split.dataset.targets)
    elif hasattr(split.dataset, 'labels'):
        all_labels = np.array(split.dataset.labels)
    else:
        all_labels = np.array([split.dataset[i][1] for i in range(len(split.dataset))])
        
    return all_labels[split.indices]
