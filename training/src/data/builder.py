import os
import shutil
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .subject_manager import DatasetMetadata, split_dataset_train_test
from ..utils.config import load_binary_config, load_multiclass_config

_COPY_WORKERS = min(32, max(4, (os.cpu_count() or 4) * 2))

def _copy_single(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    shutil.copyfile(src, dst)

def _copy_images_parallel(
    source_path: str,
    dest_path: str,
    prefix: str = ""
) -> int:
    if not os.path.exists(source_path):
        return 0

    images = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    if not images:
        return 0

    os.makedirs(dest_path, exist_ok=True)
    tasks = []
    with ThreadPoolExecutor(max_workers=_COPY_WORKERS) as pool:
        for img in images:
            fname = f"{prefix}{img}" if prefix else img
            src = os.path.join(source_path, img)
            dst = os.path.join(dest_path, fname)
            tasks.append(pool.submit(_copy_single, src, dst))

        for future in as_completed(tasks):
            future.result()

    return len(images)

def _load_existing_split(
    train_path: str,
    test_path: str,
    label: str
) -> Tuple[DatasetMetadata, DatasetMetadata, List[str]] | None:
    if os.path.exists(train_path) and os.path.exists(test_path):
        classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
        
        train_ds = DatasetMetadata(root=train_path, classes=classes)
        test_ds  = DatasetMetadata(root=test_path,  classes=classes)
        
        print(f"Dataset {label} já existe. Pulando criação.\n")
        print(f"   Treino: {len(train_ds)} imagens | Teste: {len(test_ds)} imagens")
        print(f"   Classes: {train_ds.classes}\n")
        
        return train_ds, test_ds, train_ds.classes
    
    return None

def prepare_dataset_binary(output_base_path: str = "./shared/data") -> Tuple[DatasetMetadata, DatasetMetadata, List[str]]:
    config       = load_binary_config()
    data_config  = config['data']
    model_config = config['model']
    train_ratio  = data_config['split_ratios']['train']
    random_state = data_config['random_seed']
    class_names  = model_config['class_names']

    print(f"{'-' * 60}")
    print("PREPARAÇÃO DO DATASET BINÁRIO")
    print(f"{'-' * 60}")
    print(f"   Train Ratio:  {train_ratio}")
    print(f"   Random Seed:  {random_state}")
    print(f"   Classes:      {class_names}\n")

    train_path = os.path.join(output_base_path, "splits/binary/train")
    test_path  = os.path.join(output_base_path, "splits/binary/test")
    cached = _load_existing_split(train_path, test_path, "Binário")
    if cached:
        return cached

    raw_path  = os.path.join(output_base_path, "raw")
    class_mapping        = model_config['class_mapping']
    non_demented_folders = [k for k, v in class_mapping.items() if v == 1]
    demented_folders     = [k for k, v in class_mapping.items() if v == 0]

    temp_binary = os.path.join(output_base_path, "splits/binary/temp")
    non_dem_out = os.path.join(temp_binary, "Non Demented")
    dem_out     = os.path.join(temp_binary, "Demented")

    print("-" * 60)
    print("BINARIZANDO E COPIANDO DATASET (paralelo)")
    print("-" * 60)

    stats = {'Non Demented': 0, 'Demented': 0, 'classes_merged': {}}
    stats['Non Demented'] = _copy_images_parallel(os.path.join(raw_path, non_demented_folders[0]), non_dem_out)

    for cls in demented_folders:
        n = _copy_images_parallel(
            os.path.join(raw_path, cls),
            dem_out,
            prefix=f"{cls.replace(' ', '_')}_"
        ) 
        if n == 0:
            print(f"Classe '{cls}' não encontrada, pulando...")

        stats['classes_merged'][cls] = n
        stats['Demented'] += n

    total = stats['Non Demented'] + stats['Demented']
    print(f"\n  Non Demented: {stats['Non Demented']} | Demented: {stats['Demented']} | Total: {total}")
    if stats['classes_merged']:
        for cls, cnt in stats['classes_merged'].items():
            pct = cnt / stats['Demented'] * 100 if stats['Demented'] > 0 else 0
            print(f"    - {cls}: {cnt} ({pct:.1f}%)")

    binary_classes = ['Demented', 'Non Demented']
    train_ds, test_ds = split_dataset_train_test(
        dataset_path=temp_binary,
        classes=binary_classes,
        train_ratio=train_ratio,
        output_train_path=train_path,
        output_test_path=test_path,
        random_state=random_state
    )

    try:
        shutil.rmtree(temp_binary)
        print(f"\nPasta temporária removida: {temp_binary}")
    except Exception as e:
        print(f"\nErro ao remover pasta temporária: {e}")

    print(f"\n{'-' * 60}")
    print("PREPARAÇÃO DO DATASET BINÁRIO CONCLUÍDA")
    print(f"{'-' * 60}\n")
    return train_ds, test_ds, binary_classes

def prepare_dataset_multiclass(output_base_path: str = "./shared/data") -> Tuple[DatasetMetadata, DatasetMetadata, List[str]]:
    config       = load_multiclass_config()
    data_config  = config['data']
    model_config = config['model']
    train_ratio         = data_config['split_ratios']['train']
    random_state        = data_config['random_seed']
    class_names         = model_config['class_names']
    merge_classes       = model_config.get('merge_classes', {})
    filter_non_demented = model_config.get('filter_non_demented', True)

    print(f"{'-' * 60}")
    print("PREPARAÇÃO DO DATASET MULTICLASSE (NÍVEIS DE DEMÊNCIA)")
    print(f"{'-' * 60}")
    print(f"   Train Ratio:        {train_ratio}")
    print(f"   Random Seed:        {random_state}")
    print(f"   Classes:            {class_names}")
    print(f"   Merge Classes:      {merge_classes}")
    print(f"   Filtrar Non Dem.:   {filter_non_demented}\n")

    train_path = os.path.join(output_base_path, "splits/multiclass/train")
    test_path  = os.path.join(output_base_path, "splits/multiclass/test")
    cached = _load_existing_split(train_path, test_path, "Multiclasse")
    if cached:
        return cached

    raw_path  = os.path.join(output_base_path, "raw")
    temp_path = os.path.join(output_base_path, "splits/multiclass/temp")
    total_images = 0

    print("Copiando classes do dataset multiclasse (paralelo)...\n")
    for class_name in class_names:
        dest = os.path.join(temp_path, class_name)

        if class_name in merge_classes:
            source_folders = merge_classes[class_name]
            print(f"Classe '{class_name}' (merge de: {source_folders}):")
            class_count = 0

            for src_folder in source_folders:
                n = _copy_images_parallel(
                    os.path.join(raw_path, src_folder),
                    dest,
                    prefix=f"{src_folder.replace(' ', '_')}_"
                )
                if n == 0:
                    print(f"  AVISO: Pasta '{src_folder}' não encontrada, pulando...")
                else:
                    print(f"  {src_folder}: {n} imagens copiadas")
                class_count += n
            print(f"  Total: {class_count} imagens\n")
            total_images += class_count

        else:
            n = _copy_images_parallel(os.path.join(raw_path, class_name), dest)
            if n == 0:
                print(f"Classe '{class_name}' não encontrada, pulando...")
            else:
                print(f"{class_name}: {n} imagens copiadas")
            total_images += n

    print(f"\nTotal de imagens: {total_images}")
    print(f"Dividindo (train: {train_ratio * 100:.0f}%, test: {(1 - train_ratio) * 100:.0f}%)...\n")

    train_ds, test_ds = split_dataset_train_test(
        dataset_path=temp_path,
        classes=class_names,
        train_ratio=train_ratio,
        output_train_path=train_path,
        output_test_path=test_path,
        random_state=random_state
    )

    try:
        shutil.rmtree(temp_path)
        print(f"\nPasta temporária removida: {temp_path}")
    except Exception as e:
        print(f"\nErro ao remover pasta temporária: {e}")

    print(f"\n{'-' * 60}")
    print("PREPARAÇÃO DO DATASET MULTICLASSE CONCLUÍDA")
    print(f"{'-' * 60}\n")
    return train_ds, test_ds, class_names
