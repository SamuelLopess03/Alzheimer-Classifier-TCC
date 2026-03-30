import os
import shutil
from typing import Tuple, List

from .split import DatasetMetadata, split_dataset_train_test, fast_copy
from ..utils.config_loader import load_binary_config, load_multiclass_config

def _copy_images(source_path: str, dest_path: str, prefix: str = "") -> int:
    if not os.path.exists(source_path):
        return 0
    images = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    os.makedirs(dest_path, exist_ok=True)
    for img in images:
        src = os.path.join(source_path, img)
        fname = f"{prefix}{img}" if prefix else img
        fast_copy(src, os.path.join(dest_path, fname))
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

def _split_and_cleanup(
    temp_path: str,
    classes: List[str],
    train_path: str,
    test_path: str,
    train_ratio: float,
    random_state: int,
    stratify: bool
) -> Tuple[DatasetMetadata, DatasetMetadata]:
    train_ds, test_ds = split_dataset_train_test(
        dataset_path=temp_path,
        classes=classes,
        train_ratio=train_ratio,
        output_train_path=train_path,
        output_test_path=test_path,
        random_state=random_state,
        stratify=stratify
    )
    try:
        shutil.rmtree(temp_path)
        print(f"\nPasta temporária removida: {temp_path}")
    except Exception as e:
        print(f"\nErro ao remover pasta temporária: {e}")
    return train_ds, test_ds

def binarize_alzheimer_dataset(
    dataset_path: str,
    output_path: str,
    non_demented_folder: str,
    demented_classes: List[str]
) -> Tuple[str, List[str]]:
    print("-" * 60)
    print("INICIANDO BINARIZAÇÃO DO DATASET")
    print("-" * 60)
    non_dem_out = os.path.join(output_path, "Non Demented")
    dem_out     = os.path.join(output_path, "Demented")
    os.makedirs(non_dem_out, exist_ok=True)
    os.makedirs(dem_out, exist_ok=True)

    stats: dict = {'Non Demented': 0, 'Demented': 0, 'classes_merged': {}}
    stats['Non Demented'] = _copy_images(os.path.join(dataset_path, non_demented_folder), non_dem_out)

    for cls in demented_classes:
        n = _copy_images(
            os.path.join(dataset_path, cls),
            dem_out,
            prefix=f"{cls.replace(' ', '_')}_"
        )
        if n == 0:
            print(f"Classe '{cls}' não encontrada, pulando...")
        stats['classes_merged'][cls] = n
        stats['Demented'] += n

    total = stats['Non Demented'] + stats['Demented']
    print(f"\nEstatísticas Finais:")
    print(f"  Non Demented: {stats['Non Demented']} imagens")
    print(f"  Demented:     {stats['Demented']} imagens")

    if stats['classes_merged']:
        print("    Composição da classe Demented:")
        for cls, cnt in stats['classes_merged'].items():
            pct = cnt / stats['Demented'] * 100 if stats['Demented'] > 0 else 0
            print(f"      - {cls}: {cnt} ({pct:.1f}%)")

    if total > 0:
        print(f"\n  Balanceamento:")
        print(f"    Non Demented: {stats['Non Demented'] / total * 100:.1f}%")
        print(f"    Demented:     {stats['Demented'] / total * 100:.1f}%")

    print("\n" + "-" * 60)
    print("BINARIZAÇÃO CONCLUÍDA")
    print("-" * 60 + "\n")
    return output_path, ['Demented', 'Non Demented']

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
    if cached: return cached

    raw_path  = os.path.join(output_base_path, "raw")
    temp_path = os.path.join(output_base_path, "splits/binary/temp")
    os.makedirs(temp_path, exist_ok=True)

    class_mapping      = model_config['class_mapping']
    non_demented_folders = [k for k, v in class_mapping.items() if v == 1]
    demented_folders     = [k for k, v in class_mapping.items() if v == 0]

    temp_path, binary_classes = binarize_alzheimer_dataset(
        dataset_path=raw_path,
        output_path=temp_path,
        non_demented_folder=non_demented_folders[0],
        demented_classes=demented_folders
    )

    train_ds, test_ds = _split_and_cleanup(
        temp_path, binary_classes, train_path, test_path,
        train_ratio, random_state, data_config['stratify']
    )

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
    if cached: return cached

    raw_path  = os.path.join(output_base_path, "raw")
    temp_path = os.path.join(output_base_path, "splits/multiclass/temp")
    total_images = 0
    print("Preparando classes do dataset multiclasse...\n")

    for class_name in class_names:
        dest = os.path.join(temp_path, class_name)
        if class_name in merge_classes:
            source_folders = merge_classes[class_name]
            print(f"Classe '{class_name}' (merge de: {source_folders}):")
            class_count = 0
            for src_folder in source_folders:
                n = _copy_images(
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
            n = _copy_images(os.path.join(raw_path, class_name), dest)
            if n == 0:
                print(f"Classe '{class_name}' não encontrada, pulando...")
            else:
                print(f"{class_name}: {n} imagens copiadas")
            total_images += n

    print(f"\nTotal de imagens: {total_images}")
    print(f"Dividindo (train: {train_ratio * 100:.0f}%, test: {(1 - train_ratio) * 100:.0f}%)...\n")

    train_ds, test_ds = _split_and_cleanup(
        temp_path, class_names, train_path, test_path,
        train_ratio, random_state, data_config['stratify']
    )

    print(f"\n{'-' * 60}")
    print("PREPARAÇÃO DO DATASET MULTICLASSE CONCLUÍDA")
    print(f"{'-' * 60}\n")
    return train_ds, test_ds, class_names
