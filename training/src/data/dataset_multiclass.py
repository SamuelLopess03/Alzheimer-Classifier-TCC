import os
import shutil
from typing import List, Tuple
from torchvision import datasets

from ..utils import split_dataset_train_test, load_multiclass_config

def prepare_dataset_multiclass(
        output_base_path: str = "./shared/data"
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, List[str]]:
    config = load_multiclass_config()

    data_config = config['data']
    model_config = config['model']

    train_ratio = data_config['split_ratios']['train']
    random_state = data_config['random_seed']
    class_names = model_config['class_names']
    merge_classes = model_config.get('merge_classes', {})
    filter_non_demented = model_config.get('filter_non_demented', True)

    print(f"{'-' * 60}")
    print("INICIANDO ETAPA DE PREPARAÇÃO DO DATASET MULTICLASSE (NÍVEIS DE DEMÊNCIA)")
    print(f"{'-' * 60}")
    print(f"Configurações:")
    print(f"   Train Ratio: {train_ratio}")
    print(f"   Random Seed: {random_state}")
    print(f"   Classes: {class_names}")
    print(f"   Merge Classes: {merge_classes}")
    print(f"   Filtrar Non Demented: {filter_non_demented}\n")

    train_path = os.path.join(output_base_path, "splits/multiclass/train")
    test_path = os.path.join(output_base_path, "splits/multiclass/test")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_dataset = datasets.ImageFolder(root=train_path, transform=None)
        test_dataset = datasets.ImageFolder(root=test_path, transform=None)

        print("Dataset multiclasse já existe. Pulando criação.\n")
        print(f"   Dataset de treino: {len(train_dataset)} imagens")
        print(f"   Dataset de teste: {len(test_dataset)} imagens")
        print(f"   Classes: {train_dataset.classes}")

        print(f"\n{'-' * 60}")
        print("PREPARAÇÃO DO DATASET MULTICLASSE CONCLUÍDA")
        print(f"{'-' * 60}\n")

        return train_dataset, test_dataset, train_dataset.classes

    raw_dataset_path = os.path.join(output_base_path, "raw")
    temp_path = os.path.join(output_base_path, "splits/multiclass/temp")
    os.makedirs(temp_path, exist_ok=True)

    stats = {}
    total_images = 0

    print("Preparando classes do dataset multiclasse...\n")

    for class_name in class_names:
        output_class_path = os.path.join(temp_path, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        if class_name in merge_classes:
            source_folders = merge_classes[class_name]
            print(f"Classe '{class_name}' (merge de: {source_folders}):")

            class_count = 0
            for source_folder in source_folders:
                source_path = os.path.join(raw_dataset_path, source_folder)

                if not os.path.exists(source_path):
                    print(f"  AVISO: Pasta '{source_folder}' não encontrada, pulando...")
                    continue

                images = [f for f in os.listdir(source_path)
                          if f.lower().endswith(('.jpg', '.jpeg'))]

                for image in images:
                    src = os.path.join(source_path, image)
                    new_filename = f"{source_folder.replace(' ', '_')}_{image}"
                    dst = os.path.join(output_class_path, new_filename)
                    shutil.copy2(src, dst)

                print(f"  {source_folder}: {len(images)} imagens copiadas")
                class_count += len(images)

            stats[class_name] = class_count
            total_images += class_count
            print(f"  Total da classe: {class_count} imagens\n")

        else:
            source_path = os.path.join(raw_dataset_path, class_name)

            if not os.path.exists(source_path):
                print(f"Classe '{class_name}' não encontrada, pulando...")
                continue

            images = [f for f in os.listdir(source_path)
                      if f.lower().endswith(('.jpg', '.jpeg'))]

            for image in images:
                src = os.path.join(source_path, image)
                dst = os.path.join(output_class_path, image)
                shutil.copy2(src, dst)

            stats[class_name] = len(images)
            total_images += len(images)
            print(f"{class_name}: {len(images)} imagens copiadas")

    print(f"\nTotal de imagens: {total_images}")
    print(f"Dividindo dataset por sujeito (train: {train_ratio * 100:.0f}%, test: {(1 - train_ratio) * 100:.0f}%)...\n")

    train_dataset, test_dataset = split_dataset_train_test(
        dataset_path=temp_path,
        classes=class_names,
        train_ratio=train_ratio,
        output_train_path=train_path,
        output_test_path=test_path,
        random_state=random_state,
        stratify=data_config['stratify']
    )

    try:
        shutil.rmtree(temp_path)
        print(f"\nPasta temporária removida: {temp_path}")
    except Exception as e:
        print(f"\nErro ao remover pasta temporária: {e}")

    print(f"\n{'-' * 60}")
    print("PREPARAÇÃO DO DATASET MULTICLASSE CONCLUÍDA")
    print(f"{'-' * 60}\n")

    return train_dataset, test_dataset, class_names