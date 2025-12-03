import os
import shutil
from typing import List, Tuple
from torchvision import datasets

from training.src.utils import split_dataset_train_test, load_multiclass_config

def prepare_dataset_multiclass(
        output_base_path: str = "./shared/data"
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, List[str]]:
    config = load_multiclass_config()

    data_config = config['data']
    model_config = config['model']

    train_ratio = data_config['split_ratios']['train']
    random_state = data_config['random_seed']
    class_names = model_config['class_names']
    filter_non_demented = model_config.get('filter_non_demented', True)

    print(f"{'-' * 60}")
    print("INICIANDO ETAPA DE PREPARAÇÃO DO DATASET MULTICLASSE (NÍVEIS DE DEMÊNCIA)")
    print(f"{'-' * 60}")
    print(f"Configurações:")
    print(f"   Train Ratio: {train_ratio}")
    print(f"   Random Seed: {random_state}")
    print(f"   Classes: {class_names}")
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

    print("Copiando imagens das classes de demência...\n")

    for dementia_class in class_names:
        class_path = os.path.join(raw_dataset_path, dementia_class)

        if not os.path.exists(class_path):
            print(f"Classe {dementia_class} não encontrada, pulando...")
            continue

        output_class_path = os.path.join(temp_path, dementia_class)
        os.makedirs(output_class_path, exist_ok=True)

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg'))]

        for image in images:
            src = os.path.join(class_path, image)
            dst = os.path.join(output_class_path, image)
            shutil.copy2(src, dst)

        stats[dementia_class] = len(images)
        total_images += len(images)
        print(f"{dementia_class}: {len(images)} imagens copiadas")

    print(f"\nDividindo dataset (train: {train_ratio * 100:.0f}%, test: {(1 - train_ratio) * 100:.0f}%)...\n")

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