import os
import shutil
from typing import List, Tuple
from torchvision import datasets

from training.src.utils.dataset import split_dataset_train_test

def prepare_dataset_multiclass(
        output_base_path: str,
        demented_classes: List[str],
        train_ratio: float = 0.8,
        random_state: int = 42
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, List[str]]:
    print(f"{'-' * 60}")
    print("CRIANDO DATASET MULTICLASSE (3 TIPOS DE DEMÊNCIA)")
    print(f"{'-' * 60}\n")

    train_path = os.path.join(output_base_path, "splits/multiclass/train")
    test_path = os.path.join(output_base_path, "splits/multiclass/test")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_dataset = datasets.ImageFolder(root=train_path, transform=None)
        test_dataset = datasets.ImageFolder(root=test_path, transform=None)

        print("Dataset multiclasse já existe. Pulando criação.\n")
        print(f"   Dataset de treino: {len(train_dataset)} imagens")
        print(f"   Dataset de teste: {len(test_dataset)} imagens")
        print(f"   Classes: {train_dataset.classes}")

        return train_dataset, test_dataset, train_dataset.classes

    raw_dataset_path = os.path.join(output_base_path, "raw")
    temp_path = os.path.join(output_base_path, "splits/multiclass/temp")
    os.makedirs(temp_path, exist_ok=True)

    stats = {}

    for dementia_class in demented_classes:
        class_path = os.path.join(raw_dataset_path, dementia_class)

        if not os.path.exists(class_path):
            print(f"Classe {dementia_class} não encontrada, pulando...\n")
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
        print(f"{dementia_class}: {len(images)} imagens copiadas\n")

    train_dataset, test_dataset = split_dataset_train_test(
        dataset_path=temp_path,
        classes=demented_classes,
        train_ratio=train_ratio,
        output_train_path=train_path,
        output_test_path=test_path,
        random_state=random_state
    )

    print(f"\n{'-' * 60}")
    print("DATASET MULTICLASSE CRIADO COM SUCESSO")
    print(f"{'-' * 60}\n")

    return train_dataset, test_dataset, demented_classes