import os
import shutil
from typing import Tuple, List, Dict
from torchvision import datasets

from training.src.utils import split_dataset_train_test, load_binary_config

def prepare_dataset_binary(
        output_base_path: str = "./shared/data"
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, List[str]]:
    config = load_binary_config()

    data_config = config['data']
    model_config = config['model']

    train_ratio = data_config['split_ratios']['train']
    random_state = data_config['random_seed']
    class_names = model_config['class_names']

    print(f"{'-' * 60}")
    print("INICIANDO ETAPA DE PREPARAÇÃO DO DATASET BINÁRIO")
    print(f"{'-' * 60}")
    print(f"Configurações:")
    print(f"   Train Ratio: {train_ratio}")
    print(f"   Random Seed: {random_state}")
    print(f"   Classes: {class_names}\n")

    train_binary_path = os.path.join(output_base_path, "splits/binary/train")
    test_binary_path = os.path.join(output_base_path, "splits/binary/test")

    if os.path.exists(train_binary_path) and os.path.exists(test_binary_path):
        train_dataset = datasets.ImageFolder(root=train_binary_path, transform=None)
        test_dataset = datasets.ImageFolder(root=test_binary_path, transform=None)

        print("Dataset já está pronto e binarizado. Pulando preparação.\n")
        print(f"   Dataset de treino: {len(train_dataset)} imagens")
        print(f"   Dataset de teste: {len(test_dataset)} imagens")
        print(f"   Classes: {train_dataset.classes}")

        print(f"\n{'-' * 60}")
        print("PREPARAÇÃO DO DATASET BINÁRIO CONCLUÍDA")
        print(f"{'-' * 60}\n")

        return train_dataset, test_dataset, train_dataset.classes

    raw_dataset_path = os.path.join(output_base_path, "raw")
    binary_dataset_path = os.path.join(output_base_path, "splits/binary/temp")
    os.makedirs(binary_dataset_path, exist_ok=True)

    class_mapping = model_config['class_mapping']

    original_classes = {
        0: "Non Demented",
        1: "Very Mild Dementia",
        2: "Mild Dementia",
        3: "Moderate Dementia"
    }

    non_demented_classes = [original_classes[k] for k, v in class_mapping.items() if v == 1]
    demented_classes = [original_classes[k] for k, v in class_mapping.items() if v == 0]

    binary_dataset_path, binary_classes_aux = binarize_alzheimer_dataset(
        dataset_path=raw_dataset_path,
        output_path=binary_dataset_path,
        non_demented_folder=non_demented_classes[0],
        demented_classes=demented_classes
    )

    train_dataset, test_dataset = split_dataset_train_test(
        dataset_path=binary_dataset_path,
        classes=binary_classes_aux,
        train_ratio=train_ratio,
        output_train_path=train_binary_path,
        output_test_path=test_binary_path,
        random_state=random_state,
        stratify=data_config['stratify']
    )

    try:
        shutil.rmtree(binary_dataset_path)
        print(f"\nPasta temporária removida: {binary_dataset_path}")
    except Exception as e:
        print(f"\nErro ao remover pasta temporária: {e}")

    print(f"\n{'-' * 60}")
    print("PREPARAÇÃO DO DATASET BINÁRIO CONCLUÍDA")
    print(f"{'-' * 60}\n")

    return train_dataset, test_dataset, binary_classes_aux

def binarize_alzheimer_dataset(
        dataset_path: str,
        output_path: str,
        non_demented_folder: str,
        demented_classes: List[str]
) -> Tuple[str, List[str]]:
    print("-" * 60)
    print("INICIANDO BINARIZAÇÃO DO DATASET")
    print("-" * 60)

    non_demented_output = os.path.join(output_path, "Non Demented")
    demented_output = os.path.join(output_path, "Demented")

    os.makedirs(non_demented_output, exist_ok=True)
    os.makedirs(demented_output, exist_ok=True)

    print(f"Estrutura de saída criada em: {output_path}\n")

    stats = {
        'Non Demented': 0,
        'Demented': 0,
        'classes_merged': {}
    }

    non_demented_path = os.path.join(dataset_path, non_demented_folder)
    if os.path.exists(non_demented_path):
        images = [f for f in os.listdir(non_demented_path)
                  if f.lower().endswith(('.jpg', '.jpeg'))]

        for image in images:
            src = os.path.join(non_demented_path, image)
            dst = os.path.join(non_demented_output, image)
            shutil.copy2(src, dst)

        stats['Non Demented'] = len(images)

    else:
        print(f"Pasta {non_demented_folder} não encontrada!")

    for dementia_class in demented_classes:
        class_path = os.path.join(dataset_path, dementia_class)

        if not os.path.exists(class_path):
            print(f"Classe {dementia_class} não encontrada, pulando...\n")
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg'))]

        for image in images:
            src = os.path.join(class_path, image)
            new_filename = f"{dementia_class.replace(' ', '_')}_{image}"
            dst = os.path.join(demented_output, new_filename)
            shutil.copy2(src, dst)

        stats['classes_merged'][dementia_class] = len(images)
        stats['Demented'] += len(images)

    print(f"\nEstatísticas Finais:")
    print(f"  Non Demented: {stats['Non Demented']} imagens")
    print(f"  Demented: {stats['Demented']} imagens")

    if stats['classes_merged']:
        print(f"    Composição da classe Demented:")
        for class_name, count in stats['classes_merged'].items():
            percentage = (count / stats['Demented'] * 100) if stats['Demented'] > 0 else 0
            print(f"      - {class_name}: {count} ({percentage:.1f}%)")

    total = stats['Non Demented'] + stats['Demented']
    if total > 0:
        print(f"\n  Balanceamento:")
        print(f"    Non Demented: {stats['Non Demented'] / total * 100:.1f}%")
        print(f"    Demented: {stats['Demented'] / total * 100:.1f}%")

    print("\n" + "-" * 60)
    print("BINARIZAÇÃO CONCLUÍDA")
    print("-" * 60 + "\n")

    binary_classes = ['Demented', 'Non Demented']
    return output_path, binary_classes