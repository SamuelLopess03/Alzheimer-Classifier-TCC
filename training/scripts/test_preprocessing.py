import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from collections import Counter

from src.data import (
    MedicalImagePreprocessor,
    DynamicAugmentationDataset,
    StaticPreprocessedDataset,
    augment_minority_class
)
from src.utils import load_binary_config, load_multiclass_config

def test_preprocessor():
    print("\n" + "-" * 60)
    print("TESTANDO PREPROCESSOR")
    print("-" * 60 + "\n")

    architectures = [
        'resnext50_32x4d',
        'convnext_tiny',
        'efficientnetv2_s',
        'densenet121',
        'vit_b_16',
        'swin_v2_tiny'
    ]

    for arch in architectures:
        preprocessor = MedicalImagePreprocessor(arch)
        preprocessor.print_config()

def test_augmentation_pipeline(
        dataset_type: str = "binary",
        architecture: str = "resnext50_32x4d",
        batch_size: int = 16
):
    if dataset_type == "binary":
        config = load_binary_config()
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared/data/splits/binary/train')
    else:
        config = load_multiclass_config()
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared/data/splits/multiclass/train')

    model_config = config['model']
    class_names = model_config['class_names']
    num_classes = model_config['num_classes']

    print("\n" + "-" * 60)
    print(f"TESTANDO AUGMENTATION PIPELINE: {dataset_type.upper()}")
    print("-" * 60)
    print(f"Arquitetura: {architecture}")
    print(f"Classes: {class_names}")
    print(f"Número de classes: {num_classes}")
    print("-" * 60 + "\n")

    if not os.path.exists(data_path):
        print(f"Dataset não encontrado: {data_path}")
        print("Execute primeiro: python scripts/prepare_data.py\n")
        return False

    print("Carregando dataset...\n")
    original_dataset = datasets.ImageFolder(root=data_path, transform=None)

    print(f"Dataset carregado: {len(original_dataset)} imagens")
    print(f"  Classes encontradas: {original_dataset.classes}")
    print(f"  Distribuição:")

    targets = [label for _, label in original_dataset.samples]
    class_counts = Counter(targets)
    for class_idx, class_name in enumerate(original_dataset.classes):
        count = class_counts.get(class_idx, 0)
        percentage = (count / len(original_dataset) * 100) if len(original_dataset) > 0 else 0
        print(f"    {class_name}: {count} ({percentage:.1f}%)")

    train_size = int(0.8 * len(original_dataset))
    val_size = len(original_dataset) - train_size

    train_subset, val_subset = random_split(
        original_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nSplit:")
    print(f"  Train: {len(train_subset)} imagens")
    print(f"  Val: {len(val_subset)} imagens\n")

    print("-" * 60)
    print("CRIANDO DATASETS COM AUGMENTATION")
    print("-" * 60 + "\n")

    print("Criando dataset de treino com augmentation dinâmica...\n")
    train_augmented = DynamicAugmentationDataset(
        subset_dataset=train_subset,
        architecture_name=architecture
    )

    print("Criando dataset de validação com preprocessing estático...\n")
    val_preprocessed = StaticPreprocessedDataset(
        subset_dataset=val_subset,
        architecture_name=architecture
    )

    print("Criando DataLoaders...\n")
    train_loader = DataLoader(
        train_augmented,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_preprocessed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print("-" * 60)
    print("TESTANDO BATCHES")
    print("-" * 60 + "\n")

    print(f"Testando batch do TREINO...")
    try:
        images, labels = next(iter(train_loader))

        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Image range: [{images.min():.4f}, {images.max():.4f}]")
        print(f"  Labels únicos: {labels.unique().tolist()}")
        print(f"  Esperado {num_classes} classes: {len(labels.unique()) <= num_classes}")
    except Exception as e:
        print(f"  Erro ao carregar batch de treino: {e}")
        return False

    print(f"\nTestando batch da VALIDAÇÃO...")
    try:
        images, labels = next(iter(val_loader))

        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Image range: [{images.min():.4f}, {images.max():.4f}]")
        print(f"  Labels únicos: {labels.unique().tolist()}")
        print(f"  Esperado {num_classes} classes: {len(labels.unique()) <= num_classes}")
    except Exception as e:
        print(f"  Erro ao carregar batch de validação: {e}")
        return False

    print(f"\n{'-' * 60}")
    print(f"PIPELINE DE AUGMENTATION ({dataset_type.upper()}) TESTADO COM SUCESSO!")
    print(f"{'-' * 60}\n")

    return True

def test_minority_augmentation(
        dataset_type: str = "binary",
        architecture: str = "resnext50_32x4d",
        target_strategy: str = None,
        target_ratio: float = None
):
    if dataset_type == "binary":
        config = load_binary_config()
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared/data/splits/binary/train')
    else:
        config = load_multiclass_config()
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared/data/splits/multiclass/train')

    data_config = config['data']
    model_config = config['model']
    minority_config = data_config['minority_augmentation']

    if target_strategy is None:
        target_strategy = minority_config['strategy']
    if target_ratio is None:
        target_ratio = minority_config.get('target_ratio', 0.6)

    minority_classes = data_config['minority_classes']
    class_names = model_config['class_names']

    print("\n" + "-" * 60)
    print(f"TESTANDO AUGMENTAÇÃO DE CLASSES MINORITÁRIAS: {dataset_type.upper()}")
    print("-" * 60)
    print(f"Estratégia: {target_strategy}")
    if target_strategy == 'ratio':
        print(f"Target Ratio: {target_ratio}")
    print(f"Classes minoritárias: {minority_classes}")
    print(f"Classes: {class_names}")
    print("-" * 60 + "\n")

    if not os.path.exists(data_path):
        print(f"Dataset não encontrado: {data_path}\n")
        return False

    print("Carregando dataset...\n")
    dataset = datasets.ImageFolder(root=data_path, transform=None)

    print(f"Dataset carregado: {len(dataset)} imagens")
    print(f"  Classes: {dataset.classes}\n")

    targets = [label for _, label in dataset.samples]
    class_counts = Counter(targets)

    print("Distribuição ORIGINAL:")
    for class_idx, class_name in enumerate(dataset.classes):
        count = class_counts.get(class_idx, 0)
        percentage = (count / len(dataset) * 100) if len(dataset) > 0 else 0
        minority_marker = " (MINORITÁRIA)" if class_idx in minority_classes else ""
        print(f"  Classe {class_idx} ({class_name}): {count} ({percentage:.1f}%){minority_marker}")

    indices = list(range(len(dataset)))
    train_subset = torch.utils.data.Subset(dataset, indices)

    print(f"\n{'-' * 60}")
    print("INICIANDO AUGMENTAÇÃO")
    print(f"{'-' * 60}\n")

    try:
        augmented_subset = augment_minority_class(
            train_split=train_subset,
            target_strategy=target_strategy,
            target_ratio=target_ratio,
            architecture_name=architecture,
            minority_classes=minority_classes
        )

        print(f"\n{'-' * 60}")
        print(f"AUGMENTAÇÃO CONCLUÍDA COM SUCESSO!")
        print(f"{'-' * 60}")
        print(f"  Dataset original: {len(train_subset)} amostras")
        print(f"  Dataset augmentado: {len(augmented_subset)} amostras")
        print(f"  Amostras adicionadas: {len(augmented_subset) - len(train_subset)}")
        print(f"{'-' * 60}\n")

        return True

    except Exception as e:
        print(f"\nErro durante augmentação: {e}\n")
        return False

def test_both_datasets(
        architecture: str = "resnext50_32x4d",
        batch_size: int = 16,
        strategy: str = None
):
    print("\n" + "=" * 80)
    print(" " * 20 + "TESTE COMPLETO - BINÁRIO E MULTICLASSE")
    print("=" * 80 + "\n")

    results = {
        'binary': {'augmentation': False, 'minority': False},
        'multiclass': {'augmentation': False, 'minority': False}
    }

    results['binary']['augmentation'] = test_augmentation_pipeline(
        dataset_type="binary",
        architecture=architecture,
        batch_size=batch_size
    )

    results['binary']['minority'] = test_minority_augmentation(
        dataset_type="binary",
        architecture=architecture,
        target_strategy=strategy
    )

    results['multiclass']['augmentation'] = test_augmentation_pipeline(
        dataset_type="multiclass",
        architecture=architecture,
        batch_size=batch_size
    )

    results['multiclass']['minority'] = test_minority_augmentation(
        dataset_type="multiclass",
        architecture=architecture,
        target_strategy=strategy
    )

    print("\n" + "-" * 60)
    print(" " * 25 + "SUMÁRIO DOS TESTES")
    print("-" * 60 + "\n")

    print("DATASET BINÁRIO:")
    print(f"  Augmentation Pipeline: {'PASSOU' if results['binary']['augmentation'] else 'FALHOU'}")
    print(f"  Minority Augmentation: {'PASSOU' if results['binary']['minority'] else 'FALHOU'}")

    print("\nDATASET MULTICLASSE:")
    print(f"  Augmentation Pipeline: {'PASSOU' if results['multiclass']['augmentation'] else 'FALHOU'}")
    print(f"  Minority Augmentation: {'PASSOU' if results['multiclass']['minority'] else 'FALHOU'}")

    all_passed = all(all(v.values()) for v in results.values())

    print("\n" + "-" * 60)
    if all_passed:
        print(" " * 25 + "TODOS OS TESTES PASSARAM!")
    else:
        print(" " * 25 + "ALGUNS TESTES FALHARAM")
    print("-" * 60 + "\n")

    return all_passed

def main():
    parser = argparse.ArgumentParser(description='Test preprocessing and augmentation')

    parser.add_argument(
        '--test',
        type=str,
        choices=['preprocessor', 'augmentation', 'minority', 'binary', 'multiclass', 'all'],
        default='all',
        help='Tipo de teste a executar'
    )

    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['binary', 'multiclass', 'both'],
        default='both',
        help='Tipo de dataset a testar'
    )

    parser.add_argument(
        '--architecture',
        type=str,
        default='resnext50_32x4d',
        choices=[
            'resnext50_32x4d',
            'convnext_tiny',
            'efficientnetv2_s',
            'densenet121',
            'vit_b_16',
            'swin_v2_tiny'
        ],
        help='Arquitetura a testar'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Tamanho do batch'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        choices=['balance', 'ratio', 'proportional', None],
        help='Estratégia de balanceamento (usa config do YAML se não especificado)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" " * 20 + "TESTE DE PREPROCESSING E AUGMENTATION")
    print("=" * 80)

    if args.test == 'preprocessor':
        test_preprocessor()

    elif args.test == 'all':
        test_preprocessor()
        test_both_datasets(
            architecture=args.architecture,
            batch_size=args.batch_size,
            strategy=args.strategy
        )

    elif args.test == 'augmentation':
        if args.dataset_type == 'both':
            test_augmentation_pipeline("binary", args.architecture, args.batch_size)
            test_augmentation_pipeline("multiclass", args.architecture, args.batch_size)
        else:
            test_augmentation_pipeline(args.dataset_type, args.architecture, args.batch_size)

    elif args.test == 'minority':
        if args.dataset_type == 'both':
            test_minority_augmentation("binary", args.architecture, args.strategy)
            test_minority_augmentation("multiclass", args.architecture, args.strategy)
        else:
            test_minority_augmentation(args.dataset_type, args.architecture, args.strategy)

    elif args.test in ['binary', 'multiclass']:
        dataset_type = args.test
        test_augmentation_pipeline(dataset_type, args.architecture, args.batch_size)
        test_minority_augmentation(dataset_type, args.architecture, args.strategy)

    print("\n" + "=" * 80)
    print(" " * 30 + "TESTES CONCLUÍDOS")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()