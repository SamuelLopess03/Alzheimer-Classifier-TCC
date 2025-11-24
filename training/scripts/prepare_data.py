import argparse
import sys
import os

from training.src.data import prepare_dataset_binary, prepare_dataset_multiclass
from training.src.utils import download_kaggle_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare datasets for Alzheimer Detection'
    )

    parser.add_argument(
        '--kaggle-dataset',
        type=str,
        default='ninadaithal/imagesoasis',
        help='Kaggle dataset name (default: ninadaithal/imagesoasis)'
    )

    parser.add_argument(
        '--kaggle-json',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', '..', 'kaggle.json'),
        help='Path to kaggle.json file (default: ~/.kaggle/kaggle.json)'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', '..', 'shared/data'),
        help='Base output path (default: ./shared/data)'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Train split ratio (default: 0.8)'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--skip-binary',
        action='store_true',
        help='Skip binary dataset creation'
    )

    parser.add_argument(
        '--skip-multiclass',
        action='store_true',
        help='Skip multiclass dataset creation'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing datasets'
    )

    return parser.parse_args()

def verify_datasets(output_path: str):
    print(f"\n{'-' * 60}")
    print("VERIFICANDO INTEGRIDADE DOS DATASETS")
    print(f"{'-' * 60}\n")

    datasets_to_verify = [
        ('Binary Train', os.path.join(output_path, 'splits/binary/train')),
        ('Binary Test', os.path.join(output_path, 'splits/binary/test')),
        ('Multiclass Train', os.path.join(output_path, 'splits/multiclass/train')),
        ('Multiclass Test', os.path.join(output_path, 'splits/multiclass/test')),
    ]

    all_valid = True

    for dataset_name, dataset_path in datasets_to_verify:
        if not os.path.exists(dataset_path):
            print(f"{dataset_name}: NOT FOUND")
            all_valid = False
            continue

    print(f"\n{'-' * 60}")
    if all_valid:
        print("TODOS OS DATASETS SÃO VÁLIDOS")
    else:
        print("ALGUNS DATASETS TÊM PROBLEMAS")
    print(f"{'-' * 60}\n")

    return all_valid

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("COMEÇANDO ETAPA DE PREPARAÇÃO DOS DATASETS")
    print("=" * 60 + "\n")

    print("Configuração:")
    print(f"   Dataset Kaggle: {args.kaggle_dataset}")
    print(f"   Output Path: {args.output_path}")
    print(f"   Train Ratio: {args.train_ratio}")
    print(f"   Val Ratio: {args.val_ratio}")
    print(f"   Random State: {args.random_state}\n")

    if args.verify:
        verify_datasets(args.output_path)
        return

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'splits'), exist_ok=True)

    success, classes = download_kaggle_dataset(
        dataset_name=args.kaggle_dataset,
        output_dir=os.path.join(args.output_path, 'raw'),
        kaggle_json_path=args.kaggle_json
    )

    if not success or not classes:
        print("\nErro: Dataset não foi baixado corretamente.\n")
        sys.exit(1)

    results = {}

    non_demented_folder = next((c for c in classes if "non" in c.lower()), classes[0])
    demented_classes = [c for c in classes if c != non_demented_folder]

    print(f"\nMapeamento de classes:")
    print(f"  Sem demência: {non_demented_folder}")
    print(f"  Com demência: {demented_classes}\n")

    if not args.skip_binary:
        print("\n" + "-" * 60)
        print("ETAPA 1: PREPARAÇÃO DO DATASET BINÁRIO")
        print("-" * 60 + "\n")

        try:
            binary_train, binary_test, binary_classes = prepare_dataset_binary(
                output_base_path=args.output_path,
                binary_classes={"NonDemented": non_demented_folder, "Demented": demented_classes},
                train_ratio=args.train_ratio,
                random_state=args.random_state
            )

            results['binary'] = (binary_train, binary_test, binary_classes)

        except Exception as e:
            print(f"\nErro ao preparar dataset binário: {e}")
            sys.exit(1)

    if not args.skip_multiclass:
        print("\n" + "-" * 60)
        print("ETAPA 2: PREPARAÇÃO DO DATASET MULTICLASSE")
        print("-" * 60 + "\n")

        try:
            multi_train, multi_test, multi_classes = prepare_dataset_multiclass(
                output_base_path=args.output_path,
                demented_classes=demented_classes,
                train_ratio=args.train_ratio,
                random_state=args.random_state
            )

            results['multiclass'] = (multi_train, multi_test, multi_classes)

        except Exception as e:
            print(f"\nErro ao preparar dataset multiclasse: {e}")
            sys.exit(1)

    print("\n" + "-" * 60)
    print("ETAPA 3: VERIFICAÇÃO FINAL")
    print("-" * 60 + "\n")

    all_valid = verify_datasets(args.output_path)

    if all_valid:
        print("\nPREPARAÇÃO CONCLUÍDA COM SUCESSO!")
        print("\nPróximos passos:")
        print("  1. Execute o treinamento do modelo binário:")
        print("     python scripts/train_binary.py")
        print("  2. Execute o treinamento do modelo multiclasse:")
        print("     python scripts/train_multiclass.py\n")
    else:
        print("\nPREPARAÇÃO CONCLUÍDA COM AVISOS")
        print("Revise os problemas acima antes de prosseguir.\n")
        sys.exit(1)

if __name__ == '__main__':
    main()