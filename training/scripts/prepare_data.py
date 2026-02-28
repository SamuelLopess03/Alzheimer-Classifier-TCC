import argparse
import shutil
import sys
import os
import time
import json

from src.data import prepare_dataset_binary, prepare_dataset_multiclass
from src.utils import download_kaggle_dataset, validate_image_files

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
        default=os.path.join(os.path.dirname(__file__), '..', 'kaggle.json'),
        help='Path to kaggle.json file (default: ~/.kaggle/kaggle.json)'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'shared/data'),
        help='Base output path (default: ./shared/data)'
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
        '--validate',
        action='store_true',
        help='Validate image integrity during preparation'
    )

    args, _ = parser.parse_known_args()
    return args

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

    if all_valid:
        print("TODOS OS DATASETS SÃO VÁLIDOS")
    else:
        print("ALGUNS DATASETS TÊM PROBLEMAS")

    return all_valid

def prepare_data():
    args = parse_args()

    print("\n" + "=" * 60)
    print("COMEÇANDO ETAPA DE PREPARAÇÃO DOS DATASETS")
    print("=" * 60 + "\n")

    print("Configuração:")
    print(f"   Dataset Kaggle: {args.kaggle_dataset}")
    print(f"   Output Path: {args.output_path}\n")

    if args.verify:
        all_valid = verify_datasets(args.output_path)
        
        if args.validate:
            print(f"\n{'-' * 60}")
            print("VALIDANDO INTEGRIDADE DOS ARQUIVOS (OPCIONAL)")
            print(f"{'-' * 60}")
            v, c, err = validate_image_files(args.output_path)
            print(f"Resultado: {v} imagens válidas, {c} corrompidas.")
            if err:
                print("\nArquivos corrompidos detectados:")
                for e in err[:10]: print(f"  - {e}")
                if len(err) > 10: print(f"  ... e mais {len(err)-10}")

        return

    splits_paste = os.path.join(args.output_path, 'splits')
    if os.path.exists(os.path.join(splits_paste, 'multiclass')):
        print(f"Dados do Dataset já Foram Baixados e Preparados.\n")
        return

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'raw'), exist_ok=True)
    os.makedirs(splits_paste, exist_ok=True)

    success, classes = download_kaggle_dataset(
        dataset_name=args.kaggle_dataset,
        output_dir=os.path.join(args.output_path, 'raw'),
        kaggle_json_path=args.kaggle_json
    )

    if not success or not classes:
        print("\nErro: Dataset não foi baixado corretamente.\n")
        sys.exit(1)

    results = {}

    if not args.skip_binary:
        print("\n" + "-" * 60)
        print("ETAPA 1: PREPARAÇÃO DO DATASET BINÁRIO")
        print("-" * 60 + "\n")

        try:
            binary_train, binary_test, binary_classes = prepare_dataset_binary(output_base_path=args.output_path)

            results['binary'] = (binary_train, binary_test, binary_classes)

        except Exception as e:
            print(f"\nErro ao preparar dataset binário: {e}")
            sys.exit(1)

    if not args.skip_multiclass:
        print("\n" + "-" * 60)
        print("ETAPA 2: PREPARAÇÃO DO DATASET MULTICLASSE")
        print("-" * 60 + "\n")

        try:
            multi_train, multi_test, multi_classes = prepare_dataset_multiclass(output_base_path=args.output_path)

            results['multiclass'] = (multi_train, multi_test, multi_classes)

        except Exception as e:
            print(f"\nErro ao preparar dataset multiclasse: {e}")
            sys.exit(1)

    try:
        shutil.rmtree(os.path.join(args.output_path, 'raw'))
        print(f"\nPasta base removida: {os.path.join(args.output_path, 'raw')}")
    except Exception as e:
        print(f"\nErro ao remover pasta base: {e}")

    print("\n" + "-" * 60)
    print("ETAPA 3: VERIFICAÇÃO FINAL")
    print("-" * 60 + "\n")

    all_valid = verify_datasets(args.output_path)

    print("\n" + "-" * 60)
    print("ETAPA 4: GERAÇÃO DE METADADOS")
    print("-" * 60 + "\n")

    metadata = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'kaggle_dataset': args.kaggle_dataset,
        'splits': {}
    }

    for split_name, split_path in [('binary', os.path.join(args.output_path, 'splits/binary')), 
                                   ('multiclass', os.path.join(args.output_path, 'splits/multiclass'))]:
        if os.path.exists(split_path):
            metadata['splits'][split_name] = {}
            for phase in ['train', 'test']:
                phase_path = os.path.join(split_path, phase)
                if os.path.exists(phase_path):
                    metadata['splits'][split_name][phase] = {}
                    for class_name in os.listdir(phase_path):
                        class_path = os.path.join(phase_path, class_name)
                        if os.path.isdir(class_path):
                            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))])
                            metadata['splits'][split_name][phase][class_name] = count

    metadata_file = os.path.join(args.output_path, 'split_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadados gerados em: {metadata_file}\n")

    print(f"\n{'-' * 60}")
    if all_valid:
        print("\nPREPARAÇÃO DOS DATASETS CONCLUÍDA COM SUCESSO!")
    else:
        print("\nPREPARAÇÃO CONCLUÍDA COM AVISOS")
        print("Revise os problemas acima antes de prosseguir.\n")
        sys.exit(1)
    print(f"{'-' * 60}\n")

if __name__ == '__main__':
    prepare_data()