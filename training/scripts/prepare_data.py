import argparse
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import run_data_preparation_flow

def parse_args() -> argparse.Namespace:
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
        help='Path to kaggle.json file (default: ./kaggle.json)'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Base output path (default: shared/data de acordo com ambiente)'
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

    return parser.parse_args()

def prepare_data(
    kaggle_dataset: str = "ninadaithal/imagesoasis",
    output_path: str = None,
    kaggle_json: str = None,
    skip_binary: bool = False,
    skip_multiclass: bool = False,
    validate: bool = False
) -> bool:
    try:
        # Resolve caminhos padrão se não fornecidos
        base_dir = Path(__file__).resolve().parent.parent
        if output_path is None:
            output_path = str(base_dir.parent / 'shared/data')
        if kaggle_json is None:
            kaggle_json = str(base_dir / 'kaggle.json')

        success = run_data_preparation_flow(
            kaggle_dataset=kaggle_dataset,
            output_path=output_path,
            kaggle_json=kaggle_json,
            skip_binary=skip_binary,
            skip_multiclass=skip_multiclass,
            validate=validate
        )
        return success
    except KeyboardInterrupt:
        print("\n\nPreparação interrompida pelo usuário.\n")
        return False
    except Exception as e:
        print(f"\n\nErro crítico durante a preparação: {str(e)}\n")
        return False

if __name__ == "__main__":
    args = parse_args()
    success = prepare_data(
        kaggle_dataset=args.kaggle_dataset,
        output_path=args.output_path,
        kaggle_json=args.kaggle_json,
        skip_binary=args.skip_binary,
        skip_multiclass=args.skip_multiclass,
        validate=args.validate
    )
    sys.exit(0 if success else 1)