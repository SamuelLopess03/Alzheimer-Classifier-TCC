import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training import run_training_flow
from src.evaluation import run_full_inference_pipeline
from src.utils import run_pipeline, print_banner
from scripts.prepare_data import prepare_data

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = str(BASE_DIR / 'shared/data')
DEFAULT_EXPERIMENTS_PATH = str(BASE_DIR / 'shared/logs/experiments')
DEFAULT_MODELS_PATH = str(BASE_DIR / 'shared/models')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de treinamento do modelo binário para detecção de Alzheimer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                    Exemplos de uso:
                      # Pipeline completa (todas as etapas)
                      python scripts/train_binary.py
                    
                      # Apenas treinamento
                      python scripts/train_binary.py --train
               """
    )

    parser.add_argument('--prepare-data', action='store_true', help="Executar preparação de dados")
    parser.add_argument('--train', action='store_true', help="Executar treinamento do modelo")
    parser.add_argument('--inference', action='store_true', help="Executar inferência final")
    
    parser.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--experiments-path', type=str, default=DEFAULT_EXPERIMENTS_PATH)
    parser.add_argument('--models-path', type=str, default=DEFAULT_MODELS_PATH)
    parser.add_argument('--generate-gradcam', action='store_true')
    parser.add_argument('--gradcam-samples', type=int, default=10)

    return parser.parse_args()

def main():
    args = parse_arguments()

    step_descriptions = {
        'prepare_data': 'Preparação de Dados',
        'train': 'Treinamento do Modelo Binário',
        'inference': 'Inferência Final + Grad-CAM'
    }

    step_functions = {
        'prepare_data': lambda: (prepare_data() or True),
        'train': lambda: run_training_flow('binary', args.data_path),
        'inference': lambda: (run_full_inference_pipeline(
            'binary', args.experiments_path, args.data_path, args.models_path, 
            args.generate_gradcam, args.gradcam_samples
        ) or True)
    }

    pipeline_steps = []
    if args.prepare_data: pipeline_steps.append('prepare_data')
    if args.train: pipeline_steps.append('train')
    if args.inference: pipeline_steps.append('inference')

    if not pipeline_steps:
        pipeline_steps = ['prepare_data', 'train', 'inference']

    try:
        exit_code = run_pipeline(
            pipeline_name="TREINAMENTO BINÁRIO",
            pipeline_steps=pipeline_steps,
            step_descriptions=step_descriptions,
            step_functions=step_functions
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nPipeline interrompida pelo usuário.\n")
        sys.exit(130)

if __name__ == "__main__":
    main()