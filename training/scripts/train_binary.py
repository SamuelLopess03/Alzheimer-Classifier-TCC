import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training import run_training_flow, run_final_training_flow
from src.evaluation import run_inference_pipeline
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
                    
                      # Apenas busca de hiperparâmetros
                      python scripts/train_binary.py --grid-search
                      
                      # Apenas treinamento final de produção
                      python scripts/train_binary.py --train-final
                      
                      # Apenas inferência
                      python scripts/train_binary.py --inference
               """
    )

    parser.add_argument('--prepare-data', action='store_true', help="Executar preparação de dados")
    parser.add_argument('--grid-search', action='store_true', help="Executar busca de hiperparâmetros")
    parser.add_argument('--train-final', action='store_true', help="Executar treinamento final de produção")
    parser.add_argument('--train', action='store_true', help="Atalho para --grid-search + --train-final")
    parser.add_argument('--inference', action='store_true', help="Executar inferência final e relatórios")
    
    parser.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--experiments-path', type=str, default=DEFAULT_EXPERIMENTS_PATH)
    parser.add_argument('--models-path', type=str, default=DEFAULT_MODELS_PATH)
    parser.add_argument('--generate-gradcam', action='store_true')
    parser.add_argument('--gradcam-samples', type=int, default=10)

    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.train:
        args.grid_search = True
        args.train_final = True

    step_descriptions = {
        'prepare_data': 'Preparação de Dados',
        'grid_search': 'Busca de Hiperparâmetros (Grid Search)',
        'train_final': 'Treinamento Final de Produção',
        'inference': 'Inferência Final + Grad-CAM'
    }

    step_functions = {
        'prepare_data': lambda: (prepare_data() or True),
        'grid_search': lambda: run_training_flow('binary', args.data_path),
        'train_final': lambda: run_final_training_flow('binary', args.experiments_path, args.data_path),
        'inference': lambda: (run_inference_pipeline(
            'binary', args.data_path, args.models_path, args.experiments_path,
            args.generate_gradcam, args.gradcam_samples
        ) or True)
    }

    pipeline_steps = []
    if args.prepare_data: pipeline_steps.append('prepare_data')
    if args.grid_search: pipeline_steps.append('grid_search')
    if args.train_final: pipeline_steps.append('train_final')
    if args.inference: pipeline_steps.append('inference')

    if not pipeline_steps:
        pipeline_steps = ['prepare_data', 'grid_search', 'train_final', 'inference']

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