import os
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.evaluation import run_full_inference_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENTS_PATH = str(BASE_DIR / 'shared/logs/experiments')
DEFAULT_DATA_PATH = str(BASE_DIR / 'shared/data')
DEFAULT_MODELS_PATH = str(BASE_DIR / 'shared/models')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treinamento final com melhores hiperparâmetros + avaliação no test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Exemplos de uso:
                  # Treinamento + avaliação binário com Grad-CAM
                  python scripts/inference.py --model_type binary --generate_gradcam
                
                # Treinamento + avaliação multiclasse com 15 amostras Grad-CAM
                  python scripts/inference.py --model_type multiclass --generate_gradcam --gradcam_samples 15
                
                # Com caminhos personalizados
                  python scripts/inference.py --model_type binary --experiments_path /custom/path
               """
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='binary',
        choices=['binary', 'multiclass'],
        help="Tipo do modelo: 'binary' ou 'multiclass' (padrão: binary)"
    )

    parser.add_argument(
        '--experiments_path',
        type=str,
        default=DEFAULT_EXPERIMENTS_PATH,
        help=f"Caminho base dos experimentos do grid search (padrão: {DEFAULT_EXPERIMENTS_PATH})"
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Caminho dos dados (padrão: {DEFAULT_DATA_PATH})"
    )

    parser.add_argument(
        '--models_path',
        type=str,
        default=DEFAULT_MODELS_PATH,
        help=f"Caminho para salvar modelos finais (padrão: {DEFAULT_MODELS_PATH})"
    )

    parser.add_argument(
        '--generate_gradcam',
        action='store_true',
        help="Gerar visualizações Grad-CAM"
    )

    parser.add_argument(
        '--gradcam_samples',
        type=int,
        default=10,
        help="Número de amostras Grad-CAM (padrão: 10)"
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    try:
        run_full_inference_pipeline(
            model_type=args.model_type,
            experiments_path=args.experiments_path,
            data_path=args.data_path,
            models_path=args.models_path,
            generate_gradcam=args.generate_gradcam,
            gradcam_samples=args.gradcam_samples
        )
    except KeyboardInterrupt:
        print("\n\nExecução interrompida pelo usuário.\n")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nErro crítico durante execução: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()