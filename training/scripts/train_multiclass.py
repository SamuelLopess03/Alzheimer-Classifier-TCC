import sys
import argparse
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training import run_training_flow, run_final_training_flow
from src.evaluation import run_inference_pipeline
from src.visualization.terminal import print_banner, print_section
from scripts.prepare_data import prepare_data

BASE_DIR = Path(__file__).resolve().parent.parent
SHARED_DIR = BASE_DIR.parent / 'shared'

DEFAULT_DATA_PATH = str(SHARED_DIR / 'data')
DEFAULT_EXPERIMENTS_PATH = str(SHARED_DIR / 'logs/experiments')
DEFAULT_MODELS_PATH = str(SHARED_DIR / 'models')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de treinamento do modelo multiclasse para detecção de Alzheimer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                    Exemplos de uso:
                      # Pipeline completa (todas as etapas)
                      python scripts/train_multiclass.py
                    
                      # Apenas busca de hiperparâmetros
                      python scripts/train_multiclass.py --grid-search
                      
                      # Apenas treinamento final de produção
                      python scripts/train_multiclass.py --train-final
                      
                      # Apenas inferência
                      python scripts/train_multiclass.py --inference
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
    parser.add_argument('--no-gradcam', dest='generate_gradcam', action='store_false', help="Desabilitar geração do Grad-CAM")
    parser.set_defaults(generate_gradcam=True)
    
    parser.add_argument('--gradcam-samples', type=int, default=5, help="Número de amostras para o Grad-CAM")

    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.train:
        args.grid_search = True
        args.train_final = True

    step_descriptions = {
        'prepare_data': 'Preparação de Dados',
        'grid_search': 'Busca de Hiperparâmetros (Random Search)',
        'train_final': 'Treinamento Final de Produção',
        'inference': 'Inferência Final + Grad-CAM'
    }

    all_possible_steps = ['prepare_data', 'grid_search', 'train_final', 'inference']
    pipeline_steps = []
    
    if args.prepare_data: pipeline_steps.append('prepare_data')
    if args.grid_search: pipeline_steps.append('grid_search')
    if args.train_final: pipeline_steps.append('train_final')
    if args.inference: pipeline_steps.append('inference')

    if not pipeline_steps:
        pipeline_steps = all_possible_steps

    print_banner("PIPELINE: TREINAMENTO MULTICLASSE", "Alzheimer Detection System")
    
    pipeline_start_time = time.time()
    steps_executed = []

    print(f"Configurando pipeline: {len(pipeline_steps)} etapa(s)...\n")
    for i, step in enumerate(pipeline_steps, 1):
        print(f"  {i}. {step_descriptions[step]}")
    print()

    for i, step in enumerate(pipeline_steps, 1):
        step_name = step_descriptions[step]
        print_section(f"ETAPA {i}/{len(pipeline_steps)}: {step_name}")
        
        step_start_time = time.time()
        success = False
        
        try:
            if step == 'prepare_data':
                success = prepare_data(output_path=args.data_path)
            elif step == 'grid_search':
                success = run_training_flow('multiclass', args.data_path)
            elif step == 'train_final':
                success = run_final_training_flow('multiclass', args.experiments_path, args.data_path)
            elif step == 'inference':
                success = (run_inference_pipeline(
                    'multiclass', args.data_path, args.models_path, args.experiments_path,
                    args.generate_gradcam, args.gradcam_samples
                ) or True)
        except KeyboardInterrupt:
            print("\n\nPipeline interrompida pelo usuário.\n")
            sys.exit(130)
        except Exception as e:
            print(f"\nErro inesperado na etapa '{step_name}': {str(e)}")
            import traceback
            traceback.print_exc()
            success = False

        step_duration = time.time() - step_start_time
        steps_executed.append({'step': step_name, 'success': success})

        if not success:
            print_section("PIPELINE INTERROMPIDA POR FALHA")
            break
        
        print_section("ETAPA CONCLUÍDA COM SUCESSO")

    print_banner("RESUMO DA EXECUÇÃO")
    successful = sum(1 for s in steps_executed if s['success'])
    total_duration = time.time() - pipeline_start_time
    
    print(f"   Total planejado: {len(pipeline_steps)}")
    print(f"   Executadas:      {len(steps_executed)}")
    print(f"   Sucessos:        {successful}")
    print(f"   Tempo Total:     {total_duration:.2f}s ({total_duration / 60:.2f} min)")
    print(f"{'=' * 80}\n")

    sys.exit(0 if successful == len(pipeline_steps) else 1)

if __name__ == "__main__":
    main()