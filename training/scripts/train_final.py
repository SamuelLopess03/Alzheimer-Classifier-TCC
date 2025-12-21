import os
import sys
import argparse
import time
from typing import Optional

from prepare_data import prepare_data
from train_binary import train_binary
from train_multiclass import train_multiclass
from inference import inference

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'shared', 'data')
DEFAULT_EXPERIMENTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'shared', 'logs', 'experiments')
DEFAULT_MODELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'shared', 'models')

PIPELINE_STEPS = {
    'prepare_data': 'Preparação de Dados',
    'train_binary': 'Treinamento Binário',
    'train_multiclass': 'Treinamento Multiclasse',
    'inference_binary': 'Inferência Final Binário',
    'inference_multiclass': 'Inferência Final Multiclasse'
}

def print_banner(title: str, subtitle: Optional[str] = None):
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    if subtitle:
        print(f"{subtitle:^80}")
    print(f"{'=' * 80}\n")

def print_step_header(step: str, step_number: int, total_steps: int):
    step_name = PIPELINE_STEPS.get(step, step)
    print(f"\n{'-' * 60}")
    print(f"ETAPA {step_number}/{total_steps}: {step_name}")
    print(f"{'-' * 60}\n")

def execute_prepare_data() -> bool:
    try:
        prepare_data()
        return True
    except Exception as ex:
        print(f"Erro na preparação de dados: {str(ex)}\n")
        return False

def execute_train_binary() -> bool:
    try:
        train_binary()
        return True
    except Exception as ex:
        print(f"Erro no treinamento binário: {str(ex)}\n")
        return False

def execute_train_multiclass() -> bool:
    try:
        train_multiclass()
        return True
    except Exception as ex:
        print(f"Erro no treinamento multiclasse: {str(ex)}\n")
        return False

def execute_inference(args, model_type: str) -> bool:
    try:
        inference_args = argparse.Namespace(
            model_type=model_type,
            experiments_path=args.experiments_path,
            data_path=args.data_path,
            generate_gradcam=args.generate_gradcam,
            gradcam_samples=args.gradcam_samples
        )

        inference(
            args=inference_args,
            model_type=model_type,
            generate_gradcam=args.generate_gradcam,
            gradcam_samples=args.gradcam_samples
        )

        return True
    except Exception as ex:
        print(f"Erro na inferência {model_type}: {str(ex)}\n")
        return False

def run_pipeline(args):
    print_banner(
        "PIPELINE DE TREINAMENTO E INFERÊNCIA",
        "Alzheimer Detection System"
    )

    pipeline_start_time = time.time()
    steps_executed = []

    print("Configurando pipeline...\n")

    pipeline_steps = []

    if args.prepare_data:
        pipeline_steps.append('prepare_data')
        print("Preparação de dados será executada\n")

    if args.train_binary:
        pipeline_steps.append('train_binary')
        print("Treinamento binário será executado\n")

    if args.train_multiclass:
        pipeline_steps.append('train_multiclass')
        print("Treinamento multiclasse será executado\n")

    if args.inference_binary:
        pipeline_steps.append('inference_binary')
        print("Inferência binária será executada\n")

    if args.inference_multiclass:
        pipeline_steps.append('inference_multiclass')
        print("Inferência multiclasse será executada\n")

    if not pipeline_steps:
        print("Nenhuma etapa específica selecionada")
        print("Executando pipeline completa (todas as etapas)\n")

        pipeline_steps = [
            'prepare_data',
            'train_binary',
            'train_multiclass',
            'inference_binary',
            'inference_multiclass'
        ]

    total_steps = len(pipeline_steps)

    print(f"\n{'-' * 60}")
    print(f"Pipeline configurada: {total_steps} etapa(s) no total")
    print(f"{'-' * 60}\n")

    for i, step in enumerate(pipeline_steps, 1):
        print(f"  {i}. {PIPELINE_STEPS[step]}")
    print()

    input("Pressione ENTER para iniciar a pipeline...")

    print_banner("INICIANDO EXECUÇÃO DA PIPELINE")

    for i, step in enumerate(pipeline_steps, 1):
        step_name = PIPELINE_STEPS.get(step, step)
        print(f"\n{'-' * 60}")
        print(f"ETAPA {i}/{total_steps}: {step_name}")
        print(f"{'-' * 60}\n")

        step_start_time = time.time()
        success = False
        error_msg = None

        print(f"Executando: {PIPELINE_STEPS[step]}...\n")

        try:
            if step == 'prepare_data':
                success = execute_prepare_data()

            elif step == 'train_binary':
                success = execute_train_binary()

            elif step == 'train_multiclass':
                success = execute_train_multiclass()

            elif step == 'inference_binary':
                success = execute_inference(args, 'binary')

            elif step == 'inference_multiclass':
                success = execute_inference(args, 'multiclass')

        except KeyboardInterrupt:
            raise

        except Exception as ex:
            success = False
            error_msg = str(ex)
            print(f"\nErro inesperado em '{PIPELINE_STEPS[step]}':")
            print(f"   {error_msg}\n")

            import traceback
            print("Traceback completo:")
            traceback.print_exc()

        step_duration = time.time() - step_start_time

        steps_executed.append({
            'step': PIPELINE_STEPS[step],
            'step_key': step,
            'success': success,
            'duration': step_duration,
            'error': error_msg
        })

        if not success:
            print(f"{'-' * 60}")
            print(f"PIPELINE INTERROMPIDA")
            print(f"Falha na etapa: {PIPELINE_STEPS[step]}")
            print(f"{'-' * 60}\n")
            break
        else:
            print(f"{'-' * 60}")
            print(f"ETAPA CONCLUÍDA COM SUCESSO")
            print(f"{'-' * 60}\n")

    total_duration = time.time() - pipeline_start_time

    print_banner("RESUMO DA EXECUÇÃO DA PIPELINE")

    successful = sum(1 for s in steps_executed if s['success'])
    failed = len(steps_executed) - successful

    print(f"ESTATÍSTICAS GERAIS:")
    print(f"{'-' * 60}")
    print(f"   Total de etapas planejadas: {total_steps}")
    print(f"   Etapas executadas: {len(steps_executed)}")
    print(f"   Sucessos: {successful}")
    print(f"   Falhas: {failed}")
    print(f"   Tempo total: {total_duration:.2f}s ({total_duration / 60:.2f} min)")
    print(f"{'-' * 60}\n")

    if failed == 0 and len(steps_executed) == total_steps:
        print_banner("PIPELINE CONCLUÍDA COM SUCESSO!")
        print("Todas as etapas foram executadas sem erros!\n")
        return 0
    elif failed > 0:
        print_banner("PIPELINE CONCLUÍDA COM FALHAS")
        print(f"{failed} etapa(s) falharam durante a execução\n")
        return 1
    else:
        print_banner("PIPELINE INCOMPLETA")
        print(f"Apenas {len(steps_executed)}/{total_steps} etapas foram executadas\n")
        return 1

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline completa de treinamento e inferência para detecção de Alzheimer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Exemplos de uso:
                
                  # Pipeline completa (todas as etapas)
                  python train_final.py
                
                  # Apenas preparação de dados
                  python train_final.py --prepare-data
                
                  # Treinar ambos os modelos
                  python train_final.py --train-binary --train-multiclass
                
                  # Apenas inferência do modelo binário com Grad-CAM
                  python train_final.py --inference-binary --generate-gradcam --gradcam-samples 15
                
                  # Pipeline completa com Grad-CAM
                  python train_final.py --generate-gradcam --gradcam-samples 20
                
                  # Pipeline sem parar em erros
                  python train_final.py --no-stop-on-error
                
                Dica: Se nenhuma etapa específica for selecionada, todas as etapas serão executadas.
               """
    )
    pipeline_group = parser.add_argument_group('Etapas da Pipeline')
    pipeline_group.add_argument(
        '--prepare-data',
        action='store_true',
        help="Executar preparação de dados"
    )
    pipeline_group.add_argument(
        '--train-binary',
        action='store_true',
        help="Executar treinamento binário"
    )
    pipeline_group.add_argument(
        '--train-multiclass',
        action='store_true',
        help="Executar treinamento multiclasse"
    )
    pipeline_group.add_argument(
        '--inference-binary',
        action='store_true',
        help="Executar inferência final do modelo binário"
    )
    pipeline_group.add_argument(
        '--inference-multiclass',
        action='store_true',
        help="Executar inferência final do modelo multiclasse"
    )

    paths_group = parser.add_argument_group('Caminhos')
    paths_group.add_argument(
        '--data-path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Caminho dos dados (padrão: {DEFAULT_DATA_PATH})"
    )
    paths_group.add_argument(
        '--experiments-path',
        type=str,
        default=DEFAULT_EXPERIMENTS_PATH,
        help=f"Caminho dos experimentos (padrão: {DEFAULT_EXPERIMENTS_PATH})"
    )
    paths_group.add_argument(
        '--output-path',
        type=str,
        default=DEFAULT_MODELS_PATH,
        help=f"Caminho de saída (padrão: {DEFAULT_MODELS_PATH})"
    )

    gradcam_group = parser.add_argument_group('Grad-CAM')
    gradcam_group.add_argument(
        '--generate-gradcam',
        action='store_true',
        help="Gerar visualizações Grad-CAM na inferência"
    )
    gradcam_group.add_argument(
        '--gradcam-samples',
        type=int,
        default=10,
        help="Número de amostras Grad-CAM (padrão: 10)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    try:
        exit_code = run_pipeline(args)
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\nPipeline interrompida pelo usuário.\n")
        sys.exit(130)

    except Exception as e:
        print(f"\n\nErro crítico na pipeline: {str(e)}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)