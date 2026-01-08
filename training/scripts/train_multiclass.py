import sys
import os
import argparse
import time
from typing import Optional
from torchvision import datasets as tv_datasets
from collections import Counter

from src.training import run_grid_search
from src.utils import load_multiclass_config

from .prepare_data import prepare_data
from .inference import inference

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'shared/data')
DEFAULT_EXPERIMENTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'shared/logs/experiments')
DEFAULT_MODELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'shared/models')

PIPELINE_STEPS = {
    'prepare_data': 'Preparação de Dados',
    'train': 'Treinamento do Modelo Multiclasse',
    'inference': 'Inferência Final'
}

def print_banner(title: str, subtitle: Optional[str] = None):
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    if subtitle:
        print(f"{subtitle:^80}")
    print(f"{'=' * 80}\n")

def execute_prepare_data() -> bool:
    try:
        prepare_data()

        return True
    except Exception as ex:
        print(f"Erro na preparação de dados: {str(ex)}\n")
        return False

def execute_train_multiclass() -> bool:
    try:
        config = load_multiclass_config()

        print("\n" + "=" * 80)
        print("TREINAMENTO DO MODELO MULTICLASSE: NÍVEIS DE DEMÊNCIA")
        print("=" * 80)
        print(f"\nConfiguração Carregada:")
        print(f"  Classes: {config['model']['class_names']}")
        print(f"  Número de Classes: {config['model']['num_classes']}")
        print(f"  Épocas: {config['training']['epochs']}")
        print(f"  Patience: {config['training']['patience']}")
        print(f"  Estratégia de Augmentação: {config['data']['minority_augmentation']['strategy']}")
        print(f"  Classes Minoritárias: {config['data']['minority_classes']}")
        print("=" * 80 + "\n")

        base_path = DEFAULT_DATA_PATH
        train_path = os.path.join(base_path, 'splits/multiclass/train')

        if not os.path.exists(train_path):
            print(f"Erro: Dataset não encontrado em {train_path}")
            print("\nExecute primeiro o script de preparação:")
            print("   python scripts/train_multiclass.py --prepare-data\n")
            return False

        print(f"Carregando dataset de: {train_path}\n")
        train_dataset = tv_datasets.ImageFolder(root=train_path, transform=None)

        print(f"Dataset carregado:")
        print(f"  Total de amostras: {len(train_dataset)}")
        print(f"  Classes encontradas: {train_dataset.classes}")

        expected_classes = config['model']['class_names']
        if set(train_dataset.classes) != set(expected_classes):
            print(f"\nAVISO: Classes do dataset não correspondem ao config!")
            print(f"  Esperado: {expected_classes}")
            print(f"  Encontrado: {train_dataset.classes}\n")

        labels = [label for _, label in train_dataset.samples]
        class_dist = Counter(labels)

        print(f"\n  Distribuição por classe:")
        for class_idx, class_name in enumerate(train_dataset.classes):
            count = class_dist.get(class_idx, 0)
            percentage = (count / len(train_dataset) * 100) if len(train_dataset) > 0 else 0
            print(f"    {class_name}: {count} ({percentage:.1f}%)")
        print()

        all_results = run_grid_search(
            train_dataset=train_dataset,
            model_type='multiclass'
        )

        if all_results:
            print(f"Arquiteturas testadas: {len(all_results)}\n")

            print(f"{'Arquitetura':<20} {'Score':<12} {'F1-Score':<12} {'Bal. Acc':<12}")
            print("-" * 80)

            for arch, results in all_results.items():
                if results.get('best_params'):
                    score = results.get('best_score', 0.0)
                    best_metrics = results.get('best_metrics', {})
                    f1 = best_metrics.get('mean_f1', 0.0)
                    bal_acc = best_metrics.get('mean_balanced_accuracy', 0.0)

                    print(f"{arch:<20} {score:<12.4f} {f1 * 100:<11.2f}% {bal_acc * 100:<11.2f}%")
                else:
                    print(f"{arch:<20} {'FALHOU':<12} {'-':<12} {'-':<12}")

            print("-" * 80 + "\n")

            valid_results = {k: v for k, v in all_results.items() if v.get('best_params')}

            if valid_results:
                best_arch, best_result = max(
                    valid_results.items(),
                    key=lambda x: x[1].get('best_score', 0.0)
                )

                print(f"MELHOR ARQUITETURA: {best_arch.upper()}")
                print(f"  Score Final: {best_result['best_score']:.4f}")
                print(f"  F1-Score: {best_result['best_metrics']['mean_f1'] * 100:.2f}%")
                print(f"  Balanced Accuracy: {best_result['best_metrics']['mean_balanced_accuracy'] * 100:.2f}%")

                print(f"\n  Hiperparâmetros Ótimos:")
                for param, value in best_result['best_params'].items():
                    print(f"    {param}: {value}")

                print(f"\n  Métricas Detalhadas (Médias):")
                metrics = best_result['best_metrics']
                print(f"    Precision (Weighted): {metrics.get('mean_precision', 0) * 100:.2f}%")
                print(f"    Recall (Weighted): {metrics.get('mean_recall', 0) * 100:.2f}%")
                print(f"    F1-Score (Weighted): {metrics.get('mean_f1', 0) * 100:.2f}%")
                print(f"    Precision (Macro): {metrics.get('mean_precision_macro', 0) * 100:.2f}%")
                print(f"    Recall (Macro): {metrics.get('mean_recall_macro', 0) * 100:.2f}%")
                print(f"    F1-Score (Macro): {metrics.get('mean_f1_macro', 0) * 100:.2f}%")
                print(f"    Specificity: {metrics.get('mean_specificity', 0) * 100:.2f}%")
                print(f"    MCC: {metrics.get('mean_mcc', 0):.4f}")

                print(f"\n  Desempenho por Classe:")
                class_names = config['model']['class_names']
                precision_per_class = metrics.get('precision_per_class', [])
                recall_per_class = metrics.get('recall_per_class', [])
                f1_per_class = metrics.get('f1_per_class', [])
                support_per_class = metrics.get('support_per_class', [])

                for i, class_name in enumerate(class_names):
                    print(f"\n    {class_name}:")
                    print(f"      Precision:   {precision_per_class[i] * 100:>6.2f}%")
                    print(f"      Recall:      {recall_per_class[i] * 100:>6.2f}%")
                    print(f"      F1-Score:    {f1_per_class[i] * 100:>6.2f}%")
                    print(f"      Support:     {support_per_class[i]:>6.0f} amostras")
        else:
            print("Nenhum resultado válido obtido!")
            return False

        print("\n" + "=" * 80)
        print("TREINAMENTO DO MODELO MULTICLASSE CONCLUÍDO")
        print("=" * 80 + "\n")

        return True

    except Exception as ex:
        print(f"Erro no treinamento multiclasse: {str(ex)}\n")
        import traceback
        traceback.print_exc()
        return False

def execute_inference(args) -> bool:
    try:
        inference_args = argparse.Namespace(
            model_type='multiclass',
            experiments_path=args.experiments_path,
            data_path=args.data_path,
            generate_gradcam=args.generate_gradcam,
            gradcam_samples=args.gradcam_samples
        )

        inference(
            args=inference_args,
            model_type='multiclass',
            generate_gradcam=args.generate_gradcam,
            gradcam_samples=args.gradcam_samples
        )

        return True
    except Exception as ex:
        print(f"Erro na inferência: {str(ex)}\n")
        import traceback
        traceback.print_exc()
        return False

def run_pipeline(args):
    print_banner(
        "PIPELINE DE TREINAMENTO MULTICLASSE",
        "Alzheimer Detection System - Multiclass Model"
    )

    pipeline_start_time = time.time()
    steps_executed = []

    print("Configurando pipeline...\n")

    pipeline_steps = []

    if args.prepare_data:
        pipeline_steps.append('prepare_data')
        print("Preparação de dados será executada\n")

    if args.train:
        pipeline_steps.append('train')
        print("Treinamento do modelo multiclasse será executado\n")

    if args.inference:
        pipeline_steps.append('inference')
        print("Inferência será executada\n")

    if not pipeline_steps:
        print("Nenhuma etapa específica selecionada")
        print("Executando pipeline completa (todas as etapas)\n")

        pipeline_steps = [
            'prepare_data',
            'train',
            'inference'
        ]

    total_steps = len(pipeline_steps)

    print(f"\n{'-' * 60}")
    print(f"Pipeline configurada: {total_steps} etapa(s) no total")
    print(f"{'-' * 60}\n")

    for i, step in enumerate(pipeline_steps, 1):
        print(f"  {i}. {PIPELINE_STEPS[step]}")
    print()

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

            elif step == 'train':
                success = execute_train_multiclass()

            elif step == 'inference':
                success = execute_inference(args)

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
        description="Pipeline de treinamento do modelo multiclasse para detecção de Alzheimer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                    Exemplos de uso:
                    
                      # Pipeline completa (todas as etapas)
                      python train_multiclass.py
                    
                      # Apenas preparação de dados
                      python train_multiclass.py --prepare-data
                    
                      # Apenas treinamento
                      python train_multiclass.py --train
                    
                      # Apenas inferência com Grad-CAM
                      python train_multiclass.py --inference --generate-gradcam --gradcam-samples 15
                    
                      # Pipeline completa com Grad-CAM
                      python train_multiclass.py --generate-gradcam --gradcam-samples 20
               """
    )

    pipeline_group = parser.add_argument_group('Etapas da Pipeline')
    pipeline_group.add_argument(
        '--prepare-data',
        action='store_true',
        help="Executar preparação de dados"
    )
    pipeline_group.add_argument(
        '--train',
        action='store_true',
        help="Executar treinamento do modelo multiclasse"
    )
    pipeline_group.add_argument(
        '--inference',
        action='store_true',
        help="Executar inferência final"
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