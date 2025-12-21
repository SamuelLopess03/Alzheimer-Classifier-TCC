import sys
import os
from torchvision import datasets as tv_datasets
from collections import Counter

from src.training import run_grid_search
from src.utils import load_multiclass_config

def train_multiclass():
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

    base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared/data')
    train_path = os.path.join(base_path, 'splits/multiclass/train')

    if not os.path.exists(train_path):
        print(f"Erro: Dataset não encontrado em {train_path}")
        print("\nExecute primeiro o script de preparação:")
        print("   python scripts/prepare_data.py\n")
        sys.exit(1)

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

            print(f"\n  Métricas Detalhadas:")
            metrics = best_result['best_metrics']
            print(f"    Precision: {metrics.get('mean_precision', 0) * 100:.2f}%")
            print(f"    Recall: {metrics.get('mean_recall', 0) * 100:.2f}%")
            print(f"    MCC: {metrics.get('mean_mcc', 0):.4f}")

            print(f"\n  Desempenho por Classe:")
            for class_name in config['model']['class_names']:
                print(f"    {class_name}")
    else:
        print("Nenhum resultado válido obtido!")

    print("\n" + "=" * 80)
    print("TREINAMENTO DO MODELO MULTICLASSE CONCLUÍDO")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    train_multiclass()