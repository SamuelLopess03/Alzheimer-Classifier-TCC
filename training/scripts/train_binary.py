import sys
import os
import argparse
from torchvision import datasets as tv_datasets

from training.src.training import run_grid_search

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train binary Alzheimer detection model'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', '..', 'shared/data'),
        help='Path to data'
    )

    parser.add_argument(
        '--save-path',
        type=str,
        default='./shared/logs/experiments',
        help='Path to save experiment results'
    )

    parser.add_argument(
        '--architectures',
        nargs='+',
        default=None,
        help='Architectures to test (default: all)'
    )

    parser.add_argument(
        '--wandb-project',
        type=str,
        default='alzheimer-detection',
        help='W&B project name'
    )

    parser.add_argument(
        '--wandb-entity',
        type=str,
        default=None,
        help='W&B entity name'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("INICIANDO A ETAPA DE TREINAMENTO DO MODELO BINÁRIO [DEMENTED E NONDEMENTED]")
    print("=" * 80 + "\n")

    train_path = os.path.join(args.data_path, 'splits/binary/train')

    if not os.path.exists(train_path):
        print(f"Erro: Dataset não encontrado em {train_path}")
        print("Execute Primeiro o Script prepare_Data para Baixar e Organizar o Dataset\n")
        sys.exit(1)

    train_dataset = tv_datasets.ImageFolder(root=train_path, transform=None)
    classes = train_dataset.classes

    print(f"Dataset carregado: {len(train_dataset)} amostras")
    print(f"Classes: {classes}\n")

    all_results = run_grid_search(
        train_dataset=train_dataset,
        classes=classes,
        architectures=args.architectures,
        save_path=args.save_path,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )

    print("\nGrid Search Concluído\n")
    print(f"   Arquiteturas testadas: {len(all_results)}")

    for arch, results in all_results.items():
        score = results.get('best_score', 0.0)
        f1 = results.get('best_metrics', {}).get('mean_f1', 0.0)
        bal_acc = results.get('best_metrics', {}).get('mean_balanced_accuracy', 0.0)

        print(f"{arch:<20} {score:<12.4f} {f1 * 100:<11.2f}% {bal_acc * 100:.2f}%")

    print("-" * 80 + "\n")

    best_arch = max(all_results.items(), key=lambda x: x[1].get('best_score', 0.0))

    print(f"\nMELHOR ARQUITETURA: {best_arch[0].upper()}")
    print(f"   Score: {best_arch[1]['best_score']:.4f}")
    print(f"   Hiperparâmetros: {best_arch[1]['best_params']}\n")

    print("\n" + "-" * 80)
    print("ETAPA DE TREINAMENTO CONCLUÍDA")
    print("-" * 80 + "\n")

if __name__ == '__main__':
    main()