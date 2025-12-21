import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets as tv_datasets

from src.models import create_model_with_architecture
from src.training import train_final_model, get_training_config
from src.utils import load_hyperparameters_config, create_stratified_holdout_split

DEFAULT_EXPERIMENTS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'shared/logs/experiments')
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'shared/data')
DEFAULT_MODELS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'shared/models')

EXECUTION_STATE_FILE = 'execution_state.json'

def find_best_experiment(experiments_path: str, model_type: str) -> Optional[Dict]:
    experiments_dir = Path(experiments_path)

    if not experiments_dir.exists():
        print(f"Diretório de experimentos não encontrado: {experiments_path}\n")
        return None

    best_experiment = None
    best_score_final = -1.0

    pattern = f"*_{model_type}"

    print(f"\n{'-' * 60}")
    print(f"BUSCANDO MELHOR EXPERIMENTO ({model_type.upper()})")
    print(f"{'-' * 60}\n")
    print(f"Vasculhando: {experiments_dir}\n")

    matching_dirs = list(experiments_dir.glob(pattern))

    if not matching_dirs:
        print(f"Nenhum experimento encontrado para o modelo '{model_type}'\n")
        return None

    print(f"Encontrados {len(matching_dirs)} experimentos do tipo '{model_type}':\n")

    for exp_dir in matching_dirs:
        state_file = exp_dir / EXECUTION_STATE_FILE

        if not state_file.exists():
            print(f"Pulando {exp_dir.name}: {EXECUTION_STATE_FILE} não encontrado\n")
            continue

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            results = state.get('results', {})

            if not results:
                print(f"Pulando {exp_dir.name}: sem resultados\n")
                continue

            best_score = results.get('best_score', -1.0)
            architecture_name = exp_dir.name.replace(f"_{model_type}", "")

            print(f"{architecture_name:35s} | Best Score: {best_score:.4f}\n")

            if best_score > best_score_final:
                best_score_final = best_score
                best_experiment = {
                    'architecture_name': architecture_name,
                    'experiment_dir': str(exp_dir),
                    'state_file': str(state_file),
                    'best_score': best_score,
                    'results': results,
                    'executed_indices': state.get('executed_indices', []),
                    'model_type': model_type
                }

        except Exception as e:
            print(f"Erro ao processar {exp_dir.name}: {str(e)}\n")
            continue

    if best_experiment:
        print(f"\n{'-' * 60}")
        print(f"MELHOR EXPERIMENTO ENCONTRADO:")
        print(f"{'-' * 60}")
        print(f"   Arquitetura: {best_experiment['architecture_name']}")
        print(f"   Best Score: {best_experiment['best_score']:.4f}")
        print(f"   Caminho: {best_experiment['experiment_dir']}")
        print(f"{'-' * 60}\n")
    else:
        print(f"\nNenhum experimento válido encontrado para '{model_type}'\n")

    return best_experiment

def extract_best_hyperparameters(experiment: Dict) -> Dict:
    results = experiment['results']

    hyperparameters = results.get('best_params', {})

    hyperparameters['architecture_name'] = experiment['architecture_name']
    hyperparameters['model_type'] = experiment['model_type']
    hyperparameters['best_score'] = experiment['best_score']

    return hyperparameters

def setup_device(hyperparams_config: Dict) -> torch.device:
    hardware_config = hyperparams_config.get('hardware', {})
    device_config = hardware_config.get('device', 'cuda')

    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)

    print(f"Dispositivo: {device}\n")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("CUDA não disponível. Usando CPU.\n")

    return device

def load_datasets(data_path: str, model_type: str) -> Tuple[tv_datasets.ImageFolder, tv_datasets.ImageFolder]:
    train_path = os.path.join(data_path, f'splits/{model_type}/train')
    test_path = os.path.join(data_path, f'splits/{model_type}/test')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Caminho de treino não encontrado: {train_path}\n")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Caminho de teste não encontrado: {test_path}\n")

    print(f"Carregando datasets...")
    train_dataset = tv_datasets.ImageFolder(root=train_path, transform=None)
    test_dataset = tv_datasets.ImageFolder(root=test_path, transform=None)

    print(f"Train dataset: {len(train_dataset)} amostras")
    print(f"Test dataset: {len(test_dataset)} amostras\n")

    return train_dataset, test_dataset

def setup_model_criterion_and_optimizer(
        hyperparameters: Dict,
        device: torch.device,
        train_dataset: torch.utils.data.Dataset,
        is_multiclass: bool
) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
    config = get_training_config(is_multiclass)
    class_names = config['model']['class_names']

    print("Criando modelo, critério e otimizador...\n")

    model, criterion, optimizer = create_model_with_architecture(
        hyperparams=hyperparameters,
        architecture_name=hyperparameters['architecture_name'],
        class_names=class_names,
        device=device,
        train_dataset=train_dataset
    )

    print(f"Modelo: {hyperparameters['architecture_name']}")
    print(f"Critério: {criterion.__class__.__name__}")
    print(f"Otimizador: {optimizer.__class__.__name__}\n")

    return model, criterion, optimizer

def create_train_val_splits(
        train_dataset: torch.utils.data.Dataset,
        is_multiclass: bool
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    config = get_training_config(is_multiclass)
    data_config = config['data']

    train_ratio = data_config['split_ratios']['train']
    val_ratio = data_config['split_ratios']['val']
    random_seed = data_config['random_seed']

    print("Criando splits estratificados...\n")

    train_split, val_split = create_stratified_holdout_split(
        train_dataset,
        train_ratio,
        val_ratio,
        random_state=random_seed
    )

    print(f"Train split: {len(train_split)} amostras")
    print(f"Val split: {len(val_split)} amostras\n")

    return train_split, val_split

def save_final_results(
        results: Dict,
        best_experiment: Dict,
        hyperparameters: Dict,
        models_path: str
) -> Path:
    final_results = {
        'best_experiment': best_experiment,
        'hyperparameters': hyperparameters,
        'final_results': {
            'best_epoch': results['best_epoch'],
            'best_val_f1': results['best_val_f1'],
            'test_metrics': results['test_metrics']
        },
        'checkpoint_path': results['checkpoint_path'],
        'gradcam_path': results.get('gradcam_path')
    }

    output_dir = Path(models_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"final_training_results.json"

    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    return results_file

def print_hyperparameters(hyperparameters: Dict):
    print("\nHIPERPARÂMETROS SELECIONADOS:")
    print("-" * 60)
    for key, value in sorted(hyperparameters.items()):
        print(f"   {key:30s}: {value}")
    print("-" * 60 + "\n")

def inference(args, model_type: str = 'binary',
              generate_gradcam: bool = True, gradcam_samples: int = 10):
    print(f"\n{'=' * 80}")
    print("TREINAMENTO E INFERÊNCIA FINAL DO MODELO")
    print(f"{'=' * 80}\n")

    hyperparams_config = load_hyperparameters_config()

    device = setup_device(hyperparams_config)

    is_multiclass = args.model_type == 'multiclass' if args.model_type else model_type == 'multiclass'

    best_experiment = find_best_experiment(
        args.experiments_path if args.experiments_path else DEFAULT_MODELS_PATH,
        args.model_type if args.model_type else model_type
    )

    if best_experiment is None:
        print("Nenhum experimento encontrado. Encerrando.\n")
        return

    hyperparameters = extract_best_hyperparameters(best_experiment)
    print_hyperparameters(hyperparameters)

    try:
        train_dataset, test_dataset = load_datasets(
            args.data_path if args.data_path else DEFAULT_DATA_PATH,
            args.model_type if args.model_type else model_type
        )
    except FileNotFoundError as excep:
        print(str(excep))
        return

    model, criterion, optimizer = setup_model_criterion_and_optimizer(
        hyperparameters=hyperparameters,
        device=device,
        train_dataset=train_dataset,
        is_multiclass=is_multiclass
    )

    train_split, val_split = create_train_val_splits(
        train_dataset=train_dataset,
        is_multiclass=is_multiclass
    )

    print(f"{'-' * 60}")
    print("INICIANDO TREINAMENTO FINAL")
    print(f"{'-' * 60}\n")

    results = train_final_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_split=train_split,
        val_split=val_split,
        test_split=test_dataset,
        hyperparameters=hyperparameters,
        device=device,
        is_multiclass=is_multiclass,
        use_gradient_clipping=hyperparameters.get('use_gradient_clipping', True),
        max_grad_norm=hyperparameters.get('max_grad_norm', 1.0),
        generate_gradcam=args.generate_gradcam if args.generate_gradcam else generate_gradcam,
        gradcam_samples=args.gradcam_samples if args.gradcam_samples else gradcam_samples
    )

    results_file = save_final_results(
        results=results,
        best_experiment=best_experiment,
        hyperparameters=hyperparameters,
        models_path=os.path.join(DEFAULT_MODELS_PATH, model_type)
    )

    print(f"\nResultados finais salvos em: {results_file}")

    print(f"\n{'=' * 80}")
    print("TREINAMENTO E TESTE FINAL CONCLUÍDO COM SUCESSO!")
    print(f"{'=' * 80}\n")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treinamento final com melhores hiperparâmetros do grid search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Exemplos de uso:
                  # Treinamento binário com Grad-CAM
                  python train_final.py --model_type binary --generate_gradcam
                
                  # Treinamento multiclasse com 15 amostras Grad-CAM
                  python train_final.py --model_type multiclass --generate_gradcam --gradcam_samples 15
                
                  # Com caminhos personalizados
                  python train_final.py --model_type binary --experiments_path /custom/path --data_path /data/path
               """
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='binary',
        choices=['binary', 'multiclass'],
        help="Tipo do modelo: 'binary' ou 'multiclass' (padrão: multiclass)"
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
        '--num_workers',
        type=int,
        default=4,
        help="Número de workers para DataLoader (padrão: 4)"
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

if __name__ == "__main__":
    args = parse_arguments()

    try:
        inference(args)
    except KeyboardInterrupt:
        print("\n\nExecução interrompida pelo usuário.\n")
    except Exception as e:
        print(f"\n\nErro durante execução: {str(e)}\n")
        raise