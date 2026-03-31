import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets as tv_datasets

from ..models import create_model_with_architecture
from ..training import train_final_model, get_training_config
from .evaluator import evaluate_model
from ..data import create_stratified_holdout_split
from ..utils import (
    load_hyperparameters_config, 
    get_pytorch_device,
    find_best_experiment,
    extract_best_hyperparameters,
    print_banner,
    print_section
)
from ..visualization import print_detailed_metrics, print_class_distribution

def load_inference_datasets(data_path: str, model_type: str) -> Tuple[tv_datasets.ImageFolder, tv_datasets.ImageFolder]:
    train_path = os.path.join(data_path, f'splits/{model_type}/train')
    test_path = os.path.join(data_path, f'splits/{model_type}/test')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Datasets não encontrados em {train_path} ou {test_path}")

    train_dataset = tv_datasets.ImageFolder(root=train_path, transform=None)
    test_dataset = tv_datasets.ImageFolder(root=test_path, transform=None)
    
    return train_dataset, test_dataset

def save_inference_results(
    training_results: Dict,
    evaluation_results: Dict,
    best_experiment: Dict,
    hyperparameters: Dict,
    models_path: str
) -> Path:
    final_results = {
        'best_experiment': best_experiment,
        'hyperparameters': hyperparameters,
        'training_results': {
            'best_epoch': training_results['best_epoch'],
            'best_val_f1': training_results['best_val_f1'],
        },
        'evaluation_results': {
            'test_metrics': evaluation_results['test_metrics'],
        },
        'checkpoint_path': training_results['checkpoint_path'],
        'gradcam_path': evaluation_results.get('gradcam_path')
    }

    output_dir = Path(models_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "final_training_results.json"

    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    return results_file

def run_full_inference_pipeline(
    model_type: str,
    experiments_path: str,
    data_path: str,
    models_path: str,
    generate_gradcam: bool = False,
    gradcam_samples: int = 10
):
    print_banner("PIPELINE: TREINAMENTO FINAL + AVALIAÇÃO", f"Model Type: {model_type.upper()}")

    device = get_pytorch_device()
    is_multiclass = (model_type == 'multiclass')

    print_section("BUSCANDO MELHOR EXPERIMENTO")
    best_exp = find_best_experiment(experiments_path, model_type)
    if not best_exp:
        print(f"Nenhum experimento encontrado para '{model_type}' em {experiments_path}")
        return

    hyperparameters = extract_best_hyperparameters(best_exp)
    print(f"Melhor experimento: {best_exp['architecture_name']} (Score: {best_exp['best_score']:.4f})")

    print_section("CARREGANDO DATASETS")
    train_dataset, test_dataset = load_inference_datasets(data_path, model_type)

    config = get_training_config(is_multiclass)
    print_class_distribution(train_dataset, config['model']['class_names'])

    print_section("CONFIGURANDO MODELO E CRITÉRIOS")
    model, criterion, optimizer = create_model_with_architecture(
        hyperparams=hyperparameters,
        architecture_name=hyperparameters['architecture_name'],
        class_names=config['model']['class_names'],
        device=device,
        train_dataset=train_dataset
    )

    train_ratio = config['data']['split_ratios']['train']
    val_ratio = config['data']['split_ratios']['train_val']
    train_split, val_split = create_stratified_holdout_split(
        train_dataset, train_ratio, val_ratio, random_state=config['data']['random_seed']
    )

    print_section("EXECUTANDO TREINAMENTO FINAL")
    training_results = train_final_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_split=train_split,
        val_split=val_split,
        hyperparameters=hyperparameters,
        device=device,
        is_multiclass=is_multiclass,
        use_gradient_clipping=hyperparameters.get('use_gradient_clipping', True)
    )

    print_section("AVALIAÇÃO NO TEST SET")
    evaluation_results = evaluate_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        test_dataset=test_dataset,
        training_results=training_results,
        hyperparameters=hyperparameters,
        device=device,
        generate_gradcam=generate_gradcam,
        gradcam_samples=gradcam_samples
    )

    print_detailed_metrics(evaluation_results['test_metrics'], config['model']['class_names'])

    results_file = save_inference_results(
        training_results, evaluation_results, best_exp, hyperparameters, 
        os.path.join(models_path, model_type)
    )

    print_banner("PIPELINE CONCLUÍDA COM SUCESSO!", f"Resultados salvos em: {results_file.name}")
