import os
import json
import torch
import glob
from pathlib import Path
from typing import Dict
from torchvision import datasets as tv_datasets

from ..models.architectures import create_model_with_architecture
from .evaluator import evaluate_model, evaluate_ensemble
from src.utils.hardware import get_pytorch_device
from src.visualization.terminal import print_banner, print_section, print_detailed_metrics
from src.utils.config import (
    load_binary_config,
    load_multiclass_config,
    load_hyperparameters_config,
    MODELS_PATH
)
from src.utils.experiments import find_best_experiment, extract_best_hyperparameters
from ..visualization.wandb_logger import init_wandb_run, finish_wandb_run

def load_test_dataset(data_path: str, model_type: str) -> tv_datasets.ImageFolder:
    test_path = os.path.join(data_path, f'splits/{model_type}/test')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset de teste não encontrado em {test_path}")
    
    dataset = tv_datasets.ImageFolder(root=test_path, transform=None)
    
    config = load_multiclass_config() if model_type == 'multiclass' else load_binary_config()
    class_names = config['model']['class_names']
    
    dataset.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    dataset.classes = class_names
    dataset.samples = [
        (path, dataset.class_to_idx[os.path.basename(os.path.dirname(path))])
        for path, _ in dataset.samples
    ]
    dataset.targets = [label for _, label in dataset.samples]
    
    return dataset

def save_inference_results(evaluation_results: Dict, models_path: str, model_type: str) -> Path:
    test_metrics = evaluation_results['test_metrics'].copy()
    
    for key in ['fold', 'epoch', 'val_loss']:
        test_metrics.pop(key, None)

    final_results = {
        'model_type': model_type,
        'test_metrics': test_metrics,
        'gradcam_path': evaluation_results.get('gradcam_path')
    }
    output_dir = Path(models_path) / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "final_inference_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    return results_file

def _open_wandb_inference_run(config: Dict, architecture_name: str, model_type: str, hyperparams: Dict, models_path: str):
    wandb_cfg = config.get('logging', {}).get('wandb', {})
    if not wandb_cfg.get('enabled', False):
        return False

    init_wandb_run(
        project_name=wandb_cfg['project'],
        run_name=f"{architecture_name}_{model_type}_inference",
        config={
            "architecture": architecture_name,
            "model_type": model_type,
            "phase": "inference",
            **hyperparams
        },
        entity=wandb_cfg.get('entity'),
        tags=["inference", architecture_name, model_type],
        group=f"kfold_inference/{model_type}",    # Pasta dedicada no WandB
        directory=os.path.join(models_path, 'wandb_logs')
    )
    return True

def run_inference_pipeline(
    model_type: str,
    data_path: str,
    models_path: str,
    experiments_path: str,
    generate_gradcam: bool = False,
    gradcam_samples: int = 10
):
    print_banner("PIPELINE: AVALIAÇÃO E INFERÊNCIA", f"Model Type: {model_type.upper()}")

    device = get_pytorch_device()
    is_multiclass = (model_type == 'multiclass')
    config = load_multiclass_config() if is_multiclass else load_binary_config()
    class_names = config['model']['class_names']

    print_section("CONFIGURANDO MODELO FINAL")
    best_exp = find_best_experiment(experiments_path, model_type)
    if not best_exp:
        print(f"Erro: Nenhum experimento encontrado para '{model_type}' em {experiments_path}")
        return False
        
    hyperparams = extract_best_hyperparameters(best_exp)
    architecture_name = hyperparams['architecture_name']

    print_section("CARREGANDO DATASET DE TESTE")
    test_dataset = load_test_dataset(data_path, model_type)
    print(f"Dataset de teste carregado: {len(test_dataset)} amostras.")

    checkpoint_dir = str(MODELS_PATH / model_type)
    
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "best_model_fold_*.pth")))
    
    if not checkpoint_files:
        print(f"Erro: Nenhum arquivo de pesos encontrado em {checkpoint_dir}")
        return False
        
    print(f"Encontrados {len(checkpoint_files)} modelos para o ensemble:")
    
    ensemble_models = []
    for ckpt_file in checkpoint_files:
        print(f"  Carregando: {os.path.basename(ckpt_file)}")
        model, criterion, _ = create_model_with_architecture(
            hyperparams=hyperparams,
            architecture_name=architecture_name,
            class_names=class_names,
            device=device,
            train_dataset=test_dataset,
            verbose=False
        )
        checkpoint = torch.load(ckpt_file, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model.eval()

        model.architecture_name = architecture_name
        model.hyperparameters = hyperparams
        model.class_names = class_names
        ensemble_models.append(model)

    wandb_active = _open_wandb_inference_run(
        config, architecture_name, model_type, hyperparams, models_path
    )

    print_section("EXECUTANDO AVALIAÇÃO ENSEMBLE NO TEST SET")
    
    evaluation_results = evaluate_ensemble(
        models=ensemble_models,
        test_dataset=test_dataset,
        device=device,
        generate_gradcam=generate_gradcam,
        gradcam_samples=gradcam_samples,
        is_multiclass=is_multiclass,
        save_path=os.path.join(models_path, model_type)
    )

    if wandb_active:
        finish_wandb_run()

    print_detailed_metrics(evaluation_results['test_metrics'], class_names)

    results_file = save_inference_results(evaluation_results, models_path, model_type)

    print_banner("AVALIAÇÃO CONCLUÍDA!", f"Resultados salvos em: {results_file.name}")
    return True
