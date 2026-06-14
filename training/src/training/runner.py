import os
from typing import Dict, Any, List, Optional
from torchvision import datasets as tv_datasets

from .factory import get_training_config
from .trainer import run_training_process
from .search import run_random_search
from src.utils.config import (
    load_yaml, 
    load_binary_config, 
    load_multiclass_config, 
    load_hyperparameters_config,
    load_augmentation_config,
    LOGS_PATH
)
from src.utils.experiments import find_best_experiment, extract_best_hyperparameters
from src.utils.hardware import detect_environment, get_pytorch_device
from src.visualization.terminal import (
    print_banner, 
    print_section, 
    print_class_distribution, 
    print_search_summary
)
from ..models import create_model_with_architecture
from ..data.dataset_wrappers import create_kfold_splits
from ..visualization.wandb_logger import init_wandb_run, finish_wandb_run

def run_training_flow(model_type: str, data_path: str) -> bool:
    is_multiclass = (model_type == 'multiclass')
    config = load_multiclass_config() if is_multiclass else load_binary_config()

    title = "NÍVEIS DE DEMÊNCIA" if is_multiclass else "NON DEMENTED vs DEMENTED"
    print_banner(f"TREINAMENTO DO MODELO {model_type.upper()}", title)

    train_path = os.path.join(data_path, f'splits/{model_type}/train')
    if not os.path.exists(train_path):
        print(f"Erro: Dataset não encontrado em {train_path}")
        return False

    print(f"Carregando dataset de: {train_path}\n")
    train_dataset = tv_datasets.ImageFolder(root=train_path, transform=None)
    
    class_names = config['model']['class_names']
    train_dataset.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_dataset.classes = class_names
    train_dataset.samples = [
        (path, train_dataset.class_to_idx[os.path.basename(os.path.dirname(path))])
        for path, _ in train_dataset.samples
    ]
    train_dataset.targets = [label for _, label in train_dataset.samples]

    print(f"Dataset carregado: {len(train_dataset)} amostras.")
    
    print_class_distribution(train_dataset, class_names)

    all_results = run_random_search(
        train_dataset=train_dataset,
        model_type=model_type
    )

    print_section("RESUMO DOS MELHORES RESULTADOS")
    print_search_summary(all_results)

    if not all_results:
        return False

    valid_results = {k: v for k, v in all_results.items() if v.get('best_params')}
    if valid_results:
        best_arch, best_result = max(
            valid_results.items(),
            key=lambda x: x[1].get('best_score', 0.0)
        )
        print(f"MELHOR ARQUITETURA: {best_arch.upper()}")
        print(f"  Score Final: {best_result['best_score']:.4f}")
        print(f"  F1-Score:    {best_result['best_metrics'].get('mean_f1', 0)*100:.2f}%")
        print("-" * 40 + "\n")

    return True

def run_final_training_flow(model_type: str, experiments_path: str, data_path: str) -> bool:
    is_multiclass = (model_type == 'multiclass')
    config = get_training_config(is_multiclass)
    device = get_pytorch_device()
    wandb_cfg = config.get('logging', {}).get('wandb', {})

    print_banner("PIPELINE: TREINAMENTO FINAL", f"Model Type: {model_type.upper()}")

    print_section("BUSCANDO MELHOR EXPERIMENTO")
    best_exp = find_best_experiment(experiments_path, model_type)
    if not best_exp:
        print(f"Erro: Nenhum experimento encontrado para '{model_type}' em {experiments_path}")
        return False

    hyperparams = extract_best_hyperparameters(best_exp)
    architecture_name = hyperparams['architecture_name']
    print(f"Melhor arquitetura encontrada: {architecture_name.upper()} (Score: {best_exp['best_score']:.4f})")

    print_section("CARREGANDO DATASET DE TREINAMENTO")
    train_path = os.path.join(data_path, f'splits/{model_type}/train')
    train_dataset = tv_datasets.ImageFolder(root=train_path, transform=None)
    
    class_names = config['model']['class_names']
    train_dataset.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_dataset.classes = class_names
    train_dataset.samples = [
        (path, train_dataset.class_to_idx[os.path.basename(os.path.dirname(path))])
        for path, _ in train_dataset.samples
    ]
    train_dataset.targets = [label for _, label in train_dataset.samples]
    
    print_class_distribution(train_dataset, class_names)

    print_section("CONFIGURANDO SPLIT K-FOLD PARA ENSEMBLE")
    
    all_folds = create_kfold_splits(
        dataset=train_dataset,
        n_folds=5,
        random_state=config['data']['random_seed'],
        max_slices_per_subject=config['data'].get('max_slices_per_subject'),
        minority_classes=config['data'].get('minority_classes', [0]),
        architecture_name=architecture_name,
        minority_config=config['data']['minority_augmentation'] if config['data']['minority_augmentation'].get('enabled') else None
    )

    ensemble_results = []
    
    for fold_idx, (train_split, val_split) in enumerate(all_folds, start=1):
        print_section(f"TREINANDO MODELO DO FOLD {fold_idx}/{len(all_folds)}")
        
        model, criterion, optimizer = create_model_with_architecture(
            hyperparams=hyperparams,
            architecture_name=architecture_name,
            class_names=config['model']['class_names'],
            device=device,
            train_dataset=train_split
        )
        
        if wandb_cfg.get('enabled', False):
            init_wandb_run(
                project_name=wandb_cfg['project'],
                run_name=f"{architecture_name}_{model_type}_fold_{fold_idx}",
                config={"architecture": architecture_name, "model_type": model_type, "fold": fold_idx, **hyperparams},
                entity=wandb_cfg.get('entity'),
                tags=["final_training", "ensemble", architecture_name, model_type],
                group=f"kfold_final_training/{model_type}"
            )

        result = run_training_process(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_split=train_split,
            val_split=val_split,
            device=device,
            hyperparams=hyperparams,
            architecture_name=architecture_name,
            is_multiclass=is_multiclass,
            is_final_training=True,
            fold_number=fold_idx
        )
        ensemble_results.append(result)
        
        if wandb_cfg.get('enabled', False):
            finish_wandb_run()

    print_banner("TREINAMENTO FINAL (ENSEMBLE) CONCLUÍDO!", f"{len(ensemble_results)} modelos salvos.")
    return True
