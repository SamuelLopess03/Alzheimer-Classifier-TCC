import os
from typing import Dict, Any, List, Optional
from torchvision import datasets as tv_datasets

from .factory import get_training_config
from .trainer import run_training_process
from .search import run_random_search
from ..utils import (
    load_binary_config, 
    load_multiclass_config, 
    print_banner, 
    print_section,
    find_best_experiment,
    extract_best_hyperparameters,
    get_pytorch_device
)
from ..visualization import print_class_distribution, print_search_summary, init_wandb_run, finish_wandb_run
from ..models import create_model_with_architecture
from ..data import create_stratified_holdout_split

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
    print(f"Dataset carregado: {len(train_dataset)} amostras.")
    
    print_class_distribution(train_dataset, config['model']['class_names'])

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
    
    print_class_distribution(train_dataset, config['model']['class_names'])

    train_ratio = config['data']['split_ratios']['train']
    val_ratio = config['data']['split_ratios']['train_val']
    train_split, val_split = create_stratified_holdout_split(
        train_dataset, train_ratio, val_ratio, random_state=config['data']['random_seed']
    )

    print_section("CONFIGURANDO MODELO FINAL")
    model, criterion, optimizer = create_model_with_architecture(
        hyperparams=hyperparams,
        architecture_name=architecture_name,
        class_names=config['model']['class_names'],
        device=device,
        train_dataset=train_dataset
    )
    
    print_section("EXECUTANDO TREINAMENTO DE PRODUÇÃO")
    
    wandb_cfg = config.get('logging', {}).get('wandb', {})
    if wandb_cfg.get('enabled', False):
        init_wandb_run(
            project_name=wandb_cfg['project'],
            run_name=f"{architecture_name}_{model_type}_final",
            config={"architecture": architecture_name, "model_type": model_type, **hyperparams},
            entity=wandb_cfg.get('entity'),
            tags=["final_training", architecture_name, model_type],
            group=f"final_training/{model_type}"  # Pasta dedicada no WandB
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
        is_final_training=True
    )

    finish_wandb_run()

    print_banner("TREINAMENTO FINAL CONCLUÍDO!", f"Modelo salvo em: {result.get('checkpoint_path', 'N/A')}")
    return True
