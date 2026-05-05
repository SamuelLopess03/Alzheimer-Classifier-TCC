import torch
import torch.nn as nn
import torch.optim as optim
import re
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.utils.class_weight import compute_class_weight

from .builder_registry import model_registry
from . import builders 
from src.utils.config import load_hyperparameters_config

def _handle_freezing(model: nn.Module, architecture_name: str, arch_cfg: Dict, verbose: bool = True):
    freeze_backbone = arch_cfg.get('freeze_backbone', False)
    strategy = arch_cfg.get('unfreeze_strategy', 'none').lower()
    unfreeze_layers = arch_cfg.get('unfreeze_layers', [])
    classifier_layer = arch_cfg.get('classifier_layer', 'fc')
    
    # Prioridade para o freeze_backbone antigo se for True
    if freeze_backbone:
        strategy = 'none'

    if strategy == 'full':
        for param in model.parameters():
            param.requires_grad = True
        if verbose: print(f"Treinamento TOTAL habilitado para {architecture_name}")
        
    elif strategy == 'partial':
        # Congela tudo primeiro
        for param in model.parameters():
            param.requires_grad = False
            
        # Descongela as camadas especificadas (suporta Regex) e o head
        for name, param in model.named_parameters():
            # Match rigoroso para o head: deve ser o nome exato ou estar no final/início do nome
            is_head = any(re.search(rf"(^|\.){k}(\.|$)", name) for k in [classifier_layer, 'classifier', 'fc', 'head'])
            is_target = any(re.search(layer, name) for layer in unfreeze_layers)
            
            if is_head or is_target:
                param.requires_grad = True
        
        if verbose:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Fine-Tuning PARCIAL habilitado para {architecture_name}")
            print(f"   Camadas desbloqueadas: {unfreeze_layers} + Head")
            print(f"   Ratio de Parâmetros Treináveis: {trainable/total*100:.1f}%")
            
    else: # strategy == 'none'
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(re.search(rf"(^|\.){k}(\.|$)", name) for k in [classifier_layer, 'classifier', 'fc', 'head']):
                param.requires_grad = True
        if verbose: print(f"Backbone TOTALMENTE congelado para {architecture_name} (apenas o head será treinado)")

def create_model(
        architecture_name: str,
        hidden_units: int,
        dropout: float,
        num_classes: int,
        device: torch.device,
        verbose: bool = True
) -> nn.Module:
    hyperparams_config = load_hyperparameters_config()
    arch_lower = architecture_name.lower()

    if arch_lower not in hyperparams_config['model_config']:
        raise ValueError(f"Arquitetura não suportada no config: {architecture_name}\n")

    builder = model_registry.get_builder(architecture_name)
    if not builder:
        raise ValueError(f"Builder não encontrado para a arquitetura: {architecture_name}")

    arch_cfg = hyperparams_config['model_config'][arch_lower]
    grayscale_cfg = hyperparams_config['grayscale_adaptation']

    if verbose:
        print(f"Criando modelo: {architecture_name}")
        print(f"   Hidden Units: {hidden_units}")
        print(f"   Dropout: {dropout}")
        print(f"   Num Classes: {num_classes}\n")

    model = builder.build_base(architecture_name, arch_cfg)
    
    if grayscale_cfg.get('enabled', True):
        if verbose: print(f"Adaptando {architecture_name} para entrada grayscale...")
        model = builder.adapt_grayscale(
            model, 
            arch_cfg, 
            preserve_weights=grayscale_cfg.get('preserve_pretrained_weights', True)
        )
        if verbose: print(f"Modelo adaptado com sucesso para grayscale!\n")
    else:
        if verbose: print(f"Adaptação para grayscale desabilitada no config. Retornando modelo original.\n")

    in_features = builder.get_in_features(model, arch_cfg)
    model = builder.replace_head(model, arch_cfg, in_features, hidden_units, dropout, num_classes)

    _handle_freezing(model, architecture_name, arch_cfg, verbose=verbose)

    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"Modelo criado")
        print(f"   Parâmetros treináveis: {trainable_params:,}")
        print(f"   Parâmetros totais: {total_params:,}")
        print(f"   Ratio: {trainable_params / total_params * 100:.1f}%\n")

    return model

def verify_grayscale_adaptation(
        model: nn.Module,
        architecture_name: str,
        expected_channels: Optional[int] = None
) -> bool:
    hyperparams_config = load_hyperparameters_config()
    arch_cfg = hyperparams_config['model_config'].get(architecture_name.lower())

    if expected_channels is None:
        expected_channels = hyperparams_config['grayscale_adaptation']['expected_channels']

    builder = model_registry.get_builder(architecture_name)
    if not builder:
        print(f"\nErro: Builder não encontrado para verificação: {architecture_name}\n")
        return False

    try:
        is_valid = builder.verify_grayscale(model, arch_cfg)
        if is_valid:
            print(f"\nVerificação OK: Modelo possui {expected_channels} canal(is) de entrada\n")
        else:
            print(f"\nVerificação FALHOU: Modelo não possui {expected_channels} canais\n")
        return is_valid
    except Exception as e:
        print(f"\nErro na verificação: {e}\n")
        return False

def _setup_criterion(loss_function: str, hyperparams_config: Dict, train_dataset, num_classes: int, architecture_name: str, device: torch.device, label_smoothing: Optional[float]) -> nn.Module:
    loss_cfg = hyperparams_config['loss_config'][loss_function]
    class_weights = None
    
    if loss_cfg['use_class_weights']:
        print("Calculando pesos das classes...\n")
        labels = [sample[1] for sample in train_dataset]
        class_weights = compute_class_weight('balanced', classes=np.array(list(range(num_classes))), y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"   Class Weights: {class_weights.cpu().numpy()}\n")

    if loss_function == 'crossentropy':
        if label_smoothing is None:
            arch_cfg = hyperparams_config['model_config'].get(architecture_name.lower())
            arch_type = arch_cfg['type']
            label_smoothing = loss_cfg['label_smoothing'][arch_type]

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing if label_smoothing else 0.0)
        print(f"Loss: CrossEntropyLoss (label_smoothing={label_smoothing})\n")
        return criterion
    
    raise ValueError(f"Loss function não suportada: {loss_function}\n")

def _setup_optimizer(model: nn.Module, hyperparams: Dict, hyperparams_config: Dict, architecture_name: str) -> optim.Optimizer:
    optimizer_name = hyperparams['optimizer'].lower()
    lr = float(hyperparams['learning_rate'])
    arch_cfg = hyperparams_config['model_config'].get(architecture_name.lower())
    arch_type = arch_cfg['type']
    
    fine_tuning_cfg = hyperparams_config.get('fine_tuning', {})
    backbone_lr_ratio = float(fine_tuning_cfg.get('backbone_lr_ratio', {}).get(arch_type, 0.01))
    
    backbone_lr = lr * backbone_lr_ratio
    head_lr = lr

    classifier_layer = arch_cfg['classifier_layer']
    head_params, backbone_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if classifier_layer in name: head_params.append(param)
        else: backbone_params.append(param)

    print(f"\nFine-Tuning com LR Diferencial:")
    print(f"   Backbone LR: {backbone_lr:.2e} (ratio: {backbone_lr_ratio})")
    print(f"   Head LR:     {head_lr:.2e}")

    param_groups = []
    if backbone_params: param_groups.append({'params': backbone_params, 'lr': backbone_lr})
    if head_params: param_groups.append({'params': head_params, 'lr': head_lr})
        
    opt_cfg = hyperparams_config['optimizer_config'][optimizer_name]
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(param_groups, weight_decay=float(opt_cfg['weight_decay']))
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=opt_cfg['momentum'], weight_decay=float(opt_cfg['weight_decay']))
    elif optimizer_name == 'adamw':
        wd = float(opt_cfg['weight_decay'][arch_type])
        optimizer = optim.AdamW(param_groups, weight_decay=wd)
    else:
        raise ValueError(f"Optimizer não suportado: {optimizer_name}\n")

    print(f"Optimizer: {optimizer_name.upper()} (backbone_lr={backbone_lr:.2e}, head_lr={head_lr:.2e})\n")
    return optimizer

def create_model_with_architecture(
        hyperparams: Dict,
        architecture_name: str,
        class_names: List[str],
        device: torch.device,
        train_dataset,
        label_smoothing: Optional[float] = None
) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
    hyperparams_config = load_hyperparameters_config()

    print(f"{'-' * 60}")
    print(f"CRIANDO MODELO COM CONFIGURAÇÃO")
    print(f"{'-' * 60}\n")

    num_classes = len(class_names)
    model = create_model(
        architecture_name=architecture_name,
        hidden_units=hyperparams['hidden_units'],
        dropout=hyperparams['dropout'],
        num_classes=num_classes,
        device=device
    )

    criterion = _setup_criterion(hyperparams['loss_function'], hyperparams_config, train_dataset, num_classes, architecture_name, device, label_smoothing)

    optimizer = _setup_optimizer(model, hyperparams, hyperparams_config, architecture_name)

    print(f"\n{'-' * 60}\n")
    return model, criterion, optimizer

def get_architecture_specific_param_grid(architecture_name: str) -> Dict:
    hyperparams_config = load_hyperparameters_config()
    arch_lower = architecture_name.lower()

    if arch_lower not in hyperparams_config['model_config']:
        raise ValueError(f"Arquitetura não suportada: {architecture_name}")

    if 'vit' in arch_lower: grid_key = 'vit'
    elif 'swin' in arch_lower: grid_key = 'swin'
    else: grid_key = 'cnn'

    return hyperparams_config['hyperparameter_grids'][grid_key]

def get_supported_architectures() -> List[str]:
    return model_registry.get_all_supported_architectures()