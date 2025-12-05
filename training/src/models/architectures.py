import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import timm
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.utils.class_weight import compute_class_weight

from .model_adapters import adapt_model_for_grayscale
from training.src.utils import load_hyperparameters_config

def create_model(
        architecture_name: str,
        hidden_units: int,
        dropout: float,
        num_classes: int,
        device: torch.device
) -> nn.Module:
    hyperparams_config = load_hyperparameters_config()
    arch_lower = architecture_name.lower()

    if arch_lower not in hyperparams_config['model_config']:
        raise ValueError(f"Arquitetura não suportada: {architecture_name}\n")

    arch_cfg = hyperparams_config['model_config'][arch_lower]
    grayscale_cfg = hyperparams_config['grayscale_adaptation']

    print(f"Criando modelo: {architecture_name}")
    print(f"   Hidden Units: {hidden_units}")
    print(f"   Dropout: {dropout}")
    print(f"   Num Classes: {num_classes}\n")

    model = None

    # ===== CNN ARCHITECTURES =====

    if arch_lower == 'resnext50_32x4d':
        model = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        )
        model = adapt_model_for_grayscale(
            model,
            architecture_name,
            preserve_pretrained_weights=grayscale_cfg['preserve_pretrained_weights']
        )

        model.fc = nn.Sequential(
            nn.Linear(int(model.fc.in_features), hidden_units),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes)
        )

    elif arch_lower == 'convnext_tiny':
        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        model = adapt_model_for_grayscale(
            model,
            architecture_name,
            preserve_pretrained_weights=grayscale_cfg['preserve_pretrained_weights']
        )

        layernorm: nn.Module = model.classifier[0]
        flatten: nn.Module = model.classifier[1]
        in_features: int = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            layernorm,
            flatten,
            nn.Linear(in_features, hidden_units),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes)
        )

    elif arch_lower == 'efficientnetv2_s':
        model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        model = adapt_model_for_grayscale(
            model,
            architecture_name,
            preserve_pretrained_weights=grayscale_cfg['preserve_pretrained_weights']
        )

        in_features: int = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_units),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes)
        )

    elif arch_lower == 'densenet121':
        model = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        )
        model = adapt_model_for_grayscale(
            model,
            architecture_name,
            preserve_pretrained_weights=grayscale_cfg['preserve_pretrained_weights']
        )

        model.classifier = nn.Sequential(
            nn.Linear(int(model.classifier.in_features), hidden_units),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes)
        )

    # ===== TRANSFORMER ARCHITECTURES =====

    elif arch_lower == 'vit_b_16':
        model = timm.create_model(
            arch_cfg['timm_model'],
            pretrained=arch_cfg['pretrained'],
            num_classes=0
        )
        model = adapt_model_for_grayscale(
            model,
            architecture_name,
            preserve_pretrained_weights=grayscale_cfg['preserve_pretrained_weights']
        )

        feature_dim = int(model.num_features)
        model.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes)
        )

    elif arch_lower == 'swin_v2_tiny':
        model = timm.create_model(
            arch_cfg['timm_model'],
            pretrained=arch_cfg['pretrained'],
            num_classes=0
        )
        model = adapt_model_for_grayscale(
            model,
            architecture_name,
            preserve_pretrained_weights=grayscale_cfg['preserve_pretrained_weights']
        )

        feature_dim = int(model.num_features)
        model.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes)
        )

    if arch_cfg.get('freeze_backbone', False):
        print(f"Congelando backbone (apenas {arch_cfg['classifier_layer']} treinável)\n")
        for name, param in model.named_parameters():
            if not any(layer in name for layer in ['classifier', 'fc', 'head']):
                param.requires_grad = False

    elif arch_cfg.get('freeze_patch_embed', False):
        freeze_blocks = arch_cfg.get('freeze_first_blocks', 0)
        print(f"Congelando patch embedding e primeiros {freeze_blocks} blocos\n")
        for name, param in model.named_parameters():
            if 'patch_embed' in name:
                param.requires_grad = False
            elif any(f'blocks.{i}' in name for i in range(freeze_blocks)):
                param.requires_grad = False

    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Modelo criado")
    print(f"   Parâmetros treináveis: {trainable_params:,}")
    print(f"   Parâmetros totais: {total_params:,}")
    print(f"   Ratio: {trainable_params / total_params * 100:.1f}%\n")

    return model

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

    hidden_units = hyperparams['hidden_units']
    dropout = hyperparams['dropout']
    optimizer_name = hyperparams['optimizer']
    loss_function = hyperparams['loss_function']
    lr = float(hyperparams['learning_rate'])

    num_classes = len(class_names)

    model = create_model(
        architecture_name=architecture_name,
        hidden_units=hidden_units,
        dropout=dropout,
        num_classes=num_classes,
        device=device
    )

    loss_cfg = hyperparams_config['loss_config'][loss_function]

    class_weights = None
    if loss_cfg['use_class_weights']:
        print("Calculando pesos das classes...\n")
        labels = [sample[1] for sample in train_dataset]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.array(list(range(num_classes))),
            y=labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"   Class Weights: {class_weights.cpu().numpy()}\n")

    arch_lower = architecture_name.lower()
    arch_cfg = hyperparams_config['model_config'][arch_lower]
    is_transformer = arch_cfg['type'] == 'transformer'

    if loss_function == 'crossentropy':
        if label_smoothing is None:
            arch_type = 'transformer' if is_transformer else 'cnn'
            label_smoothing = loss_cfg['label_smoothing'][arch_type]

        if is_transformer and label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
            print(f"Loss: CrossEntropyLoss (label_smoothing={label_smoothing})\n")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Loss: CrossEntropyLoss\n")
    else:
        raise ValueError(f"Loss function não suportada: {loss_function}\n")

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    opt_cfg = hyperparams_config['optimizer_config'][optimizer_name.lower()]

    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=float(opt_cfg['weight_decay'])
        )
        print(f"Optimizer: Adam (lr={lr}, wd={opt_cfg['weight_decay']})\n")

    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(
            trainable_params,
            lr=lr,
            momentum=opt_cfg['momentum'],
            weight_decay=float(opt_cfg['weight_decay'])
        )
        print(f"Optimizer: SGD (lr={lr}, momentum={opt_cfg['momentum']}, wd={opt_cfg['weight_decay']})\n")

    elif optimizer_name.lower() == 'adamw':
        arch_type = 'transformer' if is_transformer else 'cnn'
        wd = float(opt_cfg['weight_decay'][arch_type])
        optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=wd
        )
        print(f"Optimizer: AdamW (lr={lr}, wd={wd})\n")

    else:
        raise ValueError(f"Optimizer não suportado: {optimizer_name}\n")

    print(f"\n{'-' * 60}\n")

    return model, criterion, optimizer

def get_architecture_specific_param_grid(architecture_name: str) -> Dict:
    hyperparams_config = load_hyperparameters_config()
    arch_lower = architecture_name.lower()

    if arch_lower not in hyperparams_config['model_config']:
        raise ValueError(f"Arquitetura não suportada: {architecture_name}")

    if arch_lower == 'vit_b_16':
        grid_key = 'vit'
    elif arch_lower == 'swin_v2_tiny':
        grid_key = 'swin'
    else:
        grid_key = 'cnn'

    return hyperparams_config['hyperparameter_grids'][grid_key]

def get_supported_architectures() -> List[str]:
    hyperparams_config = load_hyperparameters_config()

    cnn_archs = hyperparams_config['supported_architectures']['cnn']
    transformer_archs = hyperparams_config['supported_architectures']['transformer']

    return cnn_archs + transformer_archs