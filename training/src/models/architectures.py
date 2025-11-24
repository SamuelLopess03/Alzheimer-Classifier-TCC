import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import timm
import numpy as np
from typing import Dict, Tuple, List
from sklearn.utils.class_weight import compute_class_weight

from .model_adapters import adapt_model_for_grayscale

def create_model(
        architecture_name: str,
        hidden_units: int,
        dropout: float,
        num_classes: int,
        device: torch.device
) -> nn.Module:
    arch_lower = architecture_name.lower()

    print(f"Criando modelo: {architecture_name}")
    print(f"   Hidden Units: {hidden_units}")
    print(f"   Dropout: {dropout}")
    print(f"   Num Classes: {num_classes}\n")

    # ===== CNN ARCHITECTURES =====

    if arch_lower == 'resnext50_32x4d':
        model = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        )
        model = adapt_model_for_grayscale(model, architecture_name)

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
        model = adapt_model_for_grayscale(model, architecture_name)

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
        model = adapt_model_for_grayscale(model, architecture_name)

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
        model = adapt_model_for_grayscale(model, architecture_name)

        model.classifier = nn.Sequential(
            nn.Linear(int(model.classifier.in_features), hidden_units),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes)
        )

    # ===== TRANSFORMER ARCHITECTURES =====

    elif arch_lower == 'vit_b_16':
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0  # Remove head
        )
        model = adapt_model_for_grayscale(model, architecture_name)

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
            'swinv2_tiny_window8_256',
            pretrained=True,
            num_classes=0  # Remove head
        )
        model = adapt_model_for_grayscale(model, architecture_name)

        feature_dim = int(model.num_features)
        model.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes)
        )

    else:
        raise ValueError(f"Arquitetura não suportada: {architecture_name}\n")

    if arch_lower in ['resnext50_32x4d', 'convnext_tiny', 'efficientnetv2_s', 'densenet121']:
        print("Congelando backbone (apenas classifier treinável)\n")
        for name, param in model.named_parameters():
            if not any(layer in name for layer in ['classifier', 'fc']):
                param.requires_grad = False

    elif arch_lower in ['vit_b_16', 'swin_v2_tiny']:
        print("Congelando patch embedding e primeiros blocos\n")
        for name, param in model.named_parameters():
            if 'patch_embed' in name:
                param.requires_grad = False
            elif any(f'blocks.{i}' in name for i in range(3)):
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
        label_smoothing: float = 0.1
) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
    print(f"{'-' * 60}")
    print(f"CRIANDO MODELO COM CONFIGURAÇÃO")
    print(f"{'-' * 60}\n")

    hidden_units = hyperparams['hidden_units']
    dropout = hyperparams['dropout']
    optimizer_name = hyperparams['optimizer']
    loss_function = hyperparams['loss_function']
    lr = hyperparams['learning_rate']

    num_classes = len(class_names)

    model = create_model(
        architecture_name=architecture_name,
        hidden_units=hidden_units,
        dropout=dropout,
        num_classes=num_classes,
        device=device
    )

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
    is_transformer = arch_lower in ['vit_b_16', 'swin_v2_tiny']

    if loss_function == 'crossentropy':
        if is_transformer:
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

    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=1e-4
        )
        print(f"Optimizer: Adam (lr={lr}, wd=1e-4)\n")

    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(
            trainable_params,
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        print(f"Optimizer: SGD (lr={lr}, momentum=0.9, wd=1e-4)\n")

    elif optimizer_name.lower() == 'adamw':
        wd = 0.01 if is_transformer else 1e-3
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
    arch_lower = architecture_name.lower()

    # CNN grid (ResNeXt, ConvNeXt, EfficientNet, DenseNet)
    cnn_grid = {
        'learning_rate': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'hidden_units': [128, 256, 512, 1024, 2048],
        'optimizer': ['adam', 'sgd', 'adamw'],
        'batch_size': [16, 32, 64, 128],
        'loss_function': ['crossentropy']
    }

    # ViT grid
    vit_grid = {
        'learning_rate': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],  # Lower LR
        'dropout': [0.1, 0.2, 0.3],  # Lower dropout
        'hidden_units': [256, 512, 1024],  # Fewer intermediate units
        'optimizer': ['adamw', 'adam'],  # AdamW preferred
        'batch_size': [16, 32, 64],  # Smaller batches (memory)
        'loss_function': ['crossentropy']
    }

    # Swin grid
    swin_grid = {
        'learning_rate': [5e-4, 1e-4, 5e-5, 1e-5],
        'dropout': [0.1, 0.2, 0.3],
        'hidden_units': [256, 512, 1024],
        'optimizer': ['adamw', 'adam'],
        'batch_size': [32, 64],  # Swin is more memory efficient
        'loss_function': ['crossentropy']
    }

    if arch_lower == 'vit_b_16':
        return vit_grid
    elif arch_lower == 'swin_v2_tiny':
        return swin_grid
    else:
        return cnn_grid

def get_supported_architectures() -> List[str]:
    return [
        'resnext50_32x4d',
        'convnext_tiny',
        'efficientnetv2_s',
        'densenet121',
        'vit_b_16',
        'swin_v2_tiny'
    ]
