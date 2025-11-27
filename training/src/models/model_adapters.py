import torch
import torch.nn as nn
from typing import cast, Optional

from training.src.utils import load_hyperparameters_config

def adapt_model_for_grayscale(
        model: nn.Module,
        architecture_name: str,
        preserve_pretrained_weights: Optional[bool] = None
) -> nn.Module:
    hyperparams_config = load_hyperparameters_config()
    grayscale_cfg = hyperparams_config['grayscale_adaptation']

    if preserve_pretrained_weights is None:
        preserve_pretrained_weights = grayscale_cfg['preserve_pretrained_weights']

    if not grayscale_cfg['enabled']:
        print(f"\nAdaptação para grayscale desabilitada no config. Retornando modelo original.\n")
        return model

    print(f"\nAdaptando {architecture_name} para entrada grayscale...")

    arch_lower = architecture_name.lower()

    arch_cfg = hyperparams_config['model_config'][arch_lower]
    adaptation_layer = arch_cfg['adaptation_layer']

    try:
        if 'resnext' in arch_lower or 'resnet' in arch_lower:
            model = _adapt_resnet_family(model, preserve_pretrained_weights)

        elif 'convnext' in arch_lower:
            model = _adapt_convnext(model, preserve_pretrained_weights)

        elif 'efficientnet' in arch_lower:
            model = _adapt_efficientnet(model, preserve_pretrained_weights)

        elif 'densenet' in arch_lower:
            model = _adapt_densenet(model, preserve_pretrained_weights)

        elif 'vit' in arch_lower:
            model = _adapt_vit(model, preserve_pretrained_weights)

        elif 'swin' in arch_lower:
            model = _adapt_swin(model, preserve_pretrained_weights)

        print(f"Camada adaptada: {adaptation_layer}")
        print(f"Modelo adaptado com sucesso para grayscale!\n")

    except Exception as e:
        print(f"\nErro ao adaptar modelo: {e}\n")
        raise

    return model

def _adapt_resnet_family(
        model: nn.Module,
        preserve_weights: bool = True
) -> nn.Module:
    original_conv = model.conv1

    kernel_size = (original_conv.kernel_size[0], original_conv.kernel_size[1])
    stride = (original_conv.stride[0], original_conv.stride[1])
    padding = (original_conv.padding[0], original_conv.padding[1])

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )

    if preserve_weights:
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

    model.conv1 = new_conv

    return model

def _adapt_convnext(
        model: nn.Module,
        preserve_weights: bool = True
) -> nn.Module:
    original_conv = cast(nn.Conv2d, model.features[0][0])

    kernel_size = (original_conv.kernel_size[0], original_conv.kernel_size[1])
    stride = (original_conv.stride[0], original_conv.stride[1])

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )

    if preserve_weights:
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
            if original_conv.bias is not None:
                new_conv.bias = original_conv.bias

    model.features[0][0] = new_conv

    return model

def _adapt_efficientnet(
        model: nn.Module,
        preserve_weights: bool = True
) -> nn.Module:
    original_conv = cast(nn.Conv2d, model.features[0][0])

    kernel_size = (original_conv.kernel_size[0], original_conv.kernel_size[1])
    stride = (original_conv.stride[0], original_conv.stride[1])

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )

    if preserve_weights:
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
            if original_conv.bias is not None:
                new_conv.bias = original_conv.bias

    model.features[0][0] = new_conv

    return model

def _adapt_densenet(
        model: nn.Module,
        preserve_weights: bool = True
) -> nn.Module:
    original_conv = model.features.conv0

    kernel_size = (original_conv.kernel_size[0], original_conv.kernel_size[1])
    stride = (original_conv.stride[0], original_conv.stride[1])
    padding = (original_conv.padding[0], original_conv.padding[1])

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )

    if preserve_weights:
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

    model.features.conv0 = new_conv

    return model

def _adapt_vit(
        model: nn.Module,
        preserve_weights: bool = True
) -> nn.Module:
    try:
        original_conv = model.patch_embed.proj
    except AttributeError:
        raise ValueError("Modelo ViT não possui patch_embed.proj esperado")

    kernel_size = (original_conv.kernel_size[0], original_conv.kernel_size[1])
    stride = (original_conv.stride[0], original_conv.stride[1])
    padding = (original_conv.padding[0], original_conv.padding[1])

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=original_conv.bias is not None
    )

    if preserve_weights:
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
            if original_conv.bias is not None:
                new_conv.bias = original_conv.bias

    model.patch_embed.proj = new_conv

    return model

def _adapt_swin(
        model: nn.Module,
        preserve_weights: bool = True
) -> nn.Module:
    try:
        original_conv = model.patch_embed.proj
    except AttributeError:
        raise ValueError("Estrutura do Swin não possui 'patch_embed.proj'")

    kernel_size = (original_conv.kernel_size[0], original_conv.kernel_size[1])
    stride = (original_conv.stride[0], original_conv.stride[1])

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )

    if preserve_weights:
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
            if original_conv.bias is not None:
                new_conv.bias = original_conv.bias

    model.patch_embed.proj = new_conv

    return model

def verify_grayscale_adaptation(
        model: nn.Module,
        architecture_name: str,
        expected_channels: Optional[int] = None
) -> bool:
    hyperparams_config = load_hyperparameters_config()

    if expected_channels is None:
        expected_channels = hyperparams_config['grayscale_adaptation']['expected_channels']

    arch_lower = architecture_name.lower()

    if arch_lower not in hyperparams_config['model_config']:
        print(f"\nArquitetura {architecture_name} não reconhecida para verificação\n")
        return False

    arch_cfg = hyperparams_config['model_config'][arch_lower]
    adaptation_layer = arch_cfg['adaptation_layer']

    try:
        actual_channels = 1

        if 'resnext' in arch_lower or 'resnet' in arch_lower:
            actual_channels = model.conv1.in_channels

        elif 'convnext' in arch_lower:
            conv = cast(nn.Conv2d, model.features[0][0])
            actual_channels = conv.in_channels

        elif 'efficientnet' in arch_lower:
            conv = cast(nn.Conv2d, model.features[0][0])
            actual_channels = conv.in_channels

        elif 'densenet' in arch_lower:
            actual_channels = model.features.conv0.in_channels

        elif 'vit' in arch_lower:
            actual_channels = model.patch_embed.proj.in_channels

        elif 'swin' in arch_lower:
            actual_channels = model.patch_embed.proj.in_channels

        is_valid = (actual_channels == expected_channels)

        if is_valid:
            print(f"\nVerificação OK: {adaptation_layer} possui {actual_channels} canal(is) de entrada\n")
        else:
            print(
                f"\nVerificação FALHOU: {adaptation_layer} possui {actual_channels} canais (esperado: {expected_channels})\n")

        return is_valid

    except Exception as e:
        print(f"\nErro na verificação: {e}\n")
        return False

def get_adaptation_info(architecture_name: str) -> dict:
    hyperparams_config = load_hyperparameters_config()
    arch_lower = architecture_name.lower()

    if arch_lower not in hyperparams_config['model_config']:
        raise ValueError(f"Arquitetura não suportada: {architecture_name}")

    arch_cfg = hyperparams_config['model_config'][arch_lower]
    grayscale_cfg = hyperparams_config['grayscale_adaptation']

    return {
        'architecture': architecture_name,
        'type': arch_cfg['type'],
        'adaptation_layer': arch_cfg['adaptation_layer'],
        'preserve_weights': grayscale_cfg['preserve_pretrained_weights'],
        'expected_channels': grayscale_cfg['expected_channels'],
        'enabled': grayscale_cfg['enabled']
    }