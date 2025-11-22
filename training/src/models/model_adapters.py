import torch
import torch.nn as nn
from typing import cast

def adapt_model_for_grayscale(
        model: nn.Module,
        architecture_name: str,
        preserve_pretrained_weights: bool = True
) -> nn.Module:
    print(f"\nAdaptando {architecture_name} para entrada grayscale...")

    arch_lower = architecture_name.lower()

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

        else:
            raise ValueError(f"Arquitetura não suportada: {architecture_name}")

        print(f"\nModelo adaptado com sucesso para grayscale!\n")

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
        expected_channels: int = 1
) -> bool:
    arch_lower = architecture_name.lower()

    try:
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

        else:
            print(f"\nArquitetura {architecture_name} não reconhecida para verificação\n")
            return False

        is_valid = (actual_channels == expected_channels)

        if is_valid:
            print(f"\nVerificação OK: {actual_channels} canal(is) de entrada\n")
        else:
            print(f"\nVerificação FALHOU: {actual_channels} canais (esperado: {expected_channels})\n")

        return is_valid

    except Exception as e:
        print(f"\nErro na verificação: {e}\n")
        return False
