import sys
import argparse
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import (
    MedicalImagePreprocessor,
    DynamicAugmentationDataset,
)
from src.utils import load_binary_config, load_multiclass_config, print_banner, print_section

def test_preprocessor():
    print_section("TESTANDO CONFIGURAÇÕES DE PRÉ-PROCESSAMENTO")
    
    architectures = [
        'resnext50_32x4d', 'convnext_tiny', 'efficientnetv2_s',
        'densenet121', 'vit_b_16', 'swin_v2_tiny'
    ]

    for arch in architectures:
        print(f"Validando: {arch}")
        preprocessor = MedicalImagePreprocessor(arch)
        preprocessor.print_config()
        print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description="Script de teste para o pipeline de dados")
    parser.add_argument('--type', choices=['binary', 'multiclass'], default='binary')
    parser.add_argument('--arch', default='resnext50_32x4d')
    args = parser.parse_args()

    print_banner("TESTE DE PRÉ-PROCESSAMENTO E DATASET", f"Arch: {args.arch} | Tipo: {args.type}")

    test_preprocessor()

    print_section(f"TESTANDO DATASET ({args.type.upper()})")
    config = load_multiclass_config() if args.type == 'multiclass' else load_binary_config()
    
    base_path = Path(__file__).resolve().parent.parent / f"shared/data/splits/{args.type}/train"
    
    if not base_path.exists():
        print(f"Erro: Pasta do dataset não encontrada em {base_path}")
        return

    print(f"Pasta encontrada: {base_path}")
    print("\nTeste concluído com sucesso!")

if __name__ == "__main__":
    main()