import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .builder import (
    prepare_dataset_binary, 
    prepare_dataset_multiclass
)
from .downloader import download_kaggle_dataset, validate_image_files
from ..utils import print_banner, print_section

def verify_datasets(output_path: str) -> bool:
    datasets_to_verify = [
        ('Binary Train', os.path.join(output_path, 'splits/binary/train')),
        ('Binary Test', os.path.join(output_path, 'splits/binary/test')),
        ('Multiclass Train', os.path.join(output_path, 'splits/multiclass/train')),
        ('Multiclass Test', os.path.join(output_path, 'splits/multiclass/test')),
    ]

    all_valid = True
    print("\nVERIFICANDO INTEGRIDADE DAS PASTAS:")
    for name, path in datasets_to_verify:
        exists = os.path.exists(path)
        status = "OK" if exists else "NOT FOUND"
        print(f"   {name:20s}: {status}")
        if not exists:
            all_valid = False

    return all_valid

def generate_dataset_metadata(output_path: str, kaggle_dataset: str):
    metadata = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'kaggle_dataset': kaggle_dataset,
        'splits': {}
    }

    for split_name in ['binary', 'multiclass']:
        split_path = os.path.join(output_path, f'splits/{split_name}')
        if not os.path.exists(split_path):
            continue
            
        metadata['splits'][split_name] = {}
        for phase in ['train', 'test']:
            phase_path = os.path.join(split_path, phase)
            if os.path.exists(phase_path):
                metadata['splits'][split_name][phase] = {
                    cls: len([f for f in os.listdir(os.path.join(phase_path, cls)) 
                             if f.lower().endswith(('.jpg', '.jpeg'))])
                    for cls in os.listdir(phase_path) 
                    if os.path.isdir(os.path.join(phase_path, cls))
                }

    metadata_file = os.path.join(output_path, 'split_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Metadados gerados em: {metadata_file}")

def run_data_preparation_flow(
    kaggle_dataset: str,
    output_path: str,
    kaggle_json: str,
    skip_binary: bool = False,
    skip_multiclass: bool = False,
    validate: bool = False
) -> bool:
    print_banner("PREPARAÇÃO DE DADOS", f"Dataset: {kaggle_dataset}")

    raw_dir = os.path.join(output_path, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(output_path, 'splits/multiclass')):
        print("Dados já preparados. Pulando download.\n")
    else:
        success, classes = download_kaggle_dataset(kaggle_dataset, raw_dir, kaggle_json)
        if not success:
            return False

    if not skip_binary:
        print_section("FASE 1: PROCESSAMENTO BINÁRIO")
        prepare_dataset_binary(output_base_path=output_path)

    if not skip_multiclass:
        print_section("FASE 2: PROCESSAMENTO MULTICLASSE")
        prepare_dataset_multiclass(output_base_path=output_path)

    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
        print("\nPasta temporária 'raw' removida.")

    print_section("FASE 3: VALIDAÇÃO E ESTATÍSTICAS")
    all_valid = verify_datasets(output_path)
    generate_dataset_metadata(output_path, kaggle_dataset)

    if validate:
        print("\nValidando arquivos de imagem...")
        v, c, _ = validate_image_files(output_path)
        print(f"Resultado: {v} válidas, {c} corrompidas.")

    print_banner("PREPARAÇÃO CONCLUÍDA!", "Status: SUCESSO" if all_valid else "Status: COM AVISOS")
    return all_valid
