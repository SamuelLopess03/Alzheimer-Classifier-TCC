import os
import shutil
import zipfile
import subprocess
from PIL import Image
from typing import Tuple, List, Optional

def has_images(folder_path: str, min_files: int = 1) -> bool:
    image_extensions = {'.jpg', '.jpeg'}
    image_count = 0
    try:
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_count += 1
                if image_count >= min_files:
                    return True
    except Exception as e:
        print(f"\nErro ao buscar images: {e}\n")
        return False
    return False

def find_dataset_classes(dataset_path: str) -> Optional[List[str]]:
    classes = []
    try:
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path) and has_images(item_path):
                classes.append(item)
        classes.sort()
    except Exception as e:
        print(f"\nErro ao buscar classes: {e}\n")
        return None
    return classes if classes else None

def _setup_kaggle_credentials(kaggle_json_path: str) -> bool:
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    if kaggle_json_path and os.path.exists(kaggle_json_path):
        shutil.copy(kaggle_json_path, os.path.join(kaggle_dir, "kaggle.json"))
        os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
        print("Credenciais do Kaggle configuradas\n")
        return True
    
    kaggle_config = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_config):
        print("Erro: Arquivo kaggle.json não encontrado!\n")
        print("Configure suas credenciais em ~/.kaggle/kaggle.json\n")
        return False
    return True

def _download_zip(dataset_name: str, output_dir: str) -> Optional[str]:
    zip_filename = os.path.join(output_dir, dataset_name.split("/")[-1] + ".zip")
    
    if os.path.exists(zip_filename) and zipfile.is_zipfile(zip_filename):
        print(f"Dataset '{zip_filename}' já encontrado e íntegro. Pulando download...")
        return zip_filename

    print(f"Baixando dataset '{dataset_name}' do Kaggle...\n")
    cmd_download = [
        "kaggle", "datasets", "download", 
        "-d", dataset_name, 
        "-p", output_dir, 
        "-q",
        "--force"
    ]

    try:
        subprocess.run(cmd_download, check=True)
        print("Download concluído com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro no download: {e}")
        return None

    if not os.path.exists(zip_filename):
         potential_zips = [f for f in os.listdir(output_dir) if f.endswith('.zip')]
         if potential_zips:
             zip_filename = os.path.join(output_dir, potential_zips[0])
             
    if os.path.exists(zip_filename):
        return zip_filename
    return None

def _extract_and_cleanup_zip(zip_filename: str, output_dir: str) -> bool:
    try:
        print("Extraindo arquivos...")
        shutil.unpack_archive(zip_filename, output_dir)
        print("Extração concluída com sucesso.")
    except Exception as e:
        print(f"Erro na extração: {e}")
        return False

    try:
        zip_abs_path = os.path.abspath(zip_filename)
        if os.path.exists(zip_abs_path):
            os.remove(zip_abs_path)
            print(f"Arquivo zip removido: {zip_abs_path}\n")
    except Exception as e:
        print(f"Aviso ao remover arquivo zip: {e}\n")

    return True

def _elevate_nested_data_folder(output_dir: str):
    data_folder = os.path.join(output_dir, "Data")
    if not os.path.exists(data_folder):
        return

    print(f"Detectada subpasta 'Data' em {output_dir}. Elevando arquivos...")
    for item in os.listdir(data_folder):
        src = os.path.abspath(os.path.join(data_folder, item))
        dst = os.path.abspath(os.path.join(output_dir, item))
        
        if os.path.exists(dst):
            try:
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            except Exception as e:
                print(f"Aviso: Não foi possível limpar {dst}: {e}")
        
        try:
            shutil.move(src, dst)
        except Exception:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
                shutil.rmtree(src)
            else:
                shutil.copy2(src, dst)
                os.remove(src)

    try:
        shutil.rmtree(data_folder)
    except Exception:
        pass

def download_kaggle_dataset(
        dataset_name: str,
        output_dir: str,
        kaggle_json_path: str
) -> Tuple[bool, Optional[List[str]]]:
    print(f"{'-' * 60}")
    print("INICIANDO DOWNLOAD DO DATASET")
    print(f"{'-' * 60}\n")

    if not _setup_kaggle_credentials(kaggle_json_path):
        return False, None

    zip_filename = _download_zip(dataset_name, output_dir)
    if not zip_filename:
        return False, None

    if not _extract_and_cleanup_zip(zip_filename, output_dir):
        return False, None
        
    _elevate_nested_data_folder(output_dir)

    classes = find_dataset_classes(output_dir)
    if classes:
        print(f"Classes encontradas ({len(classes)}):")
        for i, cls in enumerate(classes, 1):
            print(f"   {i}. {cls}")
    else:
        print("AVISO: Nenhuma pasta de classe encontrada após extração.")

    print("-" * 60)
    print("DOWNLOAD E PREPARAÇÃO DO DATASET FINALIZADOS")
    print("-" * 60)

    return True, classes

def validate_image_files(directory: str) -> Tuple[int, int, List[str]]:
    valid_count = 0
    corrupt_count = 0
    errors = []

    print(f"\nValidando imagens em: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, file)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                    valid_count += 1
                except Exception as e:
                    corrupt_count += 1
                    errors.append(f"{filepath}: {str(e)}")
                    
    return valid_count, corrupt_count, errors
