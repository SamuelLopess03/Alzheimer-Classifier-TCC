import os
import zipfile
import shutil
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from torchvision import datasets

def download_kaggle_dataset(
        dataset_name: str,
        output_dir: str,
        kaggle_json_path: str
) -> Tuple[bool, Optional[List[str]]]:
    print(f"{'-' * 60}")
    print("INICIANDO DOWNLOAD DO DATASET")
    print(f"{'-' * 60}\n")

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    if kaggle_json_path and os.path.exists(kaggle_json_path):
        shutil.copy(kaggle_json_path, os.path.join(kaggle_dir, "kaggle.json"))
        os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
        print("Credenciais do Kaggle configuradas\n")
    else:
        kaggle_config = os.path.join(kaggle_dir, "kaggle.json")
        if not os.path.exists(kaggle_config):
            print("Erro: Arquivo kaggle.json não encontrado!\n")
            print("Configure suas credenciais em ~/.kaggle/kaggle.json\n")
            return False, None

    print(f"Baixando dataset '{dataset_name}' do Kaggle...\n")
    download_result = os.system(f"kaggle datasets download -d {dataset_name} -p {output_dir}")

    if download_result != 0:
        print("Erro no download. Verifique o nome do dataset e suas credenciais.\n")
        return False, None

    zip_filename = os.path.join(output_dir, dataset_name.split("/")[-1] + ".zip")
    if not os.path.exists(zip_filename):
        print("Arquivo zip não encontrado após download.\n")
        return False, None

    print(f"Extraindo arquivos para: {output_dir}\n")

    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Download e extração concluídos com sucesso\n")
    except Exception as e:
        print(f"Erro na extração: {e}\n")
        return False, None

    data_folder = os.path.join(output_dir, "Data")

    if os.path.exists(data_folder):
        for item in os.listdir(data_folder):
            src = os.path.join(data_folder, item)
            dst = os.path.join(output_dir, item)

            shutil.move(src, dst)

        shutil.rmtree(data_folder)

    classes = find_dataset_classes(output_dir)

    if classes:
        print(f"\nClasses encontradas ({len(classes)}):")
        for i, class_name in enumerate(classes, 1):
            print(f"   {i}. {class_name}")
    else:
        print("Nenhuma classe encontrada\n")

    try:
        os.remove(zip_filename)
        print(f"Arquivo zip removido: {zip_filename}\n")
    except Exception as e:
        print(f"Erro ao remover arquivo zip: {e}\n")
        pass

    print("-" * 60)
    print("DOWNLOAD DO DATASET FINALIZADO")
    print("-" * 60)

    return True, classes

def find_dataset_classes(dataset_path: str) -> Optional[List[str]]:
    classes = []

    try:
        data_folder = dataset_path

        for item in os.listdir(data_folder):
            item_path = os.path.join(data_folder, item)
            if os.path.isdir(item_path) and has_images(item_path):
                classes.append(item)

        classes.sort()

    except Exception as e:
        print(f"\nErro ao buscar classes: {e}\n")
        return None

    return classes if classes else None

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

def split_dataset_train_test(
        dataset_path: str,
        classes: List[str],
        train_ratio: float,
        output_train_path: str,
        output_test_path: str,
        random_state: int
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    print("\n" + "-" * 60)
    print(f"INICIANDO DIVISÃO DO DATASET ({int(train_ratio * 100)}% TREINO / {int((1 - train_ratio) * 100)}% TESTE)")
    print("-" * 60 + "\n")

    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.exists(class_path):
            print(f"Classe {class_name} não encontrada em {class_path}\n")
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg'))]

        if not images:
            print(f"Nenhuma imagem encontrada para a classe {class_name}\n")
            continue

        train_images, test_images = train_test_split(
            images,
            train_size=train_ratio,
            random_state=random_state,
            shuffle=True
        )

        train_class_path = os.path.join(output_train_path, class_name)
        test_class_path = os.path.join(output_test_path, class_name)

        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        for image in train_images:
            src = os.path.join(class_path, image)
            dst = os.path.join(train_class_path, image)
            shutil.copy2(src, dst)

        for image in test_images:
            src = os.path.join(class_path, image)
            dst = os.path.join(test_class_path, image)
            shutil.copy2(src, dst)

        print(f"Classe {class_name}: {len(train_images)} treino, {len(test_images)} teste\n")

    train_dataset = datasets.ImageFolder(root=output_train_path, transform=None)
    test_dataset = datasets.ImageFolder(root=output_test_path, transform=None)

    print(f"\nResumo:")
    print(f"  Dataset de treino: {len(train_dataset)} imagens")
    print(f"  Dataset de teste: {len(test_dataset)} imagens")
    print(f"  Classes: {train_dataset.classes}")

    try:
        shutil.rmtree(dataset_path)
        print(f"\nPasta temporária removida: {dataset_path}")
    except Exception as e:
        print(f"\nErro ao remover pasta temporária: {e}")

    print("\n" + "-" * 60)
    print("DIVISÃO DO DATASET CONCLUÍDA")
    print("-" * 60)

    return train_dataset, test_dataset