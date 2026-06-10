import sys
import os
import random
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from collections import defaultdict

# Adiciona o diretório raiz ao path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import (
    prepare_image_for_augmentation,
    denormalize_images,
    MedicalImagePreprocessor,
    crop_mri_background
)
from src.data.augmentation import get_alzheimer_grayscale_augmentation
from src.data.subject_manager import extract_subject_id, extract_slice_index

# ─── Caminhos base ─────────────────────────────────────────────────────────────
SPLITS_DIR    = Path(__file__).resolve().parent.parent / "shared" / "data" / "splits"
OUTPUT_BASE   = Path(__file__).resolve().parent.parent / "shared" / "visualization_augmented"

# ─── Abordagens e suas classes (lê do split de treino) ─────────────────────────
APPROACHES = {
    "binary": {
        "classes": ["Non Demented", "Demented"],
        "split":   "train",
    },
    "multiclass": {
        "classes": ["Very mild Dementia", "Mild+Moderate Dementia"],
        "split":   "train",
    },
}

ARCHITECTURES = [
    "resnext50_32x4d",
    "convnext_tiny",
    "efficientnetv2_s",
    "densenet121",
    "vit_b_16",
    "swin_v2_tiny",
]


# ─── Funções auxiliares ─────────────────────────────────────────────────────────

def select_central_sample(cls_dir: Path) -> Path | None:
    """Agrupa fatias por paciente, sorteia um paciente e retorna a fatia central."""
    files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg"))
    if not files:
        return None

    patient_groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        sid = extract_subject_id(f.name)
        patient_groups[sid].append(f)

    random_sid   = random.choice(list(patient_groups.keys()))
    slices       = sorted(patient_groups[random_sid], key=lambda f: extract_slice_index(f.name))
    central      = slices[len(slices) // 2]
    return central, random_sid, len(slices)


def process_panel_image(image: np.ndarray, title: str, size_str: str,
                        panel_size: int = 256) -> np.ndarray:
    resized = cv2.resize(image, (panel_size, panel_size))
    bgr     = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, title,    (10, 25),             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(bgr, size_str, (10, panel_size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.rectangle(bgr, (0, 0), (panel_size - 1, panel_size - 1), (255, 255, 255), 1)
    return bgr


def build_flow_panel(img_path: Path, arch: str,
                     mean, std, target_size_config: int) -> np.ndarray:
    """Aplica o fluxo completo de pré-processamento + augmentation e monta o painel."""
    real_pipeline = get_alzheimer_grayscale_augmentation(
        architecture_name=arch,
        dataset_size=10,        # força nível 'heavy' para visualização clara
        is_training=True
    )

    img_orig = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    h_orig, w_orig = img_orig.shape

    # Crop de fundo preto
    img_cropped = crop_mri_background(img_orig)
    h_crop, w_crop = img_cropped.shape

    # Prepara para augmentation
    img_prepped = prepare_image_for_augmentation(img_orig)
    img_resized = cv2.resize(img_prepped, (target_size_config, target_size_config))

    # 3 versões augmentadas com o pipeline real
    augmented_versions = []
    for _ in range(3):
        aug_result = real_pipeline(image=img_prepped)
        tensor_img = aug_result["image"]
        np_img     = tensor_img.numpy().transpose(1, 2, 0)
        img_visual = denormalize_images(np_img, mean, std)
        if img_visual.ndim == 3 and img_visual.shape[2] == 1:
            img_visual = img_visual.squeeze(-1)
        augmented_versions.append(img_visual)

    p_orig    = process_panel_image(img_orig,               "1. Original",      f"{w_orig}x{h_orig}")
    p_cropped = process_panel_image(img_cropped,            "2. Crop (BBox)",   f"{w_crop}x{h_crop}")
    p_resized = process_panel_image(img_resized,            "3. Redim.",        f"{target_size_config}x{target_size_config}")
    p_aug1    = process_panel_image(augmented_versions[0],  "4. Aug Real 1",   f"{target_size_config}x{target_size_config}")
    p_aug2    = process_panel_image(augmented_versions[1],  "5. Aug Real 2",   f"{target_size_config}x{target_size_config}")
    p_aug3    = process_panel_image(augmented_versions[2],  "6. Aug Real 3",   f"{target_size_config}x{target_size_config}")

    return np.hstack([p_orig, p_cropped, p_resized, p_aug1, p_aug2, p_aug3])


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 85)
    print("  VISUALIZADOR DE PRÉ-PROCESSAMENTO E DATA AUGMENTATION (PIPELINE REAL MULTI-ARQ)")
    print("=" * 85)

    for approach_name, cfg in APPROACHES.items():
        classes   = cfg["classes"]
        split     = cfg["split"]
        split_dir = SPLITS_DIR / approach_name / split

        print(f"\n{'#' * 85}")
        print(f"  ABORDAGEM: {approach_name.upper()}  |  Split: {split}  |  Dir: {split_dir}")
        print(f"{'#' * 85}")

        # ── 1. Seleciona fatia central por paciente para cada classe ──────────
        samples: dict[str, Path] = {}
        print("\nSelecionando amostras (fatia central por paciente):")

        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"  [Aviso] Diretório não encontrado: {cls_dir}")
                continue

            result = select_central_sample(cls_dir)
            if result is None:
                print(f"  [Aviso] Nenhuma imagem encontrada em: {cls_dir}")
                continue

            selected_file, random_sid, total_fatias = result
            samples[cls] = selected_file
            print(f"  - {cls:30s}: {selected_file.name}")
            print(f"    {'':30s}  Paciente: {random_sid}  |  Total Fatias: {total_fatias}")

        if not samples:
            print(f"\n  [Erro] Nenhuma amostra encontrada para a abordagem '{approach_name}'. Pulando.")
            continue

        # ── 2. Processa cada arquitetura ──────────────────────────────────────
        for arch in ARCHITECTURES:
            print(f"\n{'-' * 85}")
            print(f"  ARQUITETURA: {arch.upper()}")
            print(f"{'-' * 85}")

            preprocessor      = MedicalImagePreprocessor(arch)
            mean              = preprocessor.config["mean"]
            std               = preprocessor.config["std"]
            target_size_cfg   = preprocessor.config["image_size"]
            print(f"  Config: Dimensões={target_size_cfg}x{target_size_cfg} | Mean={mean} | Std={std}")

            # Diretório de saída organizado por abordagem → arquitetura
            out_dir = OUTPUT_BASE / approach_name / arch
            out_dir.mkdir(parents=True, exist_ok=True)

            for class_name, img_path in samples.items():
                panel      = build_flow_panel(img_path, arch, mean, std, target_size_cfg)
                safe_label = class_name.replace(" ", "_").replace("+", "_")
                fname      = img_path.stem
                out_file   = out_dir / f"{safe_label}_{fname}_flow.jpg"
                cv2.imwrite(str(out_file), panel)
                print(f"    [Salvo] {safe_label} -> {approach_name}/{arch}/{out_file.name}")

    print(f"\n{'=' * 85}")
    print(f"  Sucesso! Visualizações salvas em: {OUTPUT_BASE}")
    print(f"{'=' * 85}\n")


if __name__ == "__main__":
    main()
