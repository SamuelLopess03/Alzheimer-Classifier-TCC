from .builder import (
    prepare_dataset_binary,
    binarize_alzheimer_dataset,
    prepare_dataset_multiclass
)

from .downloader import (
    download_kaggle_dataset,
    validate_image_files
)

from .split import (
    split_dataset_train_test,
    DatasetMetadata,
    fast_copy
)

from .subject_utils import (
    extract_subject_id,
    extract_slice_index,
    resolve_dataset_chain,
    count_unique_subjects,
    resolve_subset_labels
)

from .pipeline import (
    run_data_preparation_flow,
    verify_datasets,
    generate_dataset_metadata
)

# Componentes que exigem PyTorch/Albumentations (Lazy Loading)
# Isso permite que 'import src.data' não carregue o Torch no container de setup.

_HEAVY_COMPONENTS = {
    # Preprocessing
    'MedicalImagePreprocessor': '.preprocessing',
    'convert_pil_to_numpy': '.preprocessing',
    'convert_tensor_to_numpy': '.preprocessing',
    'prepare_image_for_augmentation': '.preprocessing',
    'validate_image_format': '.preprocessing',

    # Augmentation
    'get_alzheimer_grayscale_augmentation': '.augmentation',
    'create_synthetic_augmentation_for_minority': '.augmentation',

    # Dataset Wrappers e Split em Memória (PyTorch)
    'DynamicAugmentationDataset': '.dataset_wrappers',
    'StaticPreprocessedDataset': '.dataset_wrappers',
    'augment_minority_class': '.dataset_wrappers',
    'SubjectSamplingSubset': '.dataset_wrappers',
    'create_stratified_holdout_split': '.dataset_wrappers',
}

def __getattr__(name):
    if name in _HEAVY_COMPONENTS:
        import importlib
        module_path = _HEAVY_COMPONENTS[name]
        module = importlib.import_module(module_path, __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    # Dataset Preparation Runners - LEVE
    'run_data_preparation_flow',
    'verify_datasets',
    'generate_dataset_metadata',

    # Subject Utils - LEVE (sem PyTorch no top-level)
    'extract_subject_id',
    'extract_slice_index',
    'resolve_dataset_chain',
    'count_unique_subjects',
    'resolve_subset_labels',

    # Dataset Loaders (binary & multiclass) - LEVE
    'prepare_dataset_binary',
    'binarize_alzheimer_dataset',
    'prepare_dataset_multiclass',
    'download_kaggle_dataset',
    'validate_image_files',

    # Split e Diagnóstico - LEVE (sem PyTorch no top-level)
    'split_dataset_train_test',
    'DatasetMetadata',
    'fast_copy',

    # Preprocessing - PESADO (Lazy)
    'MedicalImagePreprocessor',
    'convert_pil_to_numpy',
    'convert_tensor_to_numpy',
    'prepare_image_for_augmentation',

    # Augmentation - PESADO (Lazy)
    'get_alzheimer_grayscale_augmentation',
    'create_synthetic_augmentation_for_minority',

    # Dataset Wrappers + Split em Memória - PESADO (Lazy, depende de PyTorch)
    'DynamicAugmentationDataset',
    'StaticPreprocessedDataset',
    'augment_minority_class',
    'SubjectSamplingSubset',
    'create_stratified_holdout_split',
]