from .subject_manager import (
    extract_subject_id,
    extract_slice_index,
    count_unique_subjects,
    get_subject_ids_from_dataset,
    resolve_subset_labels,
    split_dataset_train_test,
    DatasetMetadata,
    resolve_dataset_chain
)

from .pipeline import (
    run_data_preparation_flow,
    generate_dataset_metadata
)

from .downloader import (
    download_kaggle_dataset,
    validate_image_files
)

__all__ = [
    # Subject Management (leve, sem dependências pesadas)
    "extract_subject_id",
    "extract_slice_index",
    "count_unique_subjects",
    "get_subject_ids_from_dataset",
    "resolve_subset_labels",
    "split_dataset_train_test",
    "DatasetMetadata",
    "resolve_dataset_chain",

    # Pipeline & Download (usados pelo container de setup)
    "run_data_preparation_flow",
    "generate_dataset_metadata",
    "download_kaggle_dataset",
    "validate_image_files",
]