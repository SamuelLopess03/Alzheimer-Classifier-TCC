import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from typing import Tuple

def create_stratified_holdout_split(
        dataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.3,
        random_state: int = 42
) -> Tuple[Subset, Subset]:
    print(f"{'-' * 60}")
    print(f"CRIANDO HOLDOUT SPLIT ESTRATIFICADO")
    print(f"{'-' * 60}\n")

    print(f"Configuração:")
    print(f"  Train Ratio: {train_ratio:.1%}")
    print(f"  Val Ratio: {val_ratio:.1%}")
    print(f"  Random State: {random_state}\n")

    total_ratio = train_ratio + val_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio devem somar 1.0, "
            f"mas somam {total_ratio:.4f}"
        )

    print("Extraindo labels do dataset...\n")
    labels = [dataset[i][1] for i in range(len(dataset))]
    indices = list(range(len(dataset)))

    print(f"Dataset total: {len(dataset)} amostras\n")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=random_state
    )

    train_indices, val_indices = next(splitter.split(indices, labels))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)

    print("Distribuição por Classe:")
    print("\n  TREINO:")
    total_train = sum(train_counts.values())
    for cls in sorted(train_counts.keys()):
        count = train_counts[cls]
        pct = count / total_train * 100
        print(f"    Classe {cls}: {count:>5} amostras ({pct:>5.1f}%)")
    print(f"    Total:     {total_train:>5} amostras")

    print("\n  VALIDAÇÃO:")
    total_val = sum(val_counts.values())
    for cls in sorted(val_counts.keys()):
        count = val_counts[cls]
        pct = count / total_val * 100
        print(f"    Classe {cls}: {count:>5} amostras ({pct:>5.1f}%)")
    print(f"    Total:     {total_val:>5} amostras")

    print(f"\n{'-' * 60}\n")

    return train_dataset, val_dataset

def verify_split_stratification(
        train_subset: Subset,
        val_subset: Subset,
        tolerance: float = 0.05
) -> bool:
    base_dataset = train_subset.dataset
    while hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset

    if hasattr(base_dataset, 'targets'):
        all_labels = np.array(base_dataset.targets)
    elif hasattr(base_dataset, 'labels'):
        all_labels = np.array(base_dataset.labels)
    else:
        all_labels = np.array([base_dataset[i][1] for i in range(len(base_dataset))])

    train_labels = all_labels[train_subset.indices]
    val_labels = all_labels[val_subset.indices]

    train_props = Counter(train_labels)
    val_props = Counter(val_labels)

    train_total = len(train_labels)
    val_total = len(val_labels)

    is_valid = True
    for cls in set(train_labels) | set(val_labels):
        train_pct = train_props.get(cls, 0) / train_total
        val_pct = val_props.get(cls, 0) / val_total

        diff = abs(train_pct - val_pct)

        if diff > tolerance:
            print(f"Classe {cls}: diferença de {diff:.1%} excede tolerância de {tolerance:.1%}\n")
            is_valid = False

    if is_valid:
        print("Estratificação válida\n")

    return is_valid

def get_split_statistics(
        train_subset: Subset,
        val_subset: Subset,
        test_subset: Subset = None
) -> dict:
    base_dataset = train_subset.dataset
    while hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset

    if hasattr(base_dataset, 'targets'):
        all_labels = np.array(base_dataset.targets)
    elif hasattr(base_dataset, 'labels'):
        all_labels = np.array(base_dataset.labels)
    else:
        all_labels = np.array([base_dataset[i][1] for i in range(len(base_dataset))])

    train_labels = all_labels[train_subset.indices]
    val_labels = all_labels[val_subset.indices]

    stats = {
        'train_size': len(train_labels),
        'val_size': len(val_labels),
        'train_distribution': dict(Counter(train_labels)),
        'val_distribution': dict(Counter(val_labels)),
    }

    if test_subset is not None:
        test_labels = all_labels[test_subset.indices]
        stats['test_size'] = len(test_labels)
        stats['test_distribution'] = dict(Counter(test_labels))

    return stats