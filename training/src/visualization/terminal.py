from typing import Dict, List, Any, Optional
from collections import Counter

def print_banner(title: str, subtitle: Optional[str] = None):
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    if subtitle:
        print(f"{subtitle:^80}")
    print(f"{'=' * 80}\n")

def print_section(title: str):
    print(f"\n{'-' * 60}")
    print(f"{title}")
    print(f"{'-' * 60}\n")

def print_repetition_summary(aggregated: Dict[str, Any], index: int):
    print(f"\nAgregação da Combinação #{index + 1}")
    print(f"  {'Metric':<20} | {'Mean':<10} | {'Std':<10}")
    print(f"  {'-' * 45}")
    
    metrics_to_show = [
        ('F1-Score', 'mean_f1', 'std_f1'),
        ('Balanced Acc', 'mean_balanced_accuracy', 'std_balanced_accuracy'),
        ('Loss', 'mean_loss', 'std_loss')
    ]
    
    for label, mean_key, std_key in metrics_to_show:
        mean_val = aggregated.get(mean_key, 0.0)
        std_val = aggregated.get(std_key, 0.0)
        
        if 'loss' in mean_key.lower():
            print(f"  {label:<20} | {mean_val:<10.4f} | {std_val:<10.4f}")
        else:
            print(f"  {label:<20} | {mean_val*100:<9.2f}% | {std_val*100:<9.2f}%")
    print(f"  {'-' * 45}\n")

def print_class_distribution(train_dataset: Any, class_names: List[str]):
    labels = [label for _, label in train_dataset.samples]
    class_dist = Counter(labels)

    print("\n  DISTRIBUIÇÃO POR CLASSE:")
    print(f"  {'-' * 40}")
    for class_idx, class_name in enumerate(class_names):
        count = class_dist.get(class_idx, 0)
        percentage = (count / len(train_dataset) * 100) if len(train_dataset) > 0 else 0
        print(f"    {class_name:25s}: {count:5d} ({percentage:>5.1f}%)")
    print(f"  {'-' * 40}\n")

def print_search_summary(all_results: Dict[str, Any]):
    if not all_results:
        print("\nNenhum resultado válido obtido no Random Search!\n")
        return

    print(f"\n{'Arquitetura':<20} | {'Score (Bal.Acc)':<15} | {'F1-Score':<12}")
    print(f"{'-' * 60}")

    for arch, results in all_results.items():
        if results.get('best_params'):
            score = results.get('best_score', 0.0)
            best_metrics = results.get('best_metrics', {})
            f1 = best_metrics.get('mean_f1', 0.0)
            print(f"{arch:<20} | {score:<15.4f} | {f1 * 100:<11.2f}%")
        else:
            print(f"{arch:<20} | {'FALHOU':<15} | {'-':<12}")
    print(f"{'-' * 60}\n")

def print_detailed_metrics(metrics: Dict[str, Any], class_names: List[str] = None):
    print("\nMÉTRICAS DETALHADAS (MÉDIAS):")
    print(f"  {'-' * 45}")
    
    if 'mean_balanced_accuracy' in metrics:
        print(f"    Balanced Accuracy : {metrics['mean_balanced_accuracy'] * 100:>6.2f}%")
    if 'mean_f1' in metrics:
        print(f"    F1-Score (Macro)  : {metrics['mean_f1'] * 100:>6.2f}%")
    if 'mean_precision' in metrics:
        print(f"    Precision (Macro) : {metrics['mean_precision'] * 100:>6.2f}%")
    if 'mean_recall' in metrics:
        print(f"    Recall (Macro)    : {metrics['mean_recall'] * 100:>6.2f}%")
    if 'mean_mcc' in metrics:
        print(f"    MCC               : {metrics['mean_mcc']:>6.4f}")
    print(f"  {'-' * 45}")

    if class_names and 'f1_per_class' in metrics:
        print("\nDESEMPENHO POR CLASSE:")
        f1_pc = metrics.get('f1_per_class', [])
        prec_pc = metrics.get('precision_per_class', [])
        rec_pc = metrics.get('recall_per_class', [])
        
        for i, name in enumerate(class_names):
            if i < len(f1_pc):
                print(f"    {name:25s} | F1: {f1_pc[i]*100:>5.1f}% | Prec: {prec_pc[i]*100:>5.1f}% | Rec: {rec_pc[i]*100:>5.1f}%")
        print()

def print_epoch_log(epoch: int, num_epochs: int, train_loss: float, metrics: Dict[str, Any], patience: int, is_best: bool = False):
    val_loss = metrics.get('val_loss', 0.0)
    val_f1 = metrics.get('f1_score', 0.0)
    val_acc = metrics.get('accuracy', 0.0)
    val_bacc = metrics.get('balanced_accuracy', 0.0)
    
    status_indicator = "[BEST]" if is_best else "      "
    
    print(f"| Epoch {epoch:03d}/{num_epochs:03d} "
          f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
          f"F1: {val_f1*100:5.2f}% | "
          f"Acc: {val_acc*100:5.2f}% | "
          f"B.Acc: {val_bacc*100:5.2f}% | "
          f"Patience: {patience:2d} | "
          f"{status_indicator}")
