import json
from pathlib import Path
from typing import Dict, Optional

EXECUTION_STATE_FILE = 'execution_state.json'

def find_best_experiment(experiments_path: str, model_type: str) -> Optional[Dict]:
    experiments_dir = Path(experiments_path)
    if not experiments_dir.exists():
        return None

    best_experiment = None
    best_score_final = -1.0
    pattern = f"*_{model_type}"

    matching_dirs = list(experiments_dir.glob(pattern))
    if not matching_dirs:
        return None

    for exp_dir in matching_dirs:
        state_file = exp_dir / EXECUTION_STATE_FILE
        if not state_file.exists():
            continue

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            results = state.get('results', {})
            if not results:
                continue

            best_score = results.get('best_score', -1.0)
            architecture_name = exp_dir.name.replace(f"_{model_type}", "")

            if best_score > best_score_final:
                best_score_final = best_score
                best_experiment = {
                    'architecture_name': architecture_name,
                    'experiment_dir': str(exp_dir),
                    'state_file': str(state_file),
                    'best_score': best_score,
                    'results': results,
                    'executed_indices': state.get('executed_indices', []),
                    'model_type': model_type
                }
        except Exception:
            continue

    return best_experiment

def extract_best_hyperparameters(experiment: Dict) -> Dict:
    results = experiment['results']
    hyperparameters = results.get('best_params', {})
    
    hyperparameters['architecture_name'] = experiment['architecture_name']
    hyperparameters['model_type'] = experiment['model_type']
    hyperparameters['best_score'] = experiment['best_score']
    
    return hyperparameters
