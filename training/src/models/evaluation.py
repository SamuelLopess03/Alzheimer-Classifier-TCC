import torch
import torch.nn as nn
import os
from typing import Tuple, Optional, Dict, List
from pathlib import Path

from ..utils import (
    load_binary_config, 
    load_multiclass_config, 
    load_hyperparameters_config,
    find_best_experiment,
    extract_best_hyperparameters
)
from .architectures import create_model

class Evaluation(nn.Module):
    def __init__(self, device: torch.device = None):
        super(Evaluation, self).__init__()

        self.binary_config = load_binary_config()
        self.multiclass_config = load_multiclass_config()
        self.hyperparams_config = load_hyperparameters_config()

        self.binary_class_names = self.binary_config['model']['class_names']
        self.multiclass_class_names = self.multiclass_config['model']['class_names']
        self.binary_num_classes = self.binary_config['model']['num_classes']
        self.multiclass_num_classes = self.multiclass_config['model']['num_classes']

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.binary_model = None
        self.multiclass_model = None

    def predict(self, x: torch.Tensor) -> Dict:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        binary_output, multiclass_output = self._forward(x)

        return self._format_prediction_result(
            binary_probs=torch.softmax(binary_output, dim=1)[0],
            multiclass_probs=torch.softmax(multiclass_output, dim=1)[0] if multiclass_output is not None else None
        )

    def predict_subject(self, x: torch.Tensor) -> Dict:
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        x = x.to(self.device)
        binary_output, multiclass_output = self._forward(x)

        binary_probs_avg = torch.softmax(binary_output, dim=1).mean(dim=0)
        
        multiclass_probs_avg = None
        if multiclass_output is not None:
            multiclass_probs_avg = torch.softmax(multiclass_output, dim=1).mean(dim=0)

        return self._format_prediction_result(
            binary_probs=binary_probs_avg,
            multiclass_probs=multiclass_probs_avg
        )

    def _forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.binary_model is None or self.multiclass_model is None:
            raise RuntimeError("Modelos não carregados. Chame load_models() primeiro.")

        with torch.no_grad():
            binary_output = self.binary_model(x)
            binary_pred = torch.argmax(binary_output, dim=1)

            multiclass_output = None
            if torch.any(binary_pred == 0):  
                multiclass_output = self.multiclass_model(x)

            return binary_output, multiclass_output

    def _format_prediction_result(self, binary_probs: torch.Tensor, multiclass_probs: Optional[torch.Tensor]) -> Dict:
        binary_class_idx = torch.argmax(binary_probs).item()
        binary_confidence = binary_probs[binary_class_idx].item()
        binary_class_name = self.binary_class_names[binary_class_idx]

        result = {
            'binary_prediction': {
                'class_index': binary_class_idx,
                'class_name': binary_class_name,
                'confidence': binary_confidence,
                'probabilities': {
                    self.binary_class_names[i]: binary_probs[i].item()
                    for i in range(self.binary_num_classes)
                }
            },
            'multiclass_prediction': None,
            'final_prediction': binary_class_name,
            'final_confidence': binary_confidence,
            'requires_multiclass': False
        }

        if binary_class_idx == 0 and multiclass_probs is not None:
            multiclass_class_idx = torch.argmax(multiclass_probs).item()
            multiclass_confidence = multiclass_probs[multiclass_class_idx].item()
            multiclass_class_name = self.multiclass_class_names[multiclass_class_idx]

            result['multiclass_prediction'] = {
                'class_index': multiclass_class_idx,
                'class_name': multiclass_class_name,
                'confidence': multiclass_confidence,
                'probabilities': {
                    self.multiclass_class_names[i]: multiclass_probs[i].item()
                    for i in range(self.multiclass_num_classes)
                }
            }
            result['final_prediction'] = multiclass_class_name
            result['final_confidence'] = multiclass_confidence
            result['requires_multiclass'] = True

        return result

    def _discover_best_models(self, experiments_path: str) -> Tuple[Dict, Dict]:
        try:
            binary_exp = find_best_experiment(experiments_path, 'binary')
            multiclass_exp = find_best_experiment(experiments_path, 'multiclass')
            
            if not binary_exp or not multiclass_exp:
                raise ValueError("Experimentos não encontrados.")
            
            bin_hparams = extract_best_hyperparameters(binary_exp)
            multi_hparams = extract_best_hyperparameters(multiclass_exp)
            
            print(f"Modelo Binário: {bin_hparams['architecture_name']}")
            print(f"Modelo Multiclasse: {multi_hparams['architecture_name']}")
            return bin_hparams, multi_hparams
        except Exception as e:
            print(f"Erro ao localizar configurações: {e}")
            raise

    def _initialize_model_instances(self, bin_hparams: Dict, multi_hparams: Dict):
        self.binary_model = create_model(
            architecture_name=bin_hparams['architecture_name'],
            hidden_units=bin_hparams['hidden_units'],
            dropout=bin_hparams['dropout'],
            num_classes=self.binary_num_classes,
            device=self.device
        )
        self.multiclass_model = create_model(
            architecture_name=multi_hparams['architecture_name'],
            hidden_units=multi_hparams['hidden_units'],
            dropout=multi_hparams['dropout'],
            num_classes=self.multiclass_num_classes,
            device=self.device
        )

    def _resolve_checkpoint_path(self, config: Dict) -> str:
        checkpoint_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), str(config['checkpoint']['save_path'])))
        return str(Path(checkpoint_dir) / "best_model.pth")

    def _load_weights(self, model: nn.Module, path: str, label: str):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print(f"Pesos {label} carregados de: {os.path.basename(path)}")
        except Exception as e:
            print(f"Erro ao carregar pesos {label}: {e}")
            raise

    def load_models(self, experiments_path: str = "../../shared/logs", binary_path: str = None, multiclass_path: str = None):
        print(f"\n{'-' * 60}\nCONFIGURANDO SISTEMA DE AVALIAÇÃO EM CASCADA\n{'-' * 60}")

        bin_hparams, multi_hparams = self._discover_best_models(experiments_path)

        self._initialize_model_instances(bin_hparams, multi_hparams)

        binary_path = binary_path or self._resolve_checkpoint_path(self.binary_config)
        multiclass_path = multiclass_path or self._resolve_checkpoint_path(self.multiclass_config)

        self._load_weights(self.binary_model, binary_path, "Binário")
        self._load_weights(self.multiclass_model, multiclass_path, "Multiclasse")

        self.binary_model.eval()
        self.multiclass_model.eval()

        print(f"\n{'-' * 60}\nSISTEMA PRONTO PARA INFERÊNCIA\n{'-' * 60}")

    def print_prediction(self, result: Dict):
        print(f"\n{'=' * 40}\nDIAGNÓSTICO FINAL\n{'=' * 40}")
        print(f"BINÁRIO: {result['binary_prediction']['class_name']} ({result['binary_prediction']['confidence'] * 100:.2f}%)")
        
        if result['requires_multiclass']:
            print(f"ESTÁGIO: {result['multiclass_prediction']['class_name']} ({result['multiclass_prediction']['confidence'] * 100:.2f}%)")
        
        print(f"{'-' * 40}\nPREDIÇÃO: {result['final_prediction'].upper()}\n{'=' * 40}")