import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
from pathlib import Path

from ..utils import load_binary_config, load_multiclass_config

class HierarchicalPipeline(nn.Module):
    def __init__(
            self,
            binary_model: nn.Module,
            multiclass_model: nn.Module,
            device: torch.device = None
    ):
        super(HierarchicalPipeline, self).__init__()

        self.binary_config = load_binary_config()
        self.multiclass_config = load_multiclass_config()

        self.binary_class_names = self.binary_config['model']['class_names']
        self.multiclass_class_names = self.multiclass_config['model']['class_names']
        self.binary_num_classes = self.binary_config['model']['num_classes']
        self.multiclass_num_classes = self.multiclass_config['model']['num_classes']

        self.binary_model = binary_model
        self.multiclass_model = multiclass_model

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.binary_model = self.binary_model.to(self.device)
        self.multiclass_model = self.multiclass_model.to(self.device)
        self.binary_model.eval()
        self.multiclass_model.eval()

        print(f"\n{'=' * 60}")
        print("PIPELINE HIERÁRQUICO INICIALIZADO")
        print(f"{'=' * 60}")
        print(f"Configurações:")
        print(f"  Device: {self.device}")
        print(f"\nModelo Binário:")
        print(f"  Classes: {self.binary_class_names}")
        print(f"  Número de classes: {self.binary_num_classes}")
        print(f"\nModelo Multiclasse:")
        print(f"  Classes: {self.multiclass_class_names}")
        print(f"  Número de classes: {self.multiclass_num_classes}")
        print(f"{'=' * 60}\n")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with torch.no_grad():
            binary_output = self.binary_model(x)
            binary_pred = torch.argmax(binary_output, dim=1)

            multiclass_output = None

            if torch.any(binary_pred == 0):  # 0 = Demented
                multiclass_output = self.multiclass_model(x)

            return binary_output, multiclass_output

    def predict(self, x: torch.Tensor) -> Dict:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        binary_output, multiclass_output = self.forward(x)

        binary_probs = torch.softmax(binary_output, dim=1)
        binary_class_idx = torch.argmax(binary_probs, dim=1).item()
        binary_confidence = binary_probs[0, binary_class_idx].item()
        binary_class_name = self.binary_class_names[binary_class_idx]

        result = {
            'binary_prediction': {
                'class_index': binary_class_idx,
                'class_name': binary_class_name,
                'confidence': binary_confidence,
                'probabilities': {
                    self.binary_class_names[i]: binary_probs[0, i].item()
                    for i in range(self.binary_num_classes)
                }
            },
            'multiclass_prediction': None,
            'final_prediction': binary_class_name,
            'final_confidence': binary_confidence,
            'requires_multiclass': False
        }

        if binary_class_idx == 0 and multiclass_output is not None:
            multiclass_probs = torch.softmax(multiclass_output, dim=1)
            multiclass_class_idx = torch.argmax(multiclass_probs, dim=1).item()
            multiclass_confidence = multiclass_probs[0, multiclass_class_idx].item()
            multiclass_class_name = self.multiclass_class_names[multiclass_class_idx]

            result['multiclass_prediction'] = {
                'class_index': multiclass_class_idx,
                'class_name': multiclass_class_name,
                'confidence': multiclass_confidence,
                'probabilities': {
                    self.multiclass_class_names[i]: multiclass_probs[0, i].item()
                    for i in range(self.multiclass_num_classes)
                }
            }
            result['final_prediction'] = multiclass_class_name
            result['final_confidence'] = multiclass_confidence
            result['requires_multiclass'] = True

        return result

    def predict_batch(self, x: torch.Tensor) -> List[Dict]:
        x = x.to(self.device)
        batch_size = x.shape[0]

        binary_output, multiclass_output = self.forward(x)

        results = []

        binary_probs = torch.softmax(binary_output, dim=1)

        for i in range(batch_size):
            binary_class_idx = torch.argmax(binary_probs[i]).item()
            binary_confidence = binary_probs[i, binary_class_idx].item()
            binary_class_name = self.binary_class_names[binary_class_idx]

            result = {
                'binary_prediction': {
                    'class_index': binary_class_idx,
                    'class_name': binary_class_name,
                    'confidence': binary_confidence,
                    'probabilities': {
                        self.binary_class_names[j]: binary_probs[i, j].item()
                        for j in range(self.binary_num_classes)
                    }
                },
                'multiclass_prediction': None,
                'final_prediction': binary_class_name,
                'final_confidence': binary_confidence,
                'requires_multiclass': False
            }

            if binary_class_idx == 0 and multiclass_output is not None:
                multiclass_probs = torch.softmax(multiclass_output, dim=1)
                multiclass_class_idx = torch.argmax(multiclass_probs[i]).item()
                multiclass_confidence = multiclass_probs[i, multiclass_class_idx].item()
                multiclass_class_name = self.multiclass_class_names[multiclass_class_idx]

                result['multiclass_prediction'] = {
                    'class_index': multiclass_class_idx,
                    'class_name': multiclass_class_name,
                    'confidence': multiclass_confidence,
                    'probabilities': {
                        self.multiclass_class_names[j]: multiclass_probs[i, j].item()
                        for j in range(self.multiclass_num_classes)
                    }
                }
                result['final_prediction'] = multiclass_class_name
                result['final_confidence'] = multiclass_confidence
                result['requires_multiclass'] = True

            results.append(result)

        return results

    def load_models(
            self,
            binary_path: str = None,
            multiclass_path: str = None
    ):
        if binary_path is None:
            binary_checkpoint_path = self.binary_config['checkpoint']['save_path']
            binary_path = str(Path(binary_checkpoint_path) / "best_model.pth")

        if multiclass_path is None:
            multiclass_checkpoint_path = self.multiclass_config['checkpoint']['save_path']
            multiclass_path = str(Path(multiclass_checkpoint_path) / "best_model.pth")

        print(f"\n{'-' * 60}")
        print("CARREGANDO CHECKPOINTS DOS MODELOS")
        print(f"{'-' * 60}\n")

        try:
            binary_checkpoint = torch.load(binary_path, map_location=self.device)
            if 'model_state_dict' in binary_checkpoint:
                self.binary_model.load_state_dict(binary_checkpoint['model_state_dict'])
                print(f"Modelo Binário carregado de: {binary_path}")
                if 'metrics' in binary_checkpoint:
                    metrics = binary_checkpoint['metrics']
                    print(f"  Métricas do checkpoint:")
                    print(f"    F1-Score: {metrics.get('f1_score', 0) * 100:.2f}%")
                    print(f"    Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%\n")
            else:
                self.binary_model.load_state_dict(binary_checkpoint)
                print(f"Modelo Binário carregado de: {binary_path}\n")
        except Exception as e:
            print(f"Erro ao carregar modelo binário: {e}\n")
            raise

        try:
            multiclass_checkpoint = torch.load(multiclass_path, map_location=self.device)
            if 'model_state_dict' in multiclass_checkpoint:
                self.multiclass_model.load_state_dict(multiclass_checkpoint['model_state_dict'])
                print(f"\nModelo Multiclasse carregado de: {multiclass_path}")
                if 'metrics' in multiclass_checkpoint:
                    metrics = multiclass_checkpoint['metrics']
                    print(f"  Métricas do checkpoint:")
                    print(f"    F1-Score: {metrics.get('f1_score', 0) * 100:.2f}%")
                    print(f"    Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%\n")
            else:
                self.multiclass_model.load_state_dict(multiclass_checkpoint)
                print(f"\nModelo Multiclasse carregado de: {multiclass_path}\n")
        except Exception as e:
            print(f"Erro ao carregar modelo multiclasse: {e}\n")
            raise

        self.binary_model.eval()
        self.multiclass_model.eval()

        print(f"\n{'-' * 60}")
        print("MODELOS CARREGADOS E PRONTOS PARA INFERÊNCIA")
        print(f"{'-' * 60}\n")

    def get_model_info(self) -> Dict:
        return {
            'device': str(self.device),
            'binary_model': {
                'num_classes': self.binary_num_classes,
                'class_names': self.binary_class_names,
                'checkpoint_path': self.binary_config['checkpoint']['save_path']
            },
            'multiclass_model': {
                'num_classes': self.multiclass_num_classes,
                'class_names': self.multiclass_class_names,
                'checkpoint_path': self.multiclass_config['checkpoint']['save_path']
            },
            'training_config': {
                'binary': {
                    'epochs': self.binary_config['training']['epochs'],
                    'patience': self.binary_config['training']['patience']
                },
                'multiclass': {
                    'epochs': self.multiclass_config['training']['epochs'],
                    'patience': self.multiclass_config['training']['patience']
                }
            }
        }

    def print_prediction(self, result: Dict):
        print(f"\n{'-' * 60}")
        print("RESULTADO DA PREDIÇÃO")
        print(f"{'-' * 60}\n")

        print("Classificação Binária:")
        binary_pred = result['binary_prediction']
        print(f"  Classe: {binary_pred['class_name']}")
        print(f"  Confiança: {binary_pred['confidence'] * 100:.2f}%")
        print(f"  Probabilidades:")
        for class_name, prob in binary_pred['probabilities'].items():
            print(f"    {class_name}: {prob * 100:.2f}%")

        if result['requires_multiclass']:
            print(f"\nClassificação Multiclasse:")
            multiclass_pred = result['multiclass_prediction']
            print(f"  Classe: {multiclass_pred['class_name']}")
            print(f"  Confiança: {multiclass_pred['confidence'] * 100:.2f}%")
            print(f"  Probabilidades:")
            for class_name, prob in multiclass_pred['probabilities'].items():
                print(f"    {class_name}: {prob * 100:.2f}%")

        print(f"\n{'─' * 60}")
        print(f"PREDIÇÃO FINAL: {result['final_prediction']}")
        print(f"CONFIANÇA: {result['final_confidence'] * 100:.2f}%")
        print(f"{'-' * 60}\n")