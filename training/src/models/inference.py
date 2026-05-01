import torch
import torch.nn as nn
import os
import glob
from typing import Tuple, Optional, Dict, List, Any
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.utils.config import (
    load_binary_config, 
    load_multiclass_config, 
    load_hyperparameters_config
)
from src.utils.hardware import get_pytorch_device
from .architectures import create_model
from ..data.preprocessing import denormalize_images
from ..evaluation.gradcam import run_single_gradcam

class InferenceWrapper(nn.Module):
    def __init__(self, device: torch.device = None):
        super(InferenceWrapper, self).__init__()

        self.binary_config = load_binary_config()
        self.multiclass_config = load_multiclass_config()
        self.hyperparams_config = load_hyperparameters_config()

        self.binary_class_names = self.binary_config['model']['class_names']
        self.multiclass_class_names = self.multiclass_config['model']['class_names']
        self.binary_num_classes = self.binary_config['model']['num_classes']
        self.multiclass_num_classes = self.multiclass_config['model']['num_classes']

        if device is None:
            device = get_pytorch_device()
        self.device = device
        
        # Agora são listas de modelos (Ensemble)
        self.binary_models = []
        self.multiclass_models = []
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226])
        ])

    def load_models(self):
        print(f"\n{'=' * 60}\nINICIALIZANDO SISTEMA DE INFERÊNCIA EM CASCATA (ENSEMBLE)\n{'=' * 60}")

        # Resolve diretórios de checkpoints
        binary_dir = self._resolve_checkpoint_dir(self.binary_config)
        multiclass_dir = self._resolve_checkpoint_dir(self.multiclass_config)

        # Carrega Ensembles
        self.binary_models = self._load_ensemble(binary_dir, 'binary')
        self.multiclass_models = self._load_ensemble(multiclass_dir, 'multiclass')

        print(f"\n{'=' * 60}\nSISTEMA ENSEMBLE PRONTO PARA PREDIÇÃO\n{'=' * 60}")

    def predict_image(self, image_path: str) -> Dict:
        img_tensor = self._load_and_preprocess_image(image_path)
        return self._predict(img_tensor)

    def predict_subject_folder(self, folder_path: str) -> Dict:
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Pasta não encontrada: {folder_path}")
            
        file_list = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not file_list:
            raise ValueError(f"Nenhuma imagem encontrada na pasta: {folder_path}")
            
        tensors = [self._load_and_preprocess_image(p) for p in file_list]
        subject_tensor = torch.cat(tensors, dim=0)
        
        return self._predict_subject(subject_tensor)

    def _load_and_preprocess_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('L')
        return self.transform(img).unsqueeze(0)

    def _predict(self, x: torch.Tensor) -> Dict:
        x = x.to(self.device)
        binary_probs, multiclass_probs = self._forward_ensemble(x)

        # Pegamos a média se houver múltiplas fatias (aqui é só 1)
        return self._format_prediction_result(
            binary_probs=binary_probs[0],
            multiclass_probs=multiclass_probs[0] if multiclass_probs is not None else None
        )

    def _predict_subject(self, x: torch.Tensor) -> Dict:
        x = x.to(self.device)
        binary_probs, multiclass_probs = self._forward_ensemble(x)

        # Média das probabilidades de todas as fatias do sujeito
        binary_probs_avg = binary_probs.mean(dim=0)
        multiclass_probs_avg = None
        if multiclass_probs is not None:
            multiclass_probs_avg = multiclass_probs.mean(dim=0)

        return self._format_prediction_result(
            binary_probs=binary_probs_avg,
            multiclass_probs=multiclass_probs_avg
        )

    def _forward_ensemble(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.binary_models or not self.multiclass_models:
            raise RuntimeError("Modelos não carregados. Chame load_models() primeiro.")

        with torch.no_grad():
            # 1. Predição Binária Ensemble (Soft Voting)
            all_bin_probs = []
            for model in self.binary_models:
                output = model(x)
                all_bin_probs.append(torch.softmax(output, dim=1))
            
            avg_binary_probs = torch.stack(all_bin_probs).mean(dim=0)
            binary_pred = torch.argmax(avg_binary_probs, dim=1)

            # 2. Predição Multiclasse Ensemble (se necessário)
            avg_multiclass_probs = None
            if torch.any(binary_pred == 0): # Se algum foi predito como demente
                all_multi_probs = []
                for model in self.multiclass_models:
                    output = model(x)
                    all_multi_probs.append(torch.softmax(output, dim=1))
                avg_multiclass_probs = torch.stack(all_multi_probs).mean(dim=0)

            return avg_binary_probs, avg_multiclass_probs

    def _format_prediction_result(self, binary_probs: torch.Tensor, multiclass_probs: Optional[torch.Tensor]) -> Dict:
        binary_class_idx = torch.argmax(binary_probs).item()
        binary_class_name = self.binary_class_names[binary_class_idx]

        result = {
            'binary_prediction': {
                'class_name': binary_class_name,
                'confidence': binary_probs[binary_class_idx].item(),
                'probabilities': {self.binary_class_names[i]: binary_probs[i].item() for i in range(self.binary_num_classes)}
            },
            'multiclass_prediction': None,
            'final_prediction': binary_class_name,
            'requires_multiclass': False
        }

        if binary_class_idx == 0 and multiclass_probs is not None:
            multiclass_class_idx = torch.argmax(multiclass_probs).item()
            multiclass_class_name = self.multiclass_class_names[multiclass_class_idx]

            result['multiclass_prediction'] = {
                'class_name': multiclass_class_name,
                'confidence': multiclass_probs[multiclass_class_idx].item(),
                'probabilities': {self.multiclass_class_names[i]: multiclass_probs[i].item() for i in range(self.multiclass_num_classes)}
            }
            result['final_prediction'] = multiclass_class_name
            result['requires_multiclass'] = True

        return result

    def _load_ensemble(self, directory: str, model_type: str) -> List[nn.Module]:
        ckpt_files = sorted(glob.glob(os.path.join(directory, "best_model_fold_*.pth")))
        if not ckpt_files:
            raise FileNotFoundError(f"Nenhum checkpoint de fold encontrado em {directory}")
            
        print(f"Carregando Ensemble {model_type.upper()} ({len(ckpt_files)} modelos)...")
        models = []
        for path in ckpt_files:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            arch_name = checkpoint['architecture_name']
            hparams = checkpoint['hyperparameters']
            
            model = create_model(
                architecture_name=arch_name,
                hidden_units=hparams['hidden_units'],
                dropout=hparams['dropout'],
                num_classes=checkpoint['num_classes'],
                device=self.device,
                verbose=False
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model)
        return models

    def _resolve_checkpoint_dir(self, config: Dict) -> str:
        return os.path.normpath(os.path.join(os.path.dirname(__file__), str(config['checkpoint']['save_path'])))

    def generate_gradcam(self, image_path: str, save_dir: str, subject_name: str, slice_index: int, requires_multiclass: bool):
        if not self.binary_models: raise RuntimeError("Modelos não carregados.")
        os.makedirs(save_dir, exist_ok=True)
        
        img_tensor = self._load_and_preprocess_image(image_path).to(self.device).squeeze(0)
        file_name = f"{subject_name}_slice_{slice_index:02d}.png"
        
        # No ensemble, o Grad-CAM é gerado usando todos os modelos para criar o mapa consensual
        target_models = self.multiclass_models if (requires_multiclass and self.multiclass_models) else self.binary_models
        class_names = self.multiclass_class_names if (requires_multiclass and self.multiclass_models) else self.binary_class_names

        from ..evaluation.gradcam import run_ensemble_gradcam
        run_ensemble_gradcam(
            models=target_models, img_tensor=img_tensor, device=self.device,
            class_names=class_names, save_path=os.path.join(save_dir, file_name)
        )

    def print_prediction(self, result: Dict):
        print(f"\n[{'=' * 60}]")
        print("RESULTADO DO DIAGNÓSTICO CLÍNICO (ENSEMBLE)")
        print(f"[{'=' * 60}]\n")
        
        print("1. Avaliação Primária (Binária Ensemble):")
        bin_pred = result['binary_prediction']
        print(f"   => Classe: {bin_pred['class_name']}")
        print(f"   => Confiança: {bin_pred['confidence'] * 100:.2f}%")
        
        if result['requires_multiclass']:
            print("\n2. Avaliação Secundária (Multiclasse Ensemble - Estagiamento):")
            multi_pred = result['multiclass_prediction']
            print(f"   => Estágio: {multi_pred['class_name']}")
            print(f"   => Confiança: {multi_pred['confidence'] * 100:.2f}%")
        
        print(f"\n{'-' * 60}")
        print(f"DIAGNÓSTICO FINAL: >> {result['final_prediction'].upper()} <<")
        print(f"{'-' * 60}\n")
