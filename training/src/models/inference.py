import torch
import torch.nn as nn
import os
import glob
import io
from typing import Tuple, Optional, Dict, List, Any, Union
from pathlib import Path
from PIL import Image
from torchvision import transforms
import base64
import cv2

from src.utils.config import (
    load_binary_config, 
    load_multiclass_config, 
    load_hyperparameters_config,
    MODELS_PATH
)
from src.utils.hardware import get_pytorch_device
from .architectures import create_model
from src.data.preprocessing import MedicalImagePreprocessor
from src.evaluation.gradcam import generate_ensemble_gradcam_image, run_ensemble_gradcam

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
        
        self.binary_models = []
        self.multiclass_models = []
        
        self.binary_transform = None
        self.multiclass_transform = None

    def load_models(self):
        print(f"\n{'=' * 60}\nINICIALIZANDO SISTEMA DE INFERÊNCIA EM CASCATA (ENSEMBLE)\n{'=' * 60}")

        binary_dir = self._resolve_checkpoint_dir(self.binary_config)
        multiclass_dir = self._resolve_checkpoint_dir(self.multiclass_config)

        self.binary_models = self._load_ensemble(binary_dir, 'binary')
        self.multiclass_models = self._load_ensemble(multiclass_dir, 'multiclass')

        if self.binary_models:
            bin_arch = self.binary_models[0].architecture_name
            bin_preprocessor = MedicalImagePreprocessor(bin_arch)
            self.binary_transform = transforms.Compose([
                transforms.Resize((bin_preprocessor.get_image_size(), bin_preprocessor.get_image_size())),
                transforms.Normalize(mean=bin_preprocessor.config["mean"], std=bin_preprocessor.config["std"])
            ])
            
        if self.multiclass_models:
            multi_arch = self.multiclass_models[0].architecture_name
            multi_preprocessor = MedicalImagePreprocessor(multi_arch)
            self.multiclass_transform = transforms.Compose([
                transforms.Resize((multi_preprocessor.get_image_size(), multi_preprocessor.get_image_size())),
                transforms.Normalize(mean=multi_preprocessor.config["mean"], std=multi_preprocessor.config["std"])
            ])

        print(f"\n{'=' * 60}\nSISTEMA ENSEMBLE PRONTO PARA PREDIÇÃO\n{'=' * 60}")

    def predict_image(self, image_path: str) -> Dict:
        img_tensor = self.load_image(image_path)
        return self.predict_tensor(img_tensor)

    def predict_subject_folder(self, folder_path: str) -> Dict:
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Pasta não encontrada: {folder_path}")
            
        file_list = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not file_list:
            raise ValueError(f"Nenhuma imagem encontrada na pasta: {folder_path}")
            
        tensors = [self.load_image(p) for p in file_list]
        subject_tensor = torch.cat(tensors, dim=0)
        
        return self.predict_tensor(subject_tensor)

    def load_image(self, path_or_stream: Union[str, io.BytesIO]) -> torch.Tensor:
        img = Image.open(path_or_stream).convert('L')
        return transforms.ToTensor()(img).unsqueeze(0)

    def predict_tensor(self, x: torch.Tensor) -> Dict:
        x = x.to(self.device)
        binary_probs, multiclass_probs = self._forward_ensemble(x)

        binary_probs_avg = binary_probs.mean(dim=0)
        multiclass_probs_avg = None
        if multiclass_probs is not None:
            multiclass_probs_avg = multiclass_probs.mean(dim=0)

        return self._format_prediction_result(
            binary_probs=binary_probs_avg,
            multiclass_probs=multiclass_probs_avg
        )

    def _forward_ensemble(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._ensure_models_loaded()

        with torch.no_grad():
            x_bin = self.binary_transform(x)
            
            all_bin_probs = []
            for model in self.binary_models:
                output = model(x_bin)
                all_bin_probs.append(torch.softmax(output, dim=1))
            
            avg_binary_probs = torch.stack(all_bin_probs).mean(dim=0)
            subject_binary_probs = avg_binary_probs.mean(dim=0)
            subject_binary_pred = torch.argmax(subject_binary_probs).item()

            demented_idx = self.binary_class_names.index("Demented")
            avg_multiclass_probs = None
            if subject_binary_pred == demented_idx:
                x_multi = self.multiclass_transform(x)
                
                all_multi_probs = []
                for model in self.multiclass_models:
                    output = model(x_multi)
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

        demented_idx = self.binary_class_names.index("Demented")
        if binary_class_idx == demented_idx and multiclass_probs is not None:
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
            model.architecture_name = arch_name # Importante para o Grad-CAM encontrar a camada certa
            model.eval()
            models.append(model)
        return models

    def _resolve_checkpoint_dir(self, config: Dict) -> str:
        model_type_dir = os.path.basename(os.path.normpath(config['checkpoint']['save_path']))
        return str(MODELS_PATH / model_type_dir)

    def _ensure_models_loaded(self):
        if not self.binary_models or not self.multiclass_models:
            raise RuntimeError("Modelos não carregados. Chame load_models() primeiro.")

    def _get_target_models_and_classes(self, requires_multiclass: bool):
        if requires_multiclass and self.multiclass_models:
            return self.multiclass_models, self.multiclass_class_names
        return self.binary_models, self.binary_class_names

    def _prepare_gradcam_input(self, image_path: Optional[str], img_tensor: Optional[torch.Tensor], requires_multiclass: bool) -> Tuple[List[nn.Module], List[str], torch.Tensor]:
        self._ensure_models_loaded()
        
        if img_tensor is None:
            img_tensor = self.load_image(image_path).squeeze(0)
            
        if img_tensor.ndim == 4:
            img_tensor = img_tensor.squeeze(0)
        
        target_models, class_names = self._get_target_models_and_classes(requires_multiclass)
        transform = self.multiclass_transform if requires_multiclass else self.binary_transform
        
        img_tensor_transformed = transform(img_tensor).to(self.device)
        return target_models, class_names, img_tensor_transformed

    def generate_gradcam(self, image_path: Optional[str], save_dir: str, subject_name: str, slice_index: int, requires_multiclass: bool, img_tensor: torch.Tensor = None):
        target_models, class_names, img_tensor_transformed = self._prepare_gradcam_input(image_path, img_tensor, requires_multiclass)
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{subject_name}_slice_{slice_index:02d}.png"

        run_ensemble_gradcam(
            models=target_models, img_tensor=img_tensor_transformed, device=self.device,
            class_names=class_names, save_path=os.path.join(save_dir, file_name)
        )

    def get_gradcam_base64(self, image_path: Optional[str], requires_multiclass: bool, img_tensor: torch.Tensor = None) -> Optional[str]: 
        try:
            target_models, _, img_tensor_transformed = self._prepare_gradcam_input(image_path, img_tensor, requires_multiclass)
            
            visualization = generate_ensemble_gradcam_image(
                models=target_models, 
                img_tensor=img_tensor_transformed, 
                device=self.device
            )
            
            vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.png', vis_bgr)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Erro ao gerar Grad-CAM em base64: {e}")
            return None

    def print_prediction(self, result: Dict):
        print(f"\n{'=' * 60}")
        print(f"RESULTADO DO DIAGNÓSTICO (PREDIÇÃO)")
        print(f"{'=' * 60}")
        print(f"Predição Final: {result['final_prediction']}")
        
        print(f"\n1. Classificação Binária:")
        bin_pred = result['binary_prediction']
        print(f"   Classe: {bin_pred['class_name']} (Confiança: {bin_pred['confidence']*100:.2f}%)")
        print(f"   Probabilidades:")
        for name, prob in bin_pred['probabilities'].items():
            print(f"     - {name}: {prob*100:.2f}%")
            
        if result['requires_multiclass'] and result['multiclass_prediction']:
            print(f"\n2. Classificação Multiclasse:")
            multi_pred = result['multiclass_prediction']
            print(f"   Classe: {multi_pred['class_name']} (Confiança: {multi_pred['confidence']*100:.2f}%)")
            print(f"   Probabilidades:")
            for name, prob in multi_pred['probabilities'].items():
                print(f"     - {name}: {prob*100:.2f}%")
        print(f"{'=' * 60}\n")