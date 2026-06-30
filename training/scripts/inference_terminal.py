import os
import argparse
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.inference import InferenceWrapper
from src.visualization.terminal import print_banner, print_section
from src.utils.hardware import get_pytorch_device
from src.data.subject_manager import extract_slice_index

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = str(BASE_DIR.parent / 'shared/gradcam_outputs')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ferramenta de Diagnóstico Alzheimer (Produção)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Exemplos de uso:
                # Diagnóstico de uma única imagem (fatia) com Grad-CAM
                python scripts/inference.py --image "shared/data/test_sample.jpg" --gradcam

                # Diagnóstico de um paciente completo (pasta com várias fatias)
                python scripts/inference.py --subject "shared/data/splits/binary/test/Demented/subject_123"

                # Forçar uso de CPU
                python scripts/inference.py --image "img.jpg" --cpu
               """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--image',
        type=str,
        help="Caminho para uma única imagem JPG/PNG"
    )
    group.add_argument(
        '--subject',
        type=str,
        help="Caminho para uma pasta contendo fatias de um sujeito"
    )


    parser.add_argument(
        '--output_path',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Caminho para salvar resultados/Grad-CAM (padrão: {DEFAULT_OUTPUT_PATH})"
    )

    parser.add_argument(
        '--gradcam',
        action='store_true',
        help="Gerar visualizações Grad-CAM para explicar o diagnóstico"
    )

    parser.add_argument(
        '--gradcam-samples',
        type=int,
        default=1,
        help="Quantidade de imagens centrais para gerar Grad-CAM ao avaliar uma pasta inteira (padrão: 1)"
    )

    parser.add_argument(
        '--cpu',
        action='store_true',
        help="Forçar execução em CPU"
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print_banner("SISTEMA DE DIAGNÓSTICO ALZHEIMER", "IA de Produção - Diagnóstico em Cascata")

    device = torch.device('cpu') if args.cpu else get_pytorch_device()
    print(f"Executando em: {device}\n")

    try:
        print_section("CARREGANDO MODELOS")
        evaluator = InferenceWrapper(device=device)
        evaluator.load_models()

        print_section("PROCESSANDO DIAGNÓSTICO")
        
        if args.image:
            print(f"Alvo: Imagem Única -> {os.path.basename(args.image)}")
            img_tensor = evaluator.load_image(args.image)
            result = evaluator.predict_tensor(img_tensor)

            if args.gradcam:
                subject_name = Path(args.image).stem
                gradcam_tensor = img_tensor.to(evaluator.device).squeeze(0)
                pred_class = result['final_prediction']
                confidence = result['multiclass_prediction']['confidence'] if result['requires_multiclass'] else result['binary_prediction']['confidence']
                binary_conf = result['binary_prediction']['probabilities'].get('Demented') if result['requires_multiclass'] else None
                
                evaluator.generate_gradcam(
                    args.image, 
                    args.output_path, 
                    subject_name, 
                    slice_index=1, 
                    requires_multiclass=result['requires_multiclass'],
                    img_tensor=gradcam_tensor,
                    predicted_class_name=pred_class,
                    confidence=confidence,
                    binary_confidence=binary_conf
                )
        else:
            print(f"Alvo: Sujeito Completo -> {os.path.basename(os.path.normpath(args.subject))}")
            result = evaluator.predict_subject_folder(args.subject)
            
            if args.gradcam:
                files = [f for f in os.listdir(args.subject) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    from src.data.subject_manager import get_central_elements
                    files.sort(key=lambda fname: extract_slice_index(fname))
                    
                    central_images = get_central_elements(files, args.gradcam_samples)
                    subject_name = os.path.basename(os.path.normpath(args.subject))
                    pred_class = result['final_prediction']
                    confidence = result['multiclass_prediction']['confidence'] if result['requires_multiclass'] else result['binary_prediction']['confidence']
                    binary_conf = result['binary_prediction']['probabilities'].get('Demented') if result['requires_multiclass'] else None
                    
                    print(f"Gerando Grad-CAM para {len(central_images)} fatia(s) central(is)...")
                    for i, img_file in enumerate(central_images):
                        evaluator.generate_gradcam(
                            os.path.join(args.subject, img_file), 
                            args.output_path,
                            subject_name=subject_name,
                            slice_index=i + 1,
                            requires_multiclass=result['requires_multiclass'],
                            predicted_class_name=pred_class,
                            confidence=confidence,
                            binary_confidence=binary_conf
                        )

        evaluator.print_prediction(result)
        
        print_section("CONCLUÍDO")
        if args.gradcam:
            print(f"Visualizações Grad-CAM disponíveis em: {args.output_path}\n")
        
    except FileNotFoundError as e:
        print(f"\n[ERRO] Arquivo ou diretório não encontrado: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERRO CRÍTICO] Ocorreu uma falha durante o diagnóstico:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()