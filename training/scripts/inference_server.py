import os
import shutil
import tempfile
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path

from src.models.inference import InferenceWrapper

BASE_DIR = Path(__file__).resolve().parent.parent
SHARED_DIR = BASE_DIR.parent / 'shared'

wrapper = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global wrapper
    print("Iniciando Microsserviço de Inferência (Motor PyTorch)...")
    try:
        wrapper = InferenceWrapper()
        wrapper.load_models()
        print("Modelos do InferenceWrapper carregados com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelos na inferência: {e}")
    yield
    print("Desligando Microsserviço de Inferência...")
    wrapper = None

app = FastAPI(title="Motor de Inferência Interno", lifespan=lifespan)

@app.get("/")
def health_check():
    return {"status": "online", "models_loaded": wrapper is not None and len(wrapper.binary_models) > 0}

@app.post("/internal_predict")
async def internal_predict(
    file: UploadFile = File(...),
    subject_id: str = Form(...)
):
    if not wrapper or not wrapper.binary_models:
        raise HTTPException(status_code=503, detail="Modelos não carregados no backend.")
        
    try:
        import io
        image_bytes = await file.read()
        image_stream = io.BytesIO(image_bytes)
            
        img_tensor = wrapper.load_image(image_stream)
        result = wrapper.predict_tensor(img_tensor)
        
        gradcam_dir = str(SHARED_DIR / f"gradcam_outputs/{subject_id}")
        gradcam_tensor = img_tensor.to(wrapper.device).squeeze(0)
        pred_class = result['final_prediction']
        confidence = result['multiclass_prediction']['confidence'] if result['requires_multiclass'] else result['binary_prediction']['confidence']
        binary_conf = result['binary_prediction']['probabilities'].get('Demented') if result['requires_multiclass'] else None
        
        wrapper.generate_gradcam(
            image_path=None,
            save_dir=gradcam_dir,
            subject_name=subject_id,
            slice_index=1,
            requires_multiclass=result['requires_multiclass'],
            img_tensor=gradcam_tensor,
            predicted_class_name=pred_class,
            confidence=confidence,
            binary_confidence=binary_conf
        )
        
        return JSONResponse({
            "result": result,
            "gradcam_saved_paths": [f"{gradcam_dir}/{subject_id}_slice_01.png"]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/internal_predict_subject")
async def internal_predict_subject(
    files: List[UploadFile],
    subject_id: str = Form(...),
    gradcam_samples: int = Form(1)
):
    if not wrapper or not wrapper.binary_models:
        raise HTTPException(status_code=503, detail="Modelos não carregados no backend.")
        
    try:
        import io
        from src.data.subject_manager import extract_slice_index
        
        file_names = []
        tensors = []
        
        for file in files:
            content = await file.read()
            stream = io.BytesIO(content)
            tensor = wrapper.load_image(stream)
            tensors.append(tensor)
            file_names.append(file.filename)

        subject_tensor = torch.cat(tensors, dim=0)
        result = wrapper.predict_tensor(subject_tensor)
        
        gradcam_saved_paths = []
        if gradcam_samples > 0 and file_names:
            from src.data.subject_manager import get_central_elements
            paired = list(zip(file_names, tensors))
            paired.sort(key=lambda x: extract_slice_index(x[0]))
            
            central_pairs = get_central_elements(paired, gradcam_samples)
            gradcam_dir = str(SHARED_DIR / f"gradcam_outputs/{subject_id}")
            pred_class = result['final_prediction']
            confidence = result['multiclass_prediction']['confidence'] if result['requires_multiclass'] else result['binary_prediction']['confidence']
            binary_conf = result['binary_prediction']['probabilities'].get('Demented') if result['requires_multiclass'] else None
            
            for i, (fname, tensor) in enumerate(central_pairs):
                gradcam_tensor = tensor.to(wrapper.device).squeeze(0)
                wrapper.generate_gradcam(
                    image_path=None, 
                    save_dir=gradcam_dir,
                    subject_name=subject_id,
                    slice_index=i + 1,
                    requires_multiclass=result['requires_multiclass'],
                    img_tensor=gradcam_tensor,
                    predicted_class_name=pred_class,
                    confidence=confidence,
                    binary_confidence=binary_conf
                )
                gradcam_saved_paths.append(f"{gradcam_dir}/{subject_id}_slice_{i+1:02d}.png")

        return JSONResponse({
            "result": result,
            "gradcam_saved_paths": gradcam_saved_paths
        })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("scripts.inference_server:app", host="0.0.0.0", port=8080)
