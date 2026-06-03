import os
import httpx
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Alzheimer Cascade Classifier API (Gateway)",
    description="API REST Gateway (Leve) para o sistema Alzheimer (Sem PyTorch)",
    version="1.0.0"
)

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://ia-alzheimer-inference:8080")

@app.get("/")
def read_root():
    return {"status": "online", "mode": "gateway", "inference_url": INFERENCE_URL}

@app.post("/predict")
async def predict_mri(
    file: UploadFile = File(...),
    subject_id: str = Form(..., description="Identificador único ou nome do paciente")
):
    try:
        image_bytes = await file.read()
        
        files = {'file': (file.filename, image_bytes, file.content_type)}
        data = {'subject_id': subject_id}
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{INFERENCE_URL}/internal_predict", files=files, data=data)
            
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
        json_data = response.json()
        result = json_data.get('result', {})
        gradcam_saved_paths = json_data.get('gradcam_saved_paths', [])
        
        return JSONResponse({
            "status": "success",
            "diagnostico_final": result.get('final_prediction'),
            "detalhes": {
                "binario": result.get('binary_prediction'),
                "multiclasse": result.get('multiclass_prediction'),
                "precisou_estagiamento": result.get('requires_multiclass')
            },
            "gradcam_saved_paths": gradcam_saved_paths
        })
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Erro de conexão com o motor de inferência: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")

@app.post("/predict_subject")
async def predict_subject_mri(
    files: List[UploadFile],
    subject_id: str = Form(..., description="Identificador único ou nome do paciente"),
    gradcam_samples: int = Form(1, description="Quantidade de fatias centrais para gerar o mapa de calor")
):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhuma imagem enviada.")
        
    try:
        files_data = []
        for file in files:
            content = await file.read()
            files_data.append(('files', (file.filename, content, file.content_type)))
            
        data = {
            'subject_id': subject_id,
            'gradcam_samples': str(gradcam_samples)
        }
            
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{INFERENCE_URL}/internal_predict_subject", files=files_data, data=data)
            
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
        json_data = response.json()
        result = json_data.get('result', {})
        gradcam_saved_paths = json_data.get('gradcam_saved_paths', [])
        
        return JSONResponse({
            "status": "success",
            "paciente_fatias_analisadas": len(files),
            "diagnostico_final": result.get('final_prediction'),
            "detalhes": {
                "binario": result.get('binary_prediction'),
                "multiclasse": result.get('multiclass_prediction'),
                "precisou_estagiamento": result.get('requires_multiclass')
            },
            "gradcam_saved_paths": gradcam_saved_paths
        })
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Erro de conexão com o motor de inferência: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento do paciente: {str(e)}")
