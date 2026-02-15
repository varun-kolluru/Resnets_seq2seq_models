from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64

from ml_models.resnet_models.resnet_inference import infer_resnet
from ml_models.seq2seq_models.inference import translate_sentence, load_model


app = FastAPI(title="ResNet Interaction API")

# ✅ CORS middleware with proper preflight handling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model - make image optional for preflight
class InferRequest(BaseModel):
    model_name: str
    attack_type: str
    noise_value: float
    image: str

MODELS = {}

class TranslateRequest(BaseModel):
    model: str
    text: str

@app.post("/api/translate")
def translate(req: TranslateRequest):
    if req.model not in MODELS:
        MODELS[req.model] = load_model(req.model)

    translation = translate_sentence(
        req.text,
        MODELS[req.model],
        req.model
    )

    return {"translation": translation}

# Handle OPTIONS preflight requests
@app.options("/infer")
async def preflight():
    return {}

# Dummy inference
@app.post("/infer")
async def infer(req: InferRequest):
    if not req.image:
        return {"error": "Image is required"}, 400
    
    try:
        return infer_resnet(
        model_name=req.model_name,
        image_b64=req.image,
        attack_type=req.attack_type,
        noise_value=req.noise_value
        )
    except Exception as e:
        return {"error": str(e)}, 400


@app.get("/")
async def root():
    return {"message": "ResNet API is running"}