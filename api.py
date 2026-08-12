from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np, torch
from train import Classifier

app = FastAPI(title="GPU Fault Classifier", version="0.1.0")
bundle = torch.load("artifacts/model.pt", map_location="cpu", weights_only=False); model = Classifier(); model.load_state_dict(bundle["state_dict"]); model.eval()

class Observation(BaseModel):
    temperature: float = Field(55, ge=-20, le=150); memory_used_pct: float = Field(45, ge=0, le=100)
    ecc_errors: int = Field(0, ge=0); xid_errors: int = Field(0, ge=0); pod_ready: int = Field(1, ge=0, le=1)
    driver_ok: int = Field(1, ge=0, le=1); cuda_visible: int = Field(1, ge=0, le=1); node_ready: int = Field(1, ge=0, le=1)

@app.get("/health")
def health(): return {"status": "ok", "model": "gpu-fault-classifier"}

@app.post("/predict")
def predict(obs: Observation):
    x = np.array([[getattr(obs, f) for f in bundle["features"]]], dtype="float32"); x = ((x - bundle["mean"]) / bundle["scale"]).astype("float32")
    with torch.no_grad(): probs = torch.softmax(model(torch.tensor(x)), 1)[0].numpy()
    order = probs.argsort()[::-1]
    return {"label": bundle["labels"][int(order[0])], "confidence": float(probs[order[0]]), "probabilities": {bundle["labels"][int(i)]: float(probs[i]) for i in order}}
