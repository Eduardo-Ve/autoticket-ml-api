from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json
import numpy as np

app = FastAPI()

model = joblib.load("models/category_pipeline.joblib")
with open("models/meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

THRESHOLDS = meta["thresholds"]
DEFAULT_TH = meta.get("default_threshold", 0.60)
REVIEW = meta.get("review_label", "REVIEW")

class TicketRequest(BaseModel):
    description: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(req: TicketRequest):
    text = req.description.strip()
    if not text:
        return {"success": False, "error": "Empty description"}

    proba = model.predict_proba([text])[0]
    classes = model.named_steps["clf"].classes_

    i = int(np.argmax(proba))
    label = str(classes[i])
    conf = float(np.max(proba))

    th = float(THRESHOLDS.get(label, DEFAULT_TH))
    decision = label if conf >= th else REVIEW

    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [(str(classes[j]), float(proba[j])) for j in top3_idx]

    return {
        "success": True,
        "data": {
            "category": decision,
            "category_label": label,
            "confidence": conf,
            "threshold_used": th,
            "top3": top3
        }
    }
