from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

queue_model = joblib.load("models/queue_pipeline.joblib")
prio_model  = joblib.load("models/priority_pipeline.joblib")

TH_QUEUE = 0.65
TH_PRIO  = 0.60

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

    # QUEUE
    q_proba = queue_model.predict_proba([text])[0]
    q_classes = queue_model.named_steps["clf"].classes_
    qi = int(np.argmax(q_proba))
    q_label = str(q_classes[qi])
    q_conf = float(np.max(q_proba))
    q_decision = q_label if q_conf >= TH_QUEUE else "REVIEW_QUEUE"

    # PRIORITY
    p_proba = prio_model.predict_proba([text])[0]
    p_classes = prio_model.named_steps["clf"].classes_
    pi = int(np.argmax(p_proba))
    p_label = str(p_classes[pi])
    p_conf = float(np.max(p_proba))
    p_decision = p_label if p_conf >= TH_PRIO else "REVIEW_PRIORITY"

    return {
        "success": True,
        "data": {
            "queue": q_decision,
            "queue_confidence": q_conf,
            "priority": p_decision,
            "priority_confidence": p_conf,
        }
    }
