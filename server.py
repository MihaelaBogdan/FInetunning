import os
import json
import queue
import threading
import torch
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Importuri din proiectul existent
from config import (
    BASE_MODEL, OUTPUT_DIR, LORA_R, LORA_ALPHA,
    TRAIN_SAMPLES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, FULL_LR
)
from data_utils import load_tokenizer
from train_lora import train_lora, get_device
from train_full import train_full
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

app = FastAPI(title="LoRA vs Full Fine-Tuning Dashboard")

# ─── State Global ─────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.is_training = False
        self.current_method = None  # "lora" | "full" | None
        self.current_progress = {}
        self.config = {
            "train_samples": TRAIN_SAMPLES,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate_lora": LEARNING_RATE,
            "learning_rate_full": FULL_LR,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
        }
        # Modele cache
        self.tokenizer = None
        self.base_model = None
        self.lora_model = None
        self.full_model = None

state = AppState()

# ─── Broadcaster pentru Progres SSE ───────────────────────────────────────────
class ProgressBroadcaster:
    def __init__(self):
        self.listeners = []

    def add_listener(self):
        q = queue.Queue()
        self.listeners.append(q)
        return q

    def remove_listener(self, q):
        if q in self.listeners:
            self.listeners.remove(q)

    def broadcast(self, data):
        for q in self.listeners:
            q.put(data)

broadcaster = ProgressBroadcaster()

# ─── Helperi Antrenament ──────────────────────────────────────────────────────
def run_training_thread(method: str, params: dict):
    state.is_training = True
    state.current_method = method
    state.current_progress = {"status": "started", "progress_pct": 0}
    
    # Trimitem starea inițială
    broadcaster.broadcast({"type": "status", "status": "started", "method": method})

    def progress_callback(info):
        state.current_progress = info
        broadcaster.broadcast({"type": "progress", "method": method, "data": info})

    try:
        if method == "lora":
            train_lora(
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                learning_rate=params["learning_rate_lora"],
                train_samples=params["train_samples"],
                lora_r=params["lora_r"],
                lora_alpha=params["lora_alpha"],
                progress_callback=progress_callback
            )
            # Resetăm modelul LoRA cached pentru a-l reîncărca la următoarea predicție
            state.lora_model = None
        elif method == "full":
            train_full(
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                learning_rate=params["learning_rate_full"],
                train_samples=params["train_samples"],
                progress_callback=progress_callback
            )
            # Resetăm modelul Full cached
            state.full_model = None
            
        state.current_progress = {"status": "completed", "progress_pct": 100}
        broadcaster.broadcast({"type": "status", "status": "completed", "method": method})
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Eroare în timpul antrenamentului: {error_msg}")
        traceback.print_exc()
        state.current_progress = {"status": "failed", "error": error_msg}
        broadcaster.broadcast({"type": "status", "status": "failed", "method": method, "error": error_msg})
    finally:
        state.is_training = False
        state.current_method = None

# ─── Endpoints API ────────────────────────────────────────────────────────────

class ConfigModel(BaseModel):
    train_samples: int
    epochs: int
    batch_size: int
    learning_rate_lora: float
    learning_rate_full: float
    lora_r: int
    lora_alpha: int

@app.get("/api/config")
def get_config():
    return state.config

@app.post("/api/config")
def update_config(new_config: ConfigModel):
    if state.is_training:
        raise HTTPException(status_code=400, detail="Nu se poate modifica configurația în timpul antrenamentului.")
    state.config = new_config.dict()
    return {"status": "success", "config": state.config}

@app.post("/api/train/{method}")
def start_train(method: str, background_tasks: BackgroundTasks):
    if method not in ["lora", "full"]:
        raise HTTPException(status_code=400, detail="Metodă invalidă. Alege 'lora' sau 'full'.")
    if state.is_training:
        raise HTTPException(status_code=400, detail="Un antrenament este deja în desfășurare.")
    
    # Pornim antrenamentul într-un thread separat pentru a nu bloca bucla de evenimente FastAPI
    t = threading.Thread(target=run_training_thread, args=(method, state.config))
    t.start()
    return {"status": "started", "method": method}

@app.get("/api/status")
def get_status():
    return {
        "is_training": state.is_training,
        "current_method": state.current_method,
        "progress": state.current_progress
    }

@app.get("/api/stream-progress")
def stream_progress():
    q = broadcaster.add_listener()
    def event_generator():
        while True:
            try:
                # Așteptăm date din coadă (cu timeout pentru a detecta deconectarea clientului)
                data = q.get(timeout=2.0)
                yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
                yield ": keep-alive\n\n"
            except GeneratorExit:
                broadcaster.remove_listener(q)
                break
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/metrics")
def get_metrics():
    metrics = {}
    for m in ["lora", "full"]:
        path = os.path.join(OUTPUT_DIR, f"metrics_{m}.json")
        if os.path.exists(path):
            with open(path) as f:
                metrics[m] = json.load(f)
        else:
            metrics[m] = None
    return metrics

class PredictionRequest(BaseModel):
    text: str

def get_sentiment_prediction(model, tokenizer, text, device):
    """Calculează probabilitățile pentru 'positive' și 'negative' folosind T5."""
    model.eval()
    model.to(device)
    
    input_text = f"sentiment: {text}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Prima etapă de decodare: verificăm probabilitățile primului token generat
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits[0, 0, :]  # Shape: (vocab_size,)
        
        # Token ID-urile pentru "positive" și "negative" în Flan-T5
        pos_token_id = tokenizer.encode("positive")[0]
        neg_token_id = tokenizer.encode("negative")[0]
        
        pos_logit = logits[pos_token_id].item()
        neg_logit = logits[neg_token_id].item()
        
        # Softmax local
        max_logit = max(pos_logit, neg_logit)
        exp_pos = torch.exp(torch.tensor(pos_logit - max_logit)).item()
        exp_neg = torch.exp(torch.tensor(neg_logit - max_logit)).item()
        sum_exp = exp_pos + exp_neg
        
        prob_pos = exp_pos / sum_exp
        prob_neg = exp_neg / sum_exp
        
        label = "positive" if prob_pos > prob_neg else "negative"
        confidence = prob_pos if label == "positive" else prob_neg
        
        return label, round(confidence, 4)

@app.post("/api/predict")
def predict(req: PredictionRequest):
    device = get_device()
    
    # Încărcăm leneș (lazy) tokenizer-ul și modelul de bază
    if state.tokenizer is None:
        state.tokenizer = load_tokenizer()
    if state.base_model is None:
        state.base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    
    results = {}
    
    # 1. Predicție Model de Bază (Netrăit)
    try:
        base_label, base_conf = get_sentiment_prediction(state.base_model, state.tokenizer, req.text, device)
        results["base"] = {"label": base_label, "confidence": base_conf, "available": True}
    except Exception as e:
        results["base"] = {"available": False, "error": str(e)}
        
    # 2. Predicție Model LoRA (dacă adaptorul a fost antrenat)
    lora_path = os.path.join(OUTPUT_DIR, "lora", "lora_adapter")
    if os.path.exists(lora_path):
        try:
            if state.lora_model is None:
                # Reîncărcăm modelul de bază curat pe CPU înainte de a aplica adaptori
                clean_base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
                state.lora_model = PeftModel.from_pretrained(clean_base, lora_path)
            
            lora_label, lora_conf = get_sentiment_prediction(state.lora_model, state.tokenizer, req.text, device)
            results["lora"] = {"label": lora_label, "confidence": lora_conf, "available": True}
        except Exception as e:
            results["lora"] = {"available": False, "error": str(e)}
    else:
        results["lora"] = {"available": False, "info": "Modelul LoRA nu a fost încă antrenat."}
        
    # 3. Predicție Model Full Fine-Tuning (dacă modelul a fost antrenat)
    full_path = os.path.join(OUTPUT_DIR, "full", "full_model")
    if os.path.exists(full_path):
        try:
            if state.full_model is None:
                state.full_model = AutoModelForSeq2SeqLM.from_pretrained(full_path)
                
            full_label, full_conf = get_sentiment_prediction(state.full_model, state.tokenizer, req.text, device)
            results["full"] = {"label": full_label, "confidence": full_conf, "available": True}
        except Exception as e:
            results["full"] = {"available": False, "error": str(e)}
    else:
        results["full"] = {"available": False, "info": "Modelul Full FT nu a fost încă antrenat."}
        
    return results

# Creăm directorul pentru fișierele statice dacă nu există
os.makedirs("static", exist_ok=True)

# Servirea frontend-ului static
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("Pornește serverul pe http://localhost:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
