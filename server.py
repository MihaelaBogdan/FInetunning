import os
import json
import queue
import threading
import torch
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Project imports
from config import (
    BASE_MODEL, OUTPUT_DIR, LORA_R, LORA_ALPHA,
    TRAIN_SAMPLES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, FULL_LR,
    TEXT_COLUMN, LABEL_NAMES, TASK,
)
from data_utils import load_tokenizer
from train_lora import train_lora, get_device
from train_full import train_full
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM

app = FastAPI(title="LoRA vs Full Fine-Tuning Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        # Cached models
        self.tokenizer = None
        self.base_model = None
        self.lora_model = None
        self.full_model = None

state = AppState()

# ─── Progress SSE broadcaster ─────────────────────────────────────────────────
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

# ─── Training helpers ────────────────────────────────────────────────────────
def run_training_thread(method: str, params: dict):
    state.is_training = True
    state.current_method = method
    state.current_progress = {"status": "started", "progress_pct": 0}
    
    # Send initial state
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
            # Reset cached LoRA model so it reloads for the next prediction
            state.lora_model = None
        elif method == "full":
            train_full(
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                learning_rate=params["learning_rate_full"],
                train_samples=params["train_samples"],
                progress_callback=progress_callback
            )
            # Reset cached Full model
            state.full_model = None
            
        state.current_progress = {"status": "completed", "progress_pct": 100}
        broadcaster.broadcast({"type": "status", "status": "completed", "method": method})
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Training error: {error_msg}")
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
        raise HTTPException(status_code=400, detail="Cannot update configuration while training is in progress.")
    state.config = new_config.dict()
    return {"status": "success", "config": state.config}

@app.post("/api/train/{method}")
def start_train(method: str, background_tasks: BackgroundTasks):
    if method not in ["lora", "full"]:
        raise HTTPException(status_code=400, detail="Invalid method. Choose 'lora' or 'full'.")
    if state.is_training:
        raise HTTPException(status_code=400, detail="A training session is already running.")
    
    # Start training in a separate thread so the FastAPI event loop is not blocked
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
                # Wait for queue data (with timeout to detect client disconnect)
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

def get_emotion_prediction(model, tokenizer, text, device):
    """Compute probabilities for all labels in `LABEL_NAMES` using T5."""
    model.eval()
    model.to(device)

    input_text = f"{TASK}: {text}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # First decoding step: inspect logits for the first generated token
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device)

    with torch.no_grad():
        outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits[0, 0, :]  # Shape: (vocab_size,)

        label_token_ids = []
        for label in LABEL_NAMES:
            token_ids = tokenizer(label, add_special_tokens=False).input_ids
            if len(token_ids) != 1:
                raise ValueError(f"Expected single-token label for '{label}', got {token_ids}")
            label_token_ids.append(token_ids[0])

        label_logits = logits[label_token_ids]
        probs = torch.softmax(label_logits, dim=0)

        top_idx = int(torch.argmax(probs).item())
        label = LABEL_NAMES[top_idx]
        confidence = float(probs[top_idx].item())
        scores = {LABEL_NAMES[i]: round(float(probs[i].item()), 4) for i in range(len(LABEL_NAMES))}

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "scores": scores
        }

@app.post("/api/predict")
def predict(req: PredictionRequest):
    device = get_device()
    
    # Lazy load the tokenizer and base model
    if state.tokenizer is None:
        state.tokenizer = load_tokenizer()
    if state.base_model is None:
        state.base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    
    results = {}
    
    # 1. Base model prediction (zero-shot)
    try:
        base_pred = get_emotion_prediction(state.base_model, state.tokenizer, req.text, device)
        results["base"] = {"label": base_pred["label"], "confidence": base_pred["confidence"], "scores": base_pred["scores"], "available": True}
    except Exception as e:
        results["base"] = {"available": False, "error": str(e)}
        
    # 2. LoRA model prediction (if trained)
    lora_path = os.path.join(OUTPUT_DIR, "lora", "lora_adapter")
    if os.path.exists(lora_path):
        try:
            if state.lora_model is None:
                # Reload clean base model on CPU before applying adapters
                clean_base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
                state.lora_model = PeftModel.from_pretrained(clean_base, lora_path)
            
            lora_pred = get_emotion_prediction(state.lora_model, state.tokenizer, req.text, device)
            results["lora"] = {"label": lora_pred["label"], "confidence": lora_pred["confidence"], "scores": lora_pred["scores"], "available": True}
        except Exception as e:
            results["lora"] = {"available": False, "error": str(e)}
    else:
        results["lora"] = {"available": False, "info": "LoRA model has not been trained yet."}
        
    # 3. Full Fine-Tuning model prediction (if trained)
    full_path = os.path.join(OUTPUT_DIR, "full", "full_model")
    if os.path.exists(full_path):
        try:
            if state.full_model is None:
                state.full_model = AutoModelForSeq2SeqLM.from_pretrained(full_path)
            
            full_pred = get_emotion_prediction(state.full_model, state.tokenizer, req.text, device)
            results["full"] = {"label": full_pred["label"], "confidence": full_pred["confidence"], "scores": full_pred["scores"], "available": True}
        except Exception as e:
            results["full"] = {"available": False, "error": str(e)}
    else:
        results["full"] = {"available": False, "info": "Full FT model has not been trained yet."}
        
    return results

# Create the static directory if it does not exist
os.makedirs("static", exist_ok=True)

# Serve the frontend static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("Starting server on http://localhost:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
