"""
Antrenament cu LoRA (Parameter-Efficient Fine-Tuning).
Folosește PEFT pentru a antrena doar adaptori de rang mic.
"""

import os
import torch
import json
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, TaskType, get_peft_model
from data_utils import load_tokenizer, get_datasets
from metrics_tracker import (
    ExperimentMetrics, count_parameters, Timer,
    peak_ram_mb, peak_gpu_mb, checkpoint_size_mb,
)
from evaluate_model import evaluate_accuracy
from config import (
    BASE_MODEL, OUTPUT_DIR, LOG_DIR,
    BATCH_SIZE, EVAL_BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, WARMUP_RATIO, SEED,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    TRAIN_SAMPLES,
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ProgressCb(TrainerCallback):
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.progress_callback:
            epoch = state.epoch
            step = state.global_step
            max_steps = state.max_steps
            loss = logs.get("loss", None)
            eval_loss = logs.get("eval_loss", None)
            progress_pct = (step / max_steps) * 100 if max_steps > 0 else 0
            
            self.progress_callback({
                "epoch": round(epoch, 2),
                "step": step,
                "max_steps": max_steps,
                "loss": loss,
                "eval_loss": eval_loss,
                "progress_pct": round(progress_pct, 1)
            })


def train_lora(
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    train_samples=TRAIN_SAMPLES,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    progress_callback=None
):
    print("=" * 60)
    print("  EXPERIMENT 1: LoRA Fine-Tuning")
    print("=" * 60)

    device = get_device()
    print(f"  Device: {device}")

    tokenizer = load_tokenizer()
    train_ds, eval_ds, test_ds, test_raw = get_datasets(tokenizer, train_samples=train_samples)

    print("\n[1/4] Încărcare model de bază...")
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    print("[2/4] Aplicare LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total, trainable = count_parameters(model)
    metrics = ExperimentMetrics(
        method="lora",
        total_params=total,
        trainable_params=trainable,
        trainable_pct=round(100 * trainable / total, 4),
    )
    print(f"  Total parametri:      {total:,}")
    print(f"  Parametri antrenați:  {trainable:,} ({metrics.trainable_pct:.4f}%)")

    out_dir = os.path.join(OUTPUT_DIR, "lora")
    log_dir = os.path.join(LOG_DIR, "lora")

    # Setăm logging_steps pentru actualizări frecvente
    logging_steps = max(1, (train_samples // batch_size) // 10)

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=log_dir,
        logging_steps=logging_steps,
        seed=SEED,
        report_to="none",
        fp16=device == "cuda",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    loss_history = {"train": [], "eval": []}

    class LossCb(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            logs = state.log_history
            train_loss = next(
                (l["loss"] for l in reversed(logs) if "loss" in l and "eval_loss" not in l),
                None,
            )
            eval_loss = next(
                (l["eval_loss"] for l in reversed(logs) if "eval_loss" in l),
                None,
            )
            if train_loss:
                loss_history["train"].append(round(train_loss, 4))
            if eval_loss:
                loss_history["eval"].append(round(eval_loss, 4))

    callbacks = [LossCb()]
    if progress_callback:
        callbacks.append(ProgressCb(progress_callback))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("\n[3/4] Antrenament LoRA...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with Timer() as t:
        trainer.train()

    metrics.train_time_sec = round(t.elapsed, 2)
    metrics.peak_ram_mb = round(peak_ram_mb(), 2)
    metrics.peak_gpu_mb = round(peak_gpu_mb(), 2)
    metrics.train_loss_history = loss_history["train"]
    metrics.eval_loss_history = loss_history["eval"]
    metrics.calculate_energy(device)

    print("\n[4/4] Evaluare acuratețe pe test set...")
    # Salvează adaptorii LoRA (mult mai mici decât modelul complet)
    adapter_dir = os.path.join(out_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics.checkpoint_size_mb = round(checkpoint_size_mb(adapter_dir), 2)

    accuracy = evaluate_accuracy(model, tokenizer, test_ds, test_raw, device)
    metrics.final_accuracy = round(accuracy, 4)

    metrics.save(os.path.join(OUTPUT_DIR, "metrics_lora.json"))

    print("\n" + "─" * 40)
    print(f"  Timp antrenament:    {metrics.train_time_sec:.1f}s")
    print(f"  RAM peak:            {metrics.peak_ram_mb:.0f} MB")
    print(f"  GPU peak:            {metrics.peak_gpu_mb:.0f} MB")
    print(f"  Acuratețe test:      {metrics.final_accuracy:.4f}")
    print(f"  Dimensiune adapter:  {metrics.checkpoint_size_mb:.1f} MB")
    print(f"  Energie estimată:    {metrics.estimated_energy_kwh:.6f} kWh")
    print(f"  Emisii CO2 estimate: {metrics.estimated_co2_g:.4f} g")
    print("─" * 40)

    return metrics


if __name__ == "__main__":
    train_lora()
