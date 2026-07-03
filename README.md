# LoRA vs Full Fine-Tuning

## Introduction

This project is a hands-on comparative study of two ways to adapt a pretrained language model to a downstream task: **LoRA (Low-Rank Adaptation)** and **Full Fine-Tuning**. Instead of just reading about the tradeoffs, the project trains both methods on the same base model and the same dataset, then measures what actually differs between them — accuracy, training time, memory footprint, energy consumption, and robustness to catastrophic forgetting.

The base model is `google/flan-t5-small`, fine-tuned as a text-to-text emotion classifier on the `dair-ai/emotion` dataset (six classes: sadness, joy, love, anger, fear, surprise). The project ships as an interactive dashboard (FastAPI + vanilla JS/Chart.js) so every run can be configured, launched, and analyzed without touching the command line.

## Goals

1. **Quantify the efficiency gap.** Show concretely how many parameters LoRA trains (~0.45% of the model) versus Full Fine-Tuning (100%), and how that translates into training time, peak memory, and checkpoint size.

2. **Measure real energy and carbon impact.** Estimate kWh and CO2 emissions for each method, and translate that into relatable equivalents (minutes of video streaming, phone charges), following the "Green AI" idea that efficient fine-tuning is also an environmental choice.

3. **Test for catastrophic forgetting.** Probe both fine-tuned models with prompts unrelated to the training task to check whether aggressive full-parameter updates erode the base model's general capabilities — a common pitfall highlighted in fine-tuning literature.

4. **Make hyperparameters tangible.** Let the user tune samples, epochs, batch size, LoRA rank/alpha, learning rates, weight decay, and warmup ratio, and see a live estimate of training time and the resulting learning-rate schedule before committing to a run.

5. **Surface data quality issues before training.** Detect dataset duplicates, train/test leakage, and label distribution mismatches between splits — problems that silently inflate or deflate reported accuracy if left unchecked.

6. **Compare failure modes, not just aggregate accuracy.** Provide a confusion matrix, per-class precision/recall/F1, and a table of the most confident misclassifications, so the comparison goes beyond a single accuracy number and shows *where* each method struggles.

## Project structure

- `config.py` — central configuration (model, dataset, hyperparameters)
- `data_utils.py` — dataset loading and tokenization
- `train_lora.py` / `train_full.py` — training scripts for each method
- `evaluate_model.py` — accuracy, macro-F1, confusion matrix, misclassified examples
- `metrics_tracker.py` — timing, memory, and energy/CO2 estimation
- `server.py` — FastAPI backend serving the dashboard and training APIs
- `static/` — dashboard frontend (Training, Analytics & Green AI, Playground, Dataset Explorer tabs)
- `run_experiment.py` / `compare_results.py` — CLI alternative to the dashboard

## Running it

```bash
source venv/bin/activate
python server.py
```

Then open `http://localhost:8000`.
