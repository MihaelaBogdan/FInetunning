"""
Central configuration for the LoRA vs Full Fine-Tuning experiment.
Edit these values to adjust the experiment.
"""

# ─── Model ───────────────────────────────────────────────────────────────────
# Small open-source model, runs well on CPU/MPS
BASE_MODEL = "google/flan-t5-small"   # ~80M parameters
# Alternatives: "google/flan-t5-base" (250M), "facebook/opt-125m"

# ─── Dataset ─────────────────────────────────────────────────────────────────
# Task: emotion classification (dair-ai/emotion) — multiclass text→text
DATASET_NAME = "dair-ai/emotion"
DATASET_CONFIG = "split"
TASK = "emotion"                       # "sentiment" | "emotion" | "summarization"
TEXT_COLUMN = "text"                   # SST-2 used "sentence", this dataset uses "text"

# Labels, in dataset index order (0→5)
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
# For quick fallback to sentiment, keep the old label list commented out:
# LABEL_NAMES = ["negative", "positive"]

# ─── Shared training settings ────────────────────────────────────────────────
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 8
TRAIN_SAMPLES = 200                  # subset redus pentru validare rapidă (implicit)
EVAL_SAMPLES = 50
TEST_SAMPLES = 50

BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

OUTPUT_DIR = "outputs"
LOG_DIR = "logs"

# ─── LoRA / QLoRA ────────────────────────────────────────────────────────────
LORA_R = 8                            # rank
LORA_ALPHA = 16                       # scaling
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q", "v"]     # pentru T5: query & value projections

# ─── Full Fine-Tuning ────────────────────────────────────────────────────────
FULL_LR = 5e-5                        # LR mai mic pentru full FT
