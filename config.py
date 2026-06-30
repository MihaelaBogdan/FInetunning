"""
Configurare centrală pentru experimentul LoRA vs Full Fine-Tuning.
Modifică parametrii aici pentru a adapta experimentul.
"""

# ─── Model ───────────────────────────────────────────────────────────────────
# Model mic open-source, rulează bine și pe CPU/MPS
BASE_MODEL = "google/flan-t5-small"   # ~80M parametri
# Alternative: "google/flan-t5-base" (250M), "facebook/opt-125m"

# ─── Dataset ─────────────────────────────────────────────────────────────────
# Sarcină: clasificare sentiment (SST-2) — binar, simplu de evaluat
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
TASK = "sentiment"                    # "sentiment" | "summarization"

# ─── Antrenament comun ───────────────────────────────────────────────────────
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
