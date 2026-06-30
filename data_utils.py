"""Încărcare și preprocesare dataset."""

import random
from datasets import load_dataset
from transformers import AutoTokenizer
from config import (
    BASE_MODEL, DATASET_NAME, DATASET_CONFIG, TASK,
    MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,
    TRAIN_SAMPLES, EVAL_SAMPLES, TEST_SAMPLES, SEED,
)


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL)


def preprocess_sst2(examples, tokenizer):
    """Transformă SST-2 într-un format text→text pentru T5."""
    inputs = [f"sentiment: {s}" for s in examples["sentence"]]
    targets = ["positive" if l == 1 else "negative" for l in examples["label"]]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
    )
    # Înlocuiește padding token cu -100 (ignorat în loss)
    label_ids = [
        [(t if t != tokenizer.pad_token_id else -100) for t in lab]
        for lab in labels["input_ids"]
    ]
    model_inputs["labels"] = label_ids
    return model_inputs


def get_datasets(tokenizer, train_samples=TRAIN_SAMPLES, eval_samples=EVAL_SAMPLES, test_samples=TEST_SAMPLES):
    """Returnează train/eval/test tokenizate."""
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG)

    random.seed(SEED)

    def sample(split, n):
        ds = raw[split].shuffle(seed=SEED)
        return ds.select(range(min(n, len(ds))))

    train_raw = sample("train", train_samples)
    val_raw   = sample("validation", eval_samples)
    # SST-2 nu are test labels publice → folosim o parte din train
    test_raw  = raw["train"].shuffle(seed=SEED + 1).select(
        range(train_samples, min(train_samples + test_samples, len(raw["train"])))
    )

    def tokenize(ds):
        return ds.map(
            lambda ex: preprocess_sst2(ex, tokenizer),
            batched=True,
            remove_columns=ds.column_names,
        )

    return tokenize(train_raw), tokenize(val_raw), tokenize(test_raw), test_raw
