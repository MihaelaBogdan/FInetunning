"""Evaluare acuratețe pe setul de test."""

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from tqdm import tqdm
from config import EVAL_BATCH_SIZE, MAX_TARGET_LENGTH


def evaluate_accuracy(model, tokenizer, test_ds, raw_ds, device):
    """
    Generează predicții și calculează acuratețea față de etichetele reale.
    raw_ds: dataset original (ne-tokenizat) cu coloana 'label'.
    """
    model.eval()
    model.to(device)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, collate_fn=collator)

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluare", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_TARGET_LENGTH,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_preds.extend(decoded)

    # Etichete reale din dataset-ul brut
    true_labels = [
        "positive" if ex["label"] == 1 else "negative"
        for ex in raw_ds
    ]

    correct = sum(p.strip().lower() == t for p, t in zip(all_preds, true_labels))
    accuracy = correct / len(true_labels)
    return accuracy
