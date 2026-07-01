"""Evaluate models for text-to-text multiclass classification tasks."""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq
from tqdm import tqdm
from config import EVAL_BATCH_SIZE, MAX_TARGET_LENGTH, LABEL_NAMES, TEXT_COLUMN


def evaluate_accuracy(model, tokenizer, test_ds, raw_ds, device):
    """Keep the old compatibility function for binary evaluation."""
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

    # True labels from the raw dataset (binary compatibility)
    true_labels = [
        (LABEL_NAMES[ex["label"]] if ex.get("label") is not None else "")
        for ex in raw_ds
    ]

    correct = sum(p.strip().lower() == t for p, t in zip(all_preds, true_labels))
    accuracy = correct / len(true_labels) if true_labels else 0.0
    return accuracy


def evaluate_detailed(model, tokenizer, test_ds, raw_ds, device):
    """Detailed evaluation for multiclass text-to-text classification.

    Returns a dict with accuracy, macro_f1, per_class, confusion_matrix, and misclassified examples.
    """
    model.eval()
    model.to(device)

    # first token id of each label --- assumes single-token labels
    label_first_token_ids = [tokenizer(w, add_special_tokens=False).input_ids[0] for w in LABEL_NAMES]

    results = []
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, collate_fn=collator)

    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluare detaliată", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=MAX_TARGET_LENGTH,
                output_scores=True, return_dict_in_generate=True,
            )
            decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
            # out.scores[0] -> logits for the first generated token
            probs = F.softmax(out.scores[0][:, label_first_token_ids], dim=-1)

            for i, pred in enumerate(decoded):
                true = LABEL_NAMES[raw_ds[idx]["label"]] if raw_ds[idx].get("label") is not None else ""
                pred_clean = pred.strip().lower()
                conf = probs[i, LABEL_NAMES.index(pred_clean)].item() if pred_clean in LABEL_NAMES else 0.0
                results.append({
                    "text": raw_ds[idx][TEXT_COLUMN], "true": true, "pred": pred_clean,
                    "confidence": round(conf, 4), "correct": pred_clean == true,
                })
                idx += 1

    # N×N confusion matrix
    confusion = {t: {p: 0 for p in LABEL_NAMES} for t in LABEL_NAMES}
    for r in results:
        if r["pred"] in LABEL_NAMES:
            confusion[r["true"]][r["pred"]] += 1

    per_class = {}
    for label in LABEL_NAMES:
        tp = confusion[label][label]
        fp = sum(confusion[t][label] for t in LABEL_NAMES if t != label)
        fn = sum(confusion[label][p] for p in LABEL_NAMES if p != label)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        per_class[label] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    correct = [r for r in results if r["correct"]]
    return {
        "accuracy": len(correct) / len(results) if results else 0.0,
        "macro_f1": round(sum(c["f1"] for c in per_class.values()) / len(LABEL_NAMES), 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "misclassified": sorted([r for r in results if not r["correct"]], key=lambda r: -r["confidence"])[:10],
    }
