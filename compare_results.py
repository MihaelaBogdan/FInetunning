"""
Generează raportul comparativ LoRA vs Full Fine-Tuning.
Produce tabele și grafice vizuale.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from metrics_tracker import ExperimentMetrics
from config import OUTPUT_DIR


COLORS = {"lora": "#2196F3", "full": "#F44336"}
LABELS = {"lora": "LoRA", "full": "Full FT"}


def load_metrics():
    results = {}
    for method in ["lora", "full"]:
        path = os.path.join(OUTPUT_DIR, f"metrics_{method}.json")
        if os.path.exists(path):
            results[method] = ExperimentMetrics.load(path)
        else:
            print(f"  [!] Lipsesc rezultatele pentru {method} ({path})")
    return results


def print_summary_table(results):
    rows = []
    for method, m in results.items():
        rows.append({
            "Metodă": LABELS[method],
            "Param. antrenați": f"{m.trainable_params:,}",
            "% din total": f"{m.trainable_pct:.4f}%",
            "Timp (s)": f"{m.train_time_sec:.1f}",
            "RAM peak (MB)": f"{m.peak_ram_mb:.0f}",
            "GPU peak (MB)": f"{m.peak_gpu_mb:.0f}",
            "Acuratețe": f"{m.final_accuracy:.4f}",
            "Checkpoint (MB)": f"{m.checkpoint_size_mb:.1f}",
        })
    df = pd.DataFrame(rows).set_index("Metodă")
    print("\n" + "=" * 70)
    print("  RAPORT COMPARATIV: LoRA vs Full Fine-Tuning")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)

    if len(results) == 2 and "lora" in results and "full" in results:
        lm, fm = results["lora"], results["full"]
        print("\n  CONCLUZII:")
        speedup = fm.train_time_sec / lm.train_time_sec if lm.train_time_sec > 0 else 0
        size_ratio = fm.checkpoint_size_mb / lm.checkpoint_size_mb if lm.checkpoint_size_mb > 0 else 0
        acc_diff = (lm.final_accuracy - fm.final_accuracy) * 100
        param_ratio = fm.trainable_params / lm.trainable_params if lm.trainable_params > 0 else 0

        print(f"  → LoRA antrenează de {param_ratio:.0f}x mai puțini parametri")
        print(f"  → Full FT este de {speedup:.1f}x mai lent")
        print(f"  → Checkpointul Full FT este de {size_ratio:.0f}x mai mare")
        acc_word = "mai bun" if acc_diff > 0 else "mai slab"
        print(f"  → LoRA este cu {abs(acc_diff):.2f}% {acc_word} ca acuratețe față de Full FT")


def plot_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    methods = list(results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Comparativ: LoRA vs Full Fine-Tuning\n(flan-t5-small · SST-2 · 2000 exemple)",
                 fontsize=14, fontweight="bold")

    # 1. Parametri antrenați
    ax = axes[0, 0]
    vals = [results[m].trainable_params / 1e6 for m in methods]
    bars = ax.bar([LABELS[m] for m in methods], vals,
                  color=[COLORS[m] for m in methods], edgecolor="white", linewidth=1.5)
    ax.set_title("Parametri antrenați (milioane)")
    ax.set_ylabel("Milioane")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}M", ha="center", va="bottom", fontsize=10)

    # 2. Timp antrenament
    ax = axes[0, 1]
    vals = [results[m].train_time_sec for m in methods]
    bars = ax.bar([LABELS[m] for m in methods], vals,
                  color=[COLORS[m] for m in methods], edgecolor="white", linewidth=1.5)
    ax.set_title("Timp antrenament (secunde)")
    ax.set_ylabel("Secunde")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}s", ha="center", va="bottom", fontsize=10)

    # 3. Acuratețe
    ax = axes[0, 2]
    vals = [results[m].final_accuracy * 100 for m in methods]
    bars = ax.bar([LABELS[m] for m in methods], vals,
                  color=[COLORS[m] for m in methods], edgecolor="white", linewidth=1.5)
    ax.set_title("Acuratețe test (%)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=10)

    # 4. RAM
    ax = axes[1, 0]
    vals = [results[m].peak_ram_mb for m in methods]
    bars = ax.bar([LABELS[m] for m in methods], vals,
                  color=[COLORS[m] for m in methods], edgecolor="white", linewidth=1.5)
    ax.set_title("RAM peak (MB)")
    ax.set_ylabel("MB")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=10)

    # 5. Dimensiune checkpoint
    ax = axes[1, 1]
    vals = [results[m].checkpoint_size_mb for m in methods]
    bars = ax.bar([LABELS[m] for m in methods], vals,
                  color=[COLORS[m] for m in methods], edgecolor="white", linewidth=1.5)
    ax.set_title("Dimensiune checkpoint (MB)")
    ax.set_ylabel("MB")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    # 6. Curbe loss
    ax = axes[1, 2]
    for method, m in results.items():
        epochs = range(1, len(m.eval_loss_history) + 1)
        if m.eval_loss_history:
            ax.plot(epochs, m.eval_loss_history,
                    color=COLORS[method], label=LABELS[method],
                    marker="o", linewidth=2, markersize=6)
    ax.set_title("Eval Loss per epocă")
    ax.set_xlabel("Epocă")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "comparison_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Grafic salvat: {out_path}")


def main():
    results = load_metrics()
    if not results:
        print("  Nu există rezultate. Rulează mai întâi run_experiment.py")
        return
    print_summary_table(results)
    plot_results(results)


if __name__ == "__main__":
    main()
