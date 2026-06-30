"""
Punct de intrare principal.
Rulează ambele experimente secvențial și generează raportul comparativ.

Utilizare:
    python run_experiment.py              # ambele metode
    python run_experiment.py --lora       # doar LoRA
    python run_experiment.py --full       # doar Full FT
    python run_experiment.py --report     # doar generează raportul din rezultate existente
"""

import argparse
import os
import sys

os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="LoRA vs Full Fine-Tuning Experiment")
    parser.add_argument("--lora",   action="store_true", help="Rulează doar LoRA")
    parser.add_argument("--full",   action="store_true", help="Rulează doar Full FT")
    parser.add_argument("--report", action="store_true", help="Generează doar raportul")
    args = parser.parse_args()

    run_lora = args.lora or (not args.lora and not args.full and not args.report)
    run_full = args.full or (not args.lora and not args.full and not args.report)

    if args.report:
        from compare_results import main as report
        report()
        return

    if run_lora:
        print("\n")
        from train_lora import train_lora
        train_lora()

    if run_full:
        print("\n")
        from train_full import train_full
        train_full()

    print("\n\n")
    from compare_results import main as report
    report()


if __name__ == "__main__":
    main()
