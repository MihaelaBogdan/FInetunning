"""
Main entry point.
Runs both experiments sequentially and generates the comparative report.

Usage:
    python run_experiment.py              # both methods
    python run_experiment.py --lora       # only LoRA
    python run_experiment.py --full       # only Full FT
    python run_experiment.py --report     # only generate the report from existing results
"""

import argparse
import os
import sys

os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="LoRA vs Full Fine-Tuning Experiment")
    parser.add_argument("--lora",   action="store_true", help="Run only LoRA")
    parser.add_argument("--full",   action="store_true", help="Run only Full FT")
    parser.add_argument("--report", action="store_true", help="Generate only the report")
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
