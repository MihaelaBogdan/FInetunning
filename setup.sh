#!/bin/bash
# Setup automat pentru experimentul LoRA vs Full Fine-Tuning
set -e

echo "======================================================"
echo "  Setup: LoRA vs Full Fine-Tuning Experiment"
echo "======================================================"

# Creează și activează venv
if [ ! -d "venv" ]; then
    echo "[1/3] Creare virtual environment..."
    python3 -m venv venv
fi

echo "[2/3] Activare venv și instalare dependențe..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[3/3] Setup complet!"
echo ""
echo "  Pentru a rula experimentul:"
echo "  source venv/bin/activate"
echo "  python run_experiment.py"
echo ""
echo "  Opțiuni:"
echo "  python run_experiment.py --lora    # doar LoRA"
echo "  python run_experiment.py --full    # doar Full FT"
echo "  python run_experiment.py --report  # generează raportul"
