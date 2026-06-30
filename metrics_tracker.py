"""Urmărire metrici: timp, memorie, parametri, acuratețe."""

import time
import os
import json
import psutil
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ExperimentMetrics:
    method: str                         # "lora" | "full"
    # Parametri
    total_params: int = 0
    trainable_params: int = 0
    trainable_pct: float = 0.0
    # Timp
    train_time_sec: float = 0.0
    # Memorie
    peak_ram_mb: float = 0.0
    peak_gpu_mb: float = 0.0
    # Performanță
    train_loss_history: list = field(default_factory=list)
    eval_loss_history: list = field(default_factory=list)
    final_accuracy: float = 0.0
    # Dimensiune checkpoint
    checkpoint_size_mb: float = 0.0
    # Green AI
    estimated_energy_kwh: float = 0.0
    estimated_co2_g: float = 0.0

    def to_dict(self):
        return asdict(self)

    def calculate_energy(self, device_type: str):
        # Estimări TDP tipice
        if device_type == "cuda":
            tdp = 250.0  # W (medie GPU Nvidia standard)
        elif device_type == "mps":
            tdp = 30.0   # W (Apple Silicon este extrem de eficient)
        else:
            tdp = 65.0   # W (CPU mediu)
        
        # kWh = (Watts * secunde) / (3600 * 1000)
        self.estimated_energy_kwh = round((tdp * self.train_time_sec) / 3600000, 6)
        # 300g CO2 per kWh (intensitate medie carbon)
        self.estimated_co2_g = round(self.estimated_energy_kwh * 300.0, 4)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.start


def peak_ram_mb():
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1024 / 1024


def peak_gpu_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS nu expune max_memory_allocated în versiunile vechi
        return 0.0
    return 0.0


def checkpoint_size_mb(directory: str) -> float:
    total = 0
    for dirpath, _, files in os.walk(directory):
        for f in files:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / 1024 / 1024
