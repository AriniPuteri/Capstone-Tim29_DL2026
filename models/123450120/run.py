"""Jalur CNN Tim 29: CNN 1D untuk klasifikasi angka isyarat 0-9.

    python3 models/123450120/run.py --seed 1 [--final]

Masukan dari src.protokol.data.muat(): X berbentuk (N, 2, 63) = 2 frame aktif x 63 fitur.
Kedua frame diperlakukan sebagai 2 kanal, dan konvolusi 1D berjalan di sepanjang 63 fitur.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "cnn"
JUMLAH_KELAS = 10

# konfigurasi Sprint 1 (penyetelan di Sprint 2)
EPOCH = 150
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
BATCH = 32


class CNN1D(nn.Module):
    def __init__(self, kanal_masuk: int, jumlah_kelas: int, dropout: float):
        super().__init__()
        self.fitur = nn.Sequential(
            nn.Conv1d(kanal_masuk, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.kepala = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(32, jumlah_kelas))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kepala(self.fitur(x))


def tetapkan_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def f1_makro(y: np.ndarray, p: np.ndarray) -> float:
    nilai = []
    for c in range(JUMLAH_KELAS):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        if tp + fp + fn:
            nilai.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(nilai)) if nilai else 0.0


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None, int]:
    tetapkan_seed(seed)
    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    ke_t = lambda X: torch.tensor(np.asarray(X, dtype=np.float32))  # noqa: E731
    X, y = ke_t(Xtr), torch.tensor(np.asarray(ytr, dtype=np.int64))

    model = CNN1D(kanal_masuk=X.shape[1], jumlah_kelas=JUMLAH_KELAS, dropout=DROPOUT)
    n_param = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed)

    for ep in range(1, EPOCH + 1):
        model.train()
        urut = torch.randperm(len(X), generator=gen)
        total = 0.0
        for i in range(0, len(X), BATCH):
            idx = urut[i:i + BATCH]
            if len(idx) < 2:          # BatchNorm butuh >= 2 sampel
                continue
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if ep % 25 == 0 or ep == 1:
            print(f"epoch {ep:3d}  loss latih {total / len(X):.4f}")

    model.eval()
    with torch.no_grad():
        pred_val = model(ke_t(Xva)).argmax(1).numpy()
        pred_test = model(ke_t(Xte)).argmax(1).numpy() if final and Xte is not None else None
    print(f"parameter: {n_param}  |  F1 validasi (P07): {f1_makro(np.asarray(yva), pred_val):.4f}")
    return pred_val, pred_test, n_param


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test, _ = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val, dtype=np.int64))
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test, dtype=np.int64))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
