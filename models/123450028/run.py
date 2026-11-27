"""Jalur Transformer Tim 29: Transformer encoder atas patch, klasifikasi angka isyarat 0-9.

    python models/123450028/run.py --seed 1 [--final]

Masukan data.muat(): (N, 2, 63) = 2 frame aktif x 63 fitur citra.
Tiap frame dipotong menjadi 9 patch berisi 7 fitur berurutan, sehingga ada 18 token.
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
JALUR = "trf"

# konfigurasi awal Sprint 1 (belum disetel)
PANJANG_PATCH = 7
D_MODEL = 32
HEAD = 4
LAPIS = 2
DROPOUT = 0.2
EPOCH = 100
LR = 1e-3
WEIGHT_DECAY = 1e-2
BATCH = 32


def jadi_patch(X) -> torch.Tensor:
    X = torch.tensor(np.asarray(X, dtype=np.float32))      # (N, 2, 63)
    return X.reshape(len(X), -1, PANJANG_PATCH)            # (N, 18, 7)


class TransformerPatch(nn.Module):
    def __init__(self, n_token: int):
        super().__init__()
        self.proyeksi = nn.Linear(PANJANG_PATCH, D_MODEL)
        self.posisi = nn.Parameter(torch.randn(1, n_token, D_MODEL) * 0.02)
        lapis = nn.TransformerEncoderLayer(D_MODEL, HEAD, dim_feedforward=2 * D_MODEL,
                                           dropout=DROPOUT, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(lapis, LAPIS, enable_nested_tensor=False)
        self.kepala = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.proyeksi(x) + self.posisi)
        return self.kepala(z.mean(dim=1))                  # rata-rata semua token


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    X, y = jadi_patch(Xtr), torch.tensor(np.asarray(ytr, dtype=np.int64))

    model = TransformerPatch(n_token=X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed)

    for ep in range(1, EPOCH + 1):
        model.train()
        urut = torch.randperm(len(X), generator=gen)
        total = 0.0
        for i in range(0, len(X), BATCH):
            idx = urut[i:i + BATCH]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if ep == 1 or ep % 20 == 0:
            print(f"epoch {ep:3d}  loss latih {total / len(X):.4f}")

    model.eval()
    with torch.no_grad():
        acc_latih = (model(X).argmax(1) == y).float().mean().item()
        pred_val = model(jadi_patch(Xva)).argmax(1).numpy()
        pred_test = model(jadi_patch(Xte)).argmax(1).numpy() if final and Xte is not None else None
    n_param = sum(p.numel() for p in model.parameters())
    print(f"parameter: {n_param}  |  akurasi latih: {acc_latih:.4f}")
    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
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