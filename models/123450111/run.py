from __future__ import annotations

import sys
from pathlib import Path

_AKAR = Path(__file__).resolve().parents[2]
if str(_AKAR) not in sys.path:
    sys.path.insert(0, str(_AKAR))

import argparse
import subprocess
import tempfile
import time

import numpy as np

from src.common import manifest, split
from src.protokol import data
from src.protokol.pemotong_jendela import PANJANG_JENDELA, buat_sesi_simulasi, potong_jendela_kondisi

NIM = "123450111"
JALUR_MODEL = "rnn"
JUMLAH_KELAS = 10


def _normalisasi_latih_saja(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (
        ((X_train - mean) / std).astype(np.float32),
        ((X_val - mean) / std).astype(np.float32),
        ((X_test - mean) / std).astype(np.float32),
    )


def _muat_data(sumber: str):
    if sumber == "simulasi":
        _ = manifest.baca()
        berkas_sim = buat_sesi_simulasi(manifest.AKAR / "data" / "simulasi_t29", seed=split.SEED_SPLIT)
        d = np.load(berkas_sim)
        landmarks_mentah = d["landmarks"]
        y = d["angka"].astype(np.int64)
        pid = d["partisipan"].astype(np.int64)
        tangan_kiri = d["tangan_kiri"].astype(bool)
        X = np.asarray(
            [
                potong_jendela_kondisi(landmarks_mentah[i], adalah_tangan_kiri=bool(tangan_kiri[i]))[0].reshape(
                    PANJANG_JENDELA, -1
                )
                for i in range(len(landmarks_mentah))
            ],
            dtype=np.float32,
        )
        m_tr, m_va, m_te = (pid <= 5), (pid == 6), (pid == 7)
        X_tr, X_va, X_te = _normalisasi_latih_saja(X[m_tr], X[m_va], X[m_te])
        return X_tr, y[m_tr], X_va, y[m_va], X_te, y[m_te]
    return data.muat()


class GRUClassifierNumPy:
    """GRU sekuensial atas urutan langkah waktu (T, F) -> logit (N, 10)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, seed: int):
        rng = np.random.default_rng(seed)
        scale_x = np.sqrt(1.0 / max(input_dim, 1))
        scale_h = np.sqrt(1.0 / max(hidden_dim, 1))
        self.Wz = rng.normal(0.0, scale_x, (input_dim, hidden_dim)).astype(np.float32)
        self.Uz = rng.normal(0.0, scale_h, (hidden_dim, hidden_dim)).astype(np.float32)
        self.bz = np.zeros((1, hidden_dim), dtype=np.float32)

        self.Wr = rng.normal(0.0, scale_x, (input_dim, hidden_dim)).astype(np.float32)
        self.Ur = rng.normal(0.0, scale_h, (hidden_dim, hidden_dim)).astype(np.float32)
        self.br = np.zeros((1, hidden_dim), dtype=np.float32)

        self.Wh = rng.normal(0.0, scale_x, (input_dim, hidden_dim)).astype(np.float32)
        self.Uh = rng.normal(0.0, scale_h, (hidden_dim, hidden_dim)).astype(np.float32)
        self.bh = np.zeros((1, hidden_dim), dtype=np.float32)

        self.Wy = rng.normal(0.0, np.sqrt(1.0 / max(hidden_dim, 1)), (hidden_dim, num_classes)).astype(np.float32)
        self.by = np.zeros((1, num_classes), dtype=np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x_clip = np.clip(x, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-x_clip))

    def encode_sequence(self, X: np.ndarray) -> np.ndarray:
        n, t_steps, _ = X.shape
        h = np.zeros((n, self.Wz.shape[1]), dtype=np.float32)
        for t in range(t_steps):
            xt = X[:, t, :]
            z = self._sigmoid(xt @ self.Wz + h @ self.Uz + self.bz)
            r = self._sigmoid(xt @ self.Wr + h @ self.Ur + self.br)
            h_tilde = np.tanh(xt @ self.Wh + (r * h) @ self.Uh + self.bh)
            h = (1.0 - z) * h + z * h_tilde
        return h

    def forward(self, X: np.ndarray) -> np.ndarray:
        h_last = self.encode_sequence(X)
        return h_last @ self.Wy + self.by

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 25, lr: float = 0.02, batch_size: int = 32):
        rng = np.random.default_rng(123450111)
        n = len(X)
        reg = 1e-3
        for _ in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                b_idx = idx[start : start + batch_size]
                xb = X[b_idx]
                yb = y[b_idx]
                h_last = self.encode_sequence(xb)
                logits = h_last @ self.Wy + self.by
                logits_shift = logits - logits.max(axis=1, keepdims=True)
                exp_l = np.exp(logits_shift)
                probs = exp_l / exp_l.sum(axis=1, keepdims=True)

                one_hot = np.zeros_like(probs)
                one_hot[np.arange(len(yb)), yb] = 1.0
                d_logits = (probs - one_hot) / len(yb)

                grad_Wy = h_last.T @ d_logits + reg * self.Wy
                grad_by = d_logits.sum(axis=0, keepdims=True)
                self.Wy -= lr * grad_Wy
                self.by -= lr * grad_by


def main() -> None:
    parser = argparse.ArgumentParser(description="Kerangka & Pelatihan Model RNN BISINDO 0-9")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sumber", choices=["simulasi", "otomatis"], default="simulasi")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--perangkat", type=str, default="cpu laptop")
    args = parser.parse_args()

    t0 = time.perf_counter()
    X_train, y_train, X_val, y_val, X_test, y_test = _muat_data(args.sumber)
    assert X_train.ndim == 3, f"Harapan X_train 3D (N, T, F), dapat {X_train.shape}"
    assert X_val.shape[1:] == X_train.shape[1:], f"Dimensi X_val {X_val.shape} tidak cocok dengan X_train {X_train.shape}"

    model = GRUClassifierNumPy(
        input_dim=X_train.shape[2],
        hidden_dim=args.hidden_dim,
        num_classes=JUMLAH_KELAS,
        seed=args.seed,
    )
    model.fit(X_train, y_train, epochs=args.epochs)
    logits_val = model.forward(X_val)
    pred_val = logits_val.argmax(axis=1).astype(np.int64)
    assert logits_val.shape == (len(X_val), JUMLAH_KELAS), f"Ukuran keluaran salah: {logits_val.shape}"
    assert pred_val.shape == (len(y_val),), f"Ukuran prediksi val salah: {pred_val.shape}"

    durasi = round(time.perf_counter() - t0, 4)
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_path = Path(tmpdir) / "pred_val.npy"
        np.save(pred_path, pred_val)
        cmd = [
            sys.executable,
            "-m",
            "src.common.evaluate",
            "--nim",
            NIM,
            "--model",
            JALUR_MODEL,
            "--seed",
            str(args.seed),
            "--pred-val",
            str(pred_path),
            "--durasi",
            str(durasi),
            "--perangkat",
            args.perangkat,
        ]
        subprocess.run(cmd, check=True)

    print(
        f"Selesai RNN ({args.sumber}) | seed={args.seed} | "
        f"X_train={X_train.shape} -> logits_val={logits_val.shape} | pred_val={pred_val.shape} | durasi={durasi}s"
    )


if __name__ == "__main__":
    main()
