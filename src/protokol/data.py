from __future__ import annotations
import numpy as np
from src.common import manifest, split
from src.protokol.pemotong_jendela import (
    PANJANG_JENDELA,
    buat_sesi_simulasi,
    potong_jendela_kondisi,
)

TUGAS = "klasifikasi"


def _normalisasi_latih_saja(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (
        ((X_train - mean) / std).astype(np.float32),
        ((X_val - mean) / std).astype(np.float32),
        ((X_test - mean) / std).astype(np.float32),
    )


def muat():
    _ = manifest.baca()
    berkas_primer = manifest.AKAR / "data" / "sesi_meja_13nov.npz"
    berkas = (
        berkas_primer
        if berkas_primer.exists()
        else buat_sesi_simulasi(manifest.AKAR / "data" / "simulasi_t29", seed=split.SEED_SPLIT)
    )
    d = np.load(berkas)
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

    mask_train, mask_val, mask_test = (pid <= 5), (pid == 6), (pid == 7)
    pid_train = set(pid[mask_train].tolist())
    pid_val = set(pid[mask_val].tolist())
    pid_test = set(pid[mask_test].tolist())

    assert pid_train.isdisjoint(pid_test), "Kebocoran partisipan latih-uji"
    assert pid_train.isdisjoint(pid_val), "Kebocoran partisipan latih-val"
    assert pid_val.isdisjoint(pid_test), "Kebocoran partisipan val-uji"

    X_train, X_val, X_test = _normalisasi_latih_saja(X[mask_train], X[mask_val], X[mask_test])
    return X_train, y[mask_train], X_val, y[mask_val], X_test, y[mask_test]