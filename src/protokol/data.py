"""Pemuat data kasus tim — MILIK TIM, dibekukan setelah Sprint 0 dengan persetujuan pembimbing.

Isi oleh Penjaga Data & Evaluasi. Kontrak yang dipakai penilai dan baseline starter:

    TUGAS = "klasifikasi"  # atau "regresi"
    muat() -> (X_train, y_train, X_val, y_val, X_test, y_test)

- X berupa array NumPy (bentuk bebas: [N, fitur], [N, C, H, W], [N, langkah, fitur]);
  y klasifikasi = label bulat 0..K-1, y regresi = float.
- Split latih/validasi WAJIB memakai ``src.common.split`` (seed 2026) agar sama di semua laptop.
- y_test boleh ``None`` bila dataset tidak punya bagian uji; bila ada, data uji hanya dievaluasi
  sekali di sprint terakhir.
- Berkas data dibaca dari ``data/`` dan dicantumkan di ``data/MANIFEST.json`` (sidik data).

Data simulasi (backlog #7): bila ``data/latih.csv`` belum ada, ``muat()`` membangkitkan dataset
sintetis deterministik (seed 2026) untuk menguji jalur evaluasi, lalu menyimpannya ke
``data/latih.csv`` agar ``src.common.manifest`` dapat menghitung ``data_hash``. Bila
``data/latih.csv`` sudah ada (mis. data primer meja), berkas itu yang dipakai. Format CSV:
header ``f0,...,fN,target``; ``target`` label bulat 0..K-1. Berkas ini tidak di-commit
(lihat ``.gitignore``); data simulasi dibangkitkan ulang otomatis.

Contoh lengkap data nyata: ``contoh_fashion_mnist.py`` di folder ini.
"""
from __future__ import annotations

import numpy as np

from src.common import manifest, split

TUGAS = "klasifikasi"

# Konfigurasi data simulasi (hanya dipakai bila data/latih.csv belum ada).
_SIM_SEED = 2026          # sama dengan split.SEED_SPLIT
_SIM_N = 400              # jumlah contoh
_SIM_FITUR = 21           # masukan tim: 21 titik tangan (lihat kartu kasus / backlog TLI)
_SIM_KELAS = 10           # jumlah kelas (angka BISINDO; sesuaikan bila kasus tim berbeda)
_SIM_NOISE = 0.15

_LATIH = manifest.AKAR / "data" / "latih.csv"


def _baca_csv(path) -> tuple[np.ndarray, np.ndarray]:
    """Baca CSV ``header + numerik + target``; kembalikan (X float32, y int64)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
    names = list(data.dtype.names or ())
    if not names or names[-1] != "target":
        raise ValueError(f"{path}: kolom terakhir harus 'target'; header ditemukan: {names}")
    fitur = [n for n in names if n != "target"]
    X = np.stack([np.atleast_1d(data[n]) for n in fitur], axis=1).astype(np.float32)
    y = np.atleast_1d(data["target"]).astype(np.int64)
    return X, y


def _tulis_csv(path, X: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join([f"f{i}" for i in range(X.shape[1])] + ["target"])
    baris = np.column_stack([X.astype(np.float64), y.astype(np.float64)])
    tmp = path.with_suffix(".csv.tmp")
    np.savetxt(tmp, baris, delimiter=",", header=header, comments="", fmt="%.6f")
    tmp.replace(path)


def _bangkitkan_simulasi(path) -> None:
    """Dataset sintetis deterministik: 10 kelas, pusat kelas + noise kecil."""
    rng = np.random.default_rng(_SIM_SEED)
    pusat = rng.normal(0.0, 1.0, size=(_SIM_KELAS, _SIM_FITUR)).astype(np.float32)
    y = rng.integers(0, _SIM_KELAS, size=_SIM_N)
    X = pusat[y] + _SIM_NOISE * rng.normal(0.0, 1.0, size=(_SIM_N, _SIM_FITUR)).astype(np.float32)
    _tulis_csv(path, X, y.astype(np.int64))


def muat():
    """Kembalikan (X_train, y_train, X_val, y_val, X_test, y_test) memakai split beku seed 2026.

    Split latih/validasi dari ``src.common.split.indeks``. Tidak ada bagian uji terpisah pada
    data simulasi; data uji hanya dievaluasi sekali di sprint terakhir oleh Penjaga Data.
    """
    if not _LATIH.exists():
        _bangkitkan_simulasi(_LATIH)
    X, y = _baca_csv(_LATIH)
    tr, va = split.indeks(len(X))
    return X[tr], y[tr], X[va], y[va], None, None
