"""Pembanding non-DL (backlog #9): statistik sudut sendi + kecepatan titik pose + Random Forest.

Bangun pada data simulasi, jalankan pada data primer meja. Pemisahan memakai protokol tim
``src.protokol.data.muat()`` (split per partisipan, Sprint tim), tidak membuat split sendiri.
Prediksi validasi ditulis lewat penilai dosen ``src.common.evaluate`` ke
``results/baseline/metrics.json`` (berkas hasil valid), metrik `macro-F1` + akurasi.

    python pembanding_non_dl.py --seed 2026

CATATAN METODE (jujur, perlu konfirmasi tim):
- ``muat()`` mengembalikan pose **sudah dinormalisasi** latih-saja; "sudut sendi" + "kecepatan"
  dihitung dari keluaran ``muat()``. Untuk sudut sendi sesungguhnya sebaiknya hitung dari
  landmark mentah sebelum normalisasi (lihat ``src.protokol.pemotong_jendela``).
- ``src.common.evaluate`` memakai ``macro-F1`` **per kelas**; kartu menulis metrik utama
  **F1 per partisipan**. Penilaian per-partisipan tambahan belum dipaksakan di sini.
- Random Forest butuh scikit-learn; ditambahkan ke requirements, tidak mengubah ``src.common``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Windows: penilai src.common.evaluate mencetak panah UTF-8; paksa UTF-8 (tidak mengubah src.common).
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

AKAR = Path(__file__).resolve().parent
sys.path.insert(0, str(AKAR))

from src.common import manifest  # noqa: E402
from src.protokol import data  # noqa: E402

JUMLAH_TITIK = 21
DIM = 3
# Rantai jari MediaPipe Hands (indeks keypoint) untuk menghitung sudut sendi.
RANTAI = {
    "telapak": [0],
    "jempol": [1, 2, 3, 4],
    "telunjuk": [5, 6, 7, 8],
    "tengah": [9, 10, 11, 12],
    "manis": [13, 14, 15, 16],
    "kelingking": [17, 18, 19, 20],
}
TITIK_PANGKAL = 0  # pergelangan tangan


def _ke_titik(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(len(X), X.shape[1], JUMLAH_TITIK, DIM)


def _kecepatan_titik_pose(pts: np.ndarray) -> np.ndarray:
    """Statistik kecepatan titik pose: |Δposisi| per frame per titik -> ringkas per titik."""
    if pts.shape[1] < 2:
        z = np.zeros((len(pts), JUMLAH_TITIK), np.float32)
        return np.concatenate([z, z, z], axis=1)
    v = np.diff(pts, axis=1)                                   # [N, T-1, 21, 3]
    s = np.linalg.norm(v, axis=-1)                             # kecepatan per titik [N, T-1, 21]
    return np.concatenate([s.mean(1), s.max(1), s.std(1)], axis=1).astype(np.float32)  # [N, 63]


def _sudut_sendi(pts: np.ndarray) -> np.ndarray:
    """Sudut sendi per rantai jari per frame -> rata-rata per sendi (radian)."""
    fitur = []
    for rantai in RANTAI.values():
        for i in range(1, len(rantai) - 1):
            a = pts[:, :, rantai[i - 1], :] - pts[:, :, rantai[i], :]   # [N,T,3]
            b = pts[:, :, rantai[i + 1], :] - pts[:, :, rantai[i], :]
            na = np.linalg.norm(a, axis=-1) + 1e-8
            nb = np.linalg.norm(b, axis=-1) + 1e-8
            cos = np.clip((a * b).sum(-1) / (na * nb), -1.0, 1.0)
            sudut = np.arccos(cos)                                      # [N,T]
            fitur.append(sudut.mean(1))                                 # rata waktu -> [N]
    if not fitur:
        return np.zeros((len(pts), 0), np.float32)
    return np.stack(fitur, axis=1).astype(np.float32)


def _jarak_pangkal(pts: np.ndarray) -> np.ndarray:
    """Jarak tiap titik ke pergelangan tangan -> ringkas per titik (statistik bentuk)."""
    d = np.linalg.norm(pts - pts[:, :, TITIK_PANGKAL:TITIK_PANGKAL + 1, :], axis=-1)  # [N,T,21]
    return np.concatenate([d.mean(1), d.std(1)], axis=1).astype(np.float32)


def fitur_pose(X: np.ndarray) -> np.ndarray:
    """Fitur non-DL: kecepatan titik pose + sudut sendi + jarak ke pangkal + dispersi temporal."""
    pts = _ke_titik(X)
    kec = _kecepatan_titik_pose(pts)
    sudut = _sudut_sendi(pts)
    jarak = _jarak_pangkal(pts)
    dispersi = pts.std(1).reshape(len(pts), -1).astype(np.float32)     # [N, 21*3]
    return np.concatenate([kec, sudut, jarak, dispersi], axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--nim", default="baseline")
    ap.add_argument("--out", default=str(AKAR / "results" / "baseline"))
    a = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier  # noqa: E402  (non-DL; tidak menyentuh src.common)

    t0 = time.time()
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()
    Ftr, Fva = fitur_pose(Xtr), fitur_pose(Xva)

    rf = RandomForestClassifier(
        n_estimators=a.n_estimators, random_state=a.seed,
        class_weight="balanced", n_jobs=-1,
    )
    rf.fit(Ftr, np.asarray(ytr, dtype=np.int64))
    pred_val = rf.predict(Fva).astype(np.int64)

    durasi = time.time() - t0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred_val)

    perintah = [sys.executable, "-m", "src.common.evaluate",
                "--nim", a.nim, "--model", "baseline_non_dl",
                "--seed", str(a.seed), "--pred-val", str(out / "pred_val.npy"),
                "--durasi", f"{durasi:.1f}", "--perangkat", "cpu laptop"]
    # data uji hanya untuk info; pembanding resmi dipakai pada data uji sama; --final hanya Penjaga Data.
    subprocess.run(perintah, check=True, cwd=AKAR)

    # F1 PER PARTISIPAN (kartu backlog #9). Pemisahan per partisipan: val = satu partisipan,
    # uji = satu partisipan lain. Kita hitung macro-F1 per partisipan (per kelas dalam partisipan itu)
    # dan simpan TERPISAH; metrics.json tetap ditulis src.common.evaluate, tidak diedit manual.
    from src.common.evaluate import macro_f1  # noqa: E402
    ringkasan = {
        "definisi": "macro-F1 per kelas dihitung di dalam satu partisipan; partisipan val/test terpisah per src.protokol",
        "metrik_utama_kartu": "F1 per partisipan",
        "catatan_src_common_evaluate": "metrics.json memakai macro-F1 per kelas, bukan otomatis per partisipan",
    }
    yva = np.asarray(yva, dtype=np.int64)
    ringkasan["f1_per_partisipan_val"] = float(macro_f1(yva, pred_val))
    ringkasan["n_val"] = int(len(yva))
    ringkasan["partisipan_val"] = "pid=6 (lihat src.protokol.data.muat)"
    # Prediksi uji (partisipan 7) disiapkan; evaluasi resmi uji sekali --final oleh Penjaga Data.
    pred_test = None
    if Xte is not None and len(Xte):
        pred_test = rf.predict(fitur_pose(Xte)).astype(np.int64)
        np.save(out / "pred_test.npy", pred_test)
        ringkasan["n_test"] = int(len(Xte))
        ringkasan["partisipan_test"] = "pid=7 (lihat src.protokol.data.muat)"
        ringkasan["pred_test_disiapkan"] = True
        if yte is not None:
            ringkasan["f1_per_partisipan_test_DIAGNOSTIK"] = float(macro_f1(np.asarray(yte, dtype=np.int64), pred_test))
            ringkasan["test_diagnostik_catatan"] = "BUKAN evaluasi resmi; uji resmi sekali --final oleh Penjaga Data sprint terakhir"
    import json  # noqa: E402
    (out / "f1_per_partisipan.json").write_text(json.dumps(ringkasan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _ = manifest


if __name__ == "__main__":
    main()
