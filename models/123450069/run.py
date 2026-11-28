"""Kerangka run.py anggota — JALUR TL (backlog #14). Salin folder ini menjadi models/<nim>/.

Jalur transfer learning (**TL = MobileNet pada potongan tangan**, catatan metode tim).
Memanggil ``src.protokol.data.muat()`` (protokol tim yang dibekukan) dan menulis berkas hasil
lewat ``src.common.evaluate``. Diuji pada data simulasi agar saat data meja datang yang tersisa
hanya pelatihan.

    python models/<nim>/run.py --seed 1 [--final]

MASUKAN MODEL TIM (backlog #14): **21 titik tangan + potongan tangan**.
- ``muat()`` mengembalikan pose **21 titik tangan × 3 koordinat** (bentuk bebas [N,...,63]).
- **potongan tangan (hand crop):** ``src.protokol.pemotong_jendela.potong_jendela_kondisi``
  dapat menghasilkan ``win_crop`` bila protokol menyediakan frame kamera; pada protokol saat ini
  ``muat()`` **hanya mengembalikan pose**, sedangkan data meja yang diterima + potongan tangan
  perlu protokol memperluas keluaran. Kerangka ini **memanggil ``muat()``** sesuai kontrak.

KERANGKA TL yang DIJALANKAN sekarang (supaya ``--seed 1`` jalan sampai akhir pada data simulasi):
- kunci ``backbone`` MobileNetV2 pralatih; ganti kepala klasifikasi sesuai **ukuran keluaran**
  (jumlah kelas klasifikasi; untuk regresi, kepala ukuran target). Ukuran **masukan** disesuaikan
  (21 titik + potongan tangan). 
- **potongan tangan** ketika tersedia sebaiknya menjadi **citra** masukan MobileNet; saat ini
  pose 21 titik di-render ke citra + potongan tangan bila ada.

Catatan `models/README.md`: optimizer dengan alasan; ≥1 regularisasi; tuning + catatan; ≥2 ablation;
3 seed konfigurasi akhir (backlog #15 untuk latih data primer; 3 seed penuh Sprint 2).
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

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "tl"                                   # backlog #14: transfer learning / MobileNet
JUMLAH_TITIK = 21
DIM = 3
NUM_KELAS = 10                                 # klasifikasi angka BISINDO; sesuaikan jenis tugas
UKURAN_GAMBAR = 224                            # MobileNetV2 standar


def _ke_titik(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(len(X), X.shape[1], JUMLAH_TITIK, DIM)


def _render_pose(pts, size: int = UKURAN_GAMBAR) -> np.ndarray:
    """Render pose 21 titik (per frame) -> citra [N,3,size,size] untuk backbone MobileNet."""
    import torch
    N = len(pts)
    kanvas = np.zeros((N, 3, size, size), np.float32)
    # rata waktu -> satu gambar per sampel
    xyz = pts.mean(axis=1)                                   # [N,21,3]
    xy = xyz[..., :2]
    lo, hi = xy.min(1, keepdims=True), xy.max(1, keepdims=True)
    xy = (xy - lo) / (hi - lo + 1e-6)
    px = (xy[..., 0] * (size - 1)).astype(int)
    py = (xy[..., 1] * (size - 1)).astype(int)
    for n in range(N):
        for t in range(JUMLAH_TITIK):
            kanvas[n, :, py[n, t], px[n, t]] = 1.0
    return kanvas


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Kembalikan (pred_val, pred_test|None). Jalur TL MobileNet; ganti sesuai kebutuhan tim."""
    import torch
    import torch.nn as nn
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, _, Xte, _ = data.muat()                    # WAJIB src.protokol.data.muat()
    pts_tr = _ke_titik(Xtr); pts_va = _ke_titik(Xva)
    Gtr = torch.tensor(_render_pose(pts_tr)); Gva = torch.tensor(_render_pose(pts_va))
    # potongan tangan: bila tersedia (protokol diperluas), tambahkan sebagai kanal citra di sini.
    y = torch.tensor(np.asarray(ytr, dtype=np.int64))

    n_kelas = int(max(int(np.asarray(ytr).max()) + 1, NUM_KELAS)) if len(ytr) else NUM_KELAS
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)  # pralatih; sesuaikan freezing/fine-tuning
    model.classifier[1] = nn.Linear(model.last_channel, n_kelas)
    for p in model.features.parameters():
        p.requires_grad = False                                # freezing backbone (kerangka TL); Sprint lanjut fine-tune

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(2):
        for i in range(0, len(Gtr), 32):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(Gtr[i:i + 32]), y[i:i + 32])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred_val = model(Gva).argmax(1).numpy().astype(np.int64)
        pred_test = None
        if final and Xte is not None:
            Gte = torch.tensor(_render_pose(_ke_titik(Xte)))
            pred_test = model(Gte).argmax(1).numpy().astype(np.int64) if len(Xte) else None
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
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
