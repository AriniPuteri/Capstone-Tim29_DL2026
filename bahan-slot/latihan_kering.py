"""Latihan kering rencana Tim 29 (PENGGANTI SEMENTARA, BUKAN meja.py).

    python bahan-slot/latihan_kering.py <path ke t29-angka-isyarat.yaml>

Membaca rencana, membangun urutan blok dan kondisi dengan seed rencana, lalu memeriksa:
240 kondisi, 30 kondisi per partisipan, tiap angka 3x per partisipan, dan perkiraan waktu sesi.
Urutan acaknya TIDAK sama dengan keluaran meja.py (generator acaknya berbeda); skrip ini hanya
memastikan desain rencana benar. Hasil: bahan-slot/urutan_latihan_kering.csv.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

rencana = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
seed = rencana["seed"]
ulangan = rencana["ulangan"]
dur, jeda = rencana["durasi_kondisi_detik"], rencana["jeda_detik"]
partisipan = rencana["faktor"]["partisipan"]["taraf"]
angka = [str(a) for a in rencana["faktor"]["angka"]["taraf"]]

rng = np.random.default_rng(seed)
blok = [partisipan[i] for i in rng.permutation(len(partisipan))]
baris, no = [], 0
for b in blok:
    kombinasi = angka * ulangan
    for i in rng.permutation(len(kombinasi)):
        no += 1
        baris.append((no, b, kombinasi[i]))

# pemeriksaan
assert len(baris) == 240, f"jumlah kondisi {len(baris)} != 240"
for p in partisipan:
    sub = [a for _, b, a in baris if b == p]
    assert len(sub) == 30, f"{p}: {len(sub)} kondisi, seharusnya 30"
    assert all(sub.count(a) == ulangan for a in angka), f"{p}: ulangan per angka tidak seimbang"
bersih = len(baris) * (dur + jeda) / 60
total = bersih + 12
print(f"OK: {len(baris)} kondisi, {len(partisipan)} blok x 30, tiap angka {ulangan}x per partisipan")
print(f"urutan blok (seed {seed}): {' '.join(blok)}")
print(f"waktu rekam bersih ~{bersih:.0f} menit; + ~12 menit pergantian blok = ~{total:.0f} menit (slot efektif ~50 menit)")
print(f"model: jalur di rencana = {rencana['metode']['jalur']}")

out = Path(__file__).with_name("urutan_latihan_kering.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["no", "partisipan", "angka"])
    w.writerows(baris)
print("ditulis:", out.name)
