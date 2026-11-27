# Catatan model 123450120 (CNN 1D)

## Masukan dan arsitektur
- Masukan dari `src.protokol.data.muat()`: (N, 2, 63) = 2 frame aktif x 63 fitur per kondisi.
- 2 frame dipakai sebagai 2 kanal; konvolusi 1D berjalan di sepanjang 63 fitur.
- Conv1d(2→16, k=5) → BatchNorm → ReLU → MaxPool(2) → Conv1d(16→32, k=3) → BatchNorm → ReLU
  → rata-rata global → Dropout → Linear(32→10).

## Alasan optimizer
AdamW (lr 1e-3, weight decay 1e-4): data latih hanya 180 kondisi, sehingga laju belajar adaptif
konvergen lebih cepat dan stabil daripada SGD tanpa banyak penyetelan; weight decay terpisah
(decoupled) membantu menahan overfitting.

## Regularisasi
- Dropout 0,3 sebelum lapisan keluaran.
- Weight decay 1e-4 (AdamW).
- BatchNorm pada setiap blok konvolusi.

## Pencarian hyperparameter (tabel)
Sprint 2.

| Konfigurasi | lr | dropout | epoch | F1 validasi |
|---|---|---|---|---|

## Ablation (>= 2)
Rencana Sprint 2:
- Tanpa 17 kondisi yang ditandai tidak sesuai di `data/periksa-isyarat.csv`.
- Tanpa BatchNorm / tanpa dropout.

## Hasil Sprint 1 (seed 1, konfigurasi awal)
| Seed | F1 validasi (P07) | Akurasi validasi | Durasi (s) | data_hash |
|---|---|---|---|---|
| 1 | 0,0986 | 0,1333 | 2,8 | 516bd1cb3a8af011d1e90790c22df16f |

Pengamatan: loss latih hanya turun dari 2,35 ke 2,22 dalam 150 epoch (tebakan acak 10 kelas
= ln 10 ≈ 2,30), dan F1 validasi setara tebakan acak (0,10). Model belum mampu mempelajari
data latih sekalipun, sehingga kemungkinan besar fitur masukan (63 fitur citra per frame) belum
membawa informasi bentuk tangan yang cukup. RNN tim juga mendapat hasil setara acak. Perlu
diperiksa bersama Penjaga Data dan dibandingkan dengan pembanding non-DL.

## Hasil 3 seed konfigurasi akhir
Sprint 2.
