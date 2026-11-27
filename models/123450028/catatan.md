# Catatan model 123450028 (Transformer encoder atas patch)

## Masukan dan arsitektur
- Masukan dari `src.protokol.data.muat()`: (N, 2, 63) = 2 frame aktif x 63 fitur citra.
- Tiap frame dipotong menjadi 9 patch berisi 7 fitur berurutan (18 token).
- Proyeksi linear ke 32 dimensi + embedding posisi, 2 lapis encoder (4 head, feedforward 64),
  rata-rata semua token, LayerNorm, Linear ke 10 kelas.

## Alasan optimizer
AdamW (lr 1e-3, weight decay 1e-2): laju belajar adaptif, dan weight decay dipisah dari
pembaruan gradien. Ini pertimbangan desain; belum dibandingkan dengan optimizer lain.

## Regularisasi
- Dropout 0,2 di lapisan encoder.
- Weight decay 1e-2.

## Pencarian hyperparameter (tabel)
Sprint 2.

## Ablation (>= 2)
Rencana Sprint 2: tanpa embedding posisi; tanpa dropout.

## Hasil 3 seed konfigurasi akhir
Sprint 2.