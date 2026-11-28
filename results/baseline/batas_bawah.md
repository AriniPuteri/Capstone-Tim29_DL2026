# Angka pembanding non-DL sebagai batas bawah (backlog #9)

Sumber: `pembanding_non_dl.py`, `results/baseline/metrics.json`, `results/baseline/f1_per_partisipan.json`.
Pemisahan protokol: `src.protokol.data.muat()`. Penilai: `src.common.evaluate`. `src.common` tidak diubah.
Status sesuai instruksi pengguna: poin 2 (evaluasi uji resmi) dikerjakan teman + `--final`; poin 3 (keputusan tim/protokol) ditanggung dosen. Aksi **Papan Tim** (mencatat di web) tetap dilakukan pengguna; artefak ini siap disalin.

## Batas bawah (sementara, data validasi; sebelum uji resmi sama)
- metrik utama kartu: **F1 per partisipan**
- **`f1_per_partisipan_val` = 0.10714285714285714** (`n_val=30`, partisipan val = pid=6)
- `val_macro_f1` (macro-F1 per kelas, `src.common.evaluate`) = **0.107143**
- `val_acc` = **0.1**
- `seed=2026`, `n_val=30`, `n_test=30`, `model=baseline_non_dl`
- `data_hash` = `ff76f8a64151b9974a3ef9c0a53e3eeb`

## Uji (data uji yang sama, sprint terakhir `--final` Penjaga Data)
- `pred_test.npy` disiapkan; angka `f1_per_partisipan_test` saya hitung = **DIAGNOSTIK**, **bukan** evaluasi resmi.
- setelah uji sama + F1 per partisipan resmi, **ganti** batas bawah dengan angka final.

## Catatan teknis (sudah dinotasikan untuk dosen)
- `src.common.evaluate` = macro-F1 **per kelas**, bukan otomatis per partisipan; F1 per partisipan disimpan terpisah, tidak mengedit `metrics.json` manual.
- `muat()` output **normalized**; sudut sendi paling benar dari landmark mentah.
- **`results/baseline` konflik:** starter `src.common.baseline` + backlog #8 + pembanding non-DL `results/baseline`; `run_all.py` memanggil `src.common.baseline`. Backlog #9 tetap `results/baseline/metrics.json` (`model=baseline_non_dl`) sesuai kartu; keputusan final diserahkan dosen.

## Salin untuk Papan Tim (`https://sd.itera.ac.id/workbench/capstone/tim/tim_b393c7363d6f`)
> Pembanding non-DL (sudut sendi + kecepatan titik pose + Random Forest) = alat ukur batas bawah. Batas bawah sementara (data validasi): **F1 per partisipan 0.1071** / macro-F1 per kelas 0.1071 / akurasi 0.1. Backlog #9: `results/baseline/metrics.json` valid; angka pembanding dicatat sebagai batas bawah. Uji resmi sama `--final` Penjaga Data; angka test saya diagnostik. PR #14 `123450069/backlog-9`; `src.common` tidak diubah.
