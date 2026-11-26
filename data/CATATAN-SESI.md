# Catatan Sesi & Pemeriksaan Mutu Data — Tim 29 (BISINDO 0–9)

Disusun oleh Penjaga Data & Evaluasi (`123450111` — Zailani Satria).

## 1. Sesi yang Dipakai
Data primer Tim 29 terdiri atas **240 kondisi** (8 partisipan `P01`–`P08`, masing-masing 10 angka × 3 ulangan = 30 kondisi) yang digabungkan dari dua arsip eksperimen lengkap tanggal **18 November 2026**:
1. **`sesi-20261118-140239`** (+ rekaman web `tim29_web-20261118-140243-ca54.tar`): dipakai untuk kondisi **`no` 1–180** (partisipan `P01`–`P06`, 180 kondisi).
2. **`sesi-20261118-145349`** (+ rekaman web `tim29_web-20261118-145356-6c85.tar`): dipakai untuk kondisi **`no` 181–240** (lanjutan 60 kondisi pada partisipan `P05` dan `P06` sesuai penanda komputer meja `lanjutan #207 · dari 181`).

## 2. Sesi & Bagian yang Dikeluarkan Beserta Alasannya
- **`sesi-20261113-153510` (13 Nov 2026, 30/240 kondisi):** Dikeluarkan karena sesi dibatalkan sebelum satu blok penuh selesai.
- **`sesi-20261113-154949` (13 Nov 2026, 240/240 kondisi):** Dikeluarkan sebagai sumber utama karena berkas kamera dihapus dari komputer meja atas permintaan pengelola (`MEDIA-DIHAPUS.txt`), sehingga hanya memuat data sensor tanpa citra isyarat tangan.
- **Kondisi `no` 181–240 pada akhir `sesi-20261118-140239`:** Digantikan oleh sesi lanjutan `sesi-20261118-145349` karena pada akhir perekaman pertama kamera menghasilkan frame kosong (`2.558` byte mulai `f002421.jpg`).

## 3. Hasil Pemeriksaan Umum & Khusus Tim (Per Partisipan)
- **Kelengkapan frame kamera:** Seluruh 240 kondisi gabungan memiliki tepat **3 frame citra valid** (`640×480` per ~1 detik selama 3 detik kondisi; frame ke-0 di detik pertama dibuang, 2 frame aktif berikutnya dipakai).
- **Keseimbangan desain:** Setiap partisipan memiliki tepat **30 kondisi** (`10 angka × 3 ulangan`).
- **Verifikasi label (`data/periksa-isyarat.csv`):** Total kondisi sesuai = **240 / 240**; kondisi ditandai tidak sesuai = **0**.

| Partisipan | Total Kondisi | Angka Unik | Sesuai (`sesuai=1`) | Tidak Sesuai (`sesuai=0`) |
|---|---:|---:|---:|---:|
| `P01` | 30 | 10 | 30 | 0 |
| `P02` | 30 | 10 | 30 | 0 |
| `P03` | 30 | 10 | 30 | 0 |
| `P04` | 30 | 10 | 30 | 0 |
| `P05` | 30 | 10 | 30 | 0 |
| `P06` | 30 | 10 | 30 | 0 |
| `P07` | 30 | 10 | 30 | 0 |
| `P08` | 30 | 10 | 30 | 0 |
