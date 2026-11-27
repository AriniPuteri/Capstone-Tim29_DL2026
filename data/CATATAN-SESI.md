# Catatan Sesi & Pemeriksaan Mutu Data — Tim 29 (BISINDO 0–9)

Disusun oleh Penjaga Data & Evaluasi (`123450111` — Zailani Satria).

## 1. Sesi yang Dipakai
Data primer Tim 29 terdiri atas **240 kondisi** (8 partisipan `P01`–`P08`, masing-masing 10 angka × 3 ulangan = 30 kondisi) yang digabungkan dari dua arsip eksperimen lengkap tanggal **18 November 2026**:
1. **`sesi-20261118-140239`** (+ rekaman web `tim29_web-20261118-140243-ca54.tar`): dipakai untuk kondisi **`no` 1–180** (partisipan `P01`–`P04`, `P07`, `P08` — 180 kondisi).
2. **`sesi-20261118-145349`** (+ rekaman web `tim29_web-20261118-145356-6c85.tar`): dipakai untuk kondisi **`no` 181–240** (lanjutan 60 kondisi pada partisipan `P05` dan `P06` sesuai penanda komputer meja `lanjutan #207 · dari 181`).

## 2. Sesi & Bagian yang Dikeluarkan Beserta Alasannya
- **`sesi-20261113-153510` (13 Nov 2026, 30/240 kondisi):** Dikeluarkan karena sesi dibatalkan sebelum satu blok penuh selesai.
- **`sesi-20261113-154949` (13 Nov 2026, 240/240 kondisi):** Dikeluarkan sebagai sumber utama karena berkas kamera dihapus dari komputer meja atas permintaan pengelola (`MEDIA-DIHAPUS.txt`), sehingga hanya memuat data sensor tanpa citra isyarat tangan.
- **Kondisi `no` 181–240 pada akhir `sesi-20261118-140239`:** Digantikan oleh sesi lanjutan `sesi-20261118-145349` karena pada akhir perekaman pertama kamera menghasilkan frame kosong (`2.558` byte mulai `f002421.jpg`).

## 3. Hasil Pemeriksaan Umum & Verifikasi Visual Citra (`data/periksa-isyarat.csv`)
- **Kelengkapan frame kamera:** Seluruh 240 kondisi gabungan memiliki tepat **3 frame citra valid** (`640×480` per ~1 detik selama 3 detik kondisi; frame ke-0 di detik pertama dibuang, 2 frame aktif berikutnya dipakai).
- **Keseimbangan desain:** Setiap partisipan memiliki tepat **30 kondisi** (`10 angka × 3 ulangan`).
- **Hasil pemeriksaan ulang citra per kondisi:** Total kondisi sesuai = **223 / 240** (`92,92%`); kondisi tidak sesuai = **17 / 240** (`no`: `[1, 2, 3, 31, 32, 33, 61, 91, 92, 121, 122, 123, 181, 182, 183, 211, 212]`), seluruhnya terjadi pada 1–3 percobaan pembuka awal blok saat partisipan baru bersiap dan tangan belum terangkat ke depan kamera.

| Partisipan | Total Kondisi | Angka Unik | Sesuai (`sesuai=1`) | Tidak Sesuai (`sesuai=0`) | Catatan Pemeriksaan Visual |
|---|---:|---:|---:|---:|---|
| `P01` | 30 | 10 | 27 | 3 | 3 kondisi awal blok (`no` 1–3) tangan belum terangkat; 27 kondisi sesuai. |
| `P02` | 30 | 10 | 28 | 2 | 2 kondisi awal blok (`no` 91–92) tangan belum terangkat; 28 kondisi sesuai. |
| `P03` | 30 | 10 | 29 | 1 | 1 kondisi awal blok (`no` 61) partisipan masih bersiap/tangan belum terangkat; 29 kondisi sesuai. |
| `P04` | 30 | 10 | 27 | 3 | 3 kondisi awal blok (`no` 31–33) tangan belum terangkat/menunduk; 27 kondisi sesuai. |
| `P05` | 30 | 10 | 28 | 2 | 2 kondisi awal blok lanjutan (`no` 211–212) tangan belum terangkat; 28 kondisi sesuai. |
| `P06` | 30 | 10 | 27 | 3 | 3 kondisi awal blok lanjutan (`no` 181–183) tangan belum terangkat; 27 kondisi sesuai. |
| `P07` | 30 | 10 | 27 | 3 | 3 kondisi awal blok (`no` 121–123) tangan belum terangkat; 27 kondisi sesuai. |
| `P08` | 30 | 10 | 30 | 0 | Seluruh 30 kondisi (`no` 151–180) peragaan tangan jelas dan sesuai. |
