# Protokol Evaluasi Tim 29 — Ruang Cerdas 5 Volt

Diisi Penjaga Data & Evaluasi (`123450111` — Zailani Satria), digabung ke `main` lewat *pull request* yang disetujui pembimbing, lalu **dibekukan**.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Pengenalan angka bahasa isyarat (BISINDO 0–9) |
| Sumber dataset | `https://github.com/sains-data/capstone-dl-2026-starter` & Meja 5 Volt Ruang Cerdas |
| Jenis tugas (`TUGAS`) | `klasifikasi` (10 kelas angka `0..9`) |
| Target / label | Kolom `angka` (`0..9`) terverifikasi pada `data/periksa-isyarat.csv` |
| Ukuran data latih / validasi / uji | 180 kondisi (6 partisipan) / 30 kondisi (1 partisipan) / 30 kondisi (1 partisipan) |
| Cara split | Pemisahan per partisipan (*subject-independent split*), seed `2026` |
| Metrik utama untuk komparasi | F1 per partisipan (`macro-F1`, rata-rata ± simpangan baku dari 3 seed), ukuran ONNX (KB), latensi (ms) |
| Praproses yang sama untuk semua model | Sampling citra `640×480` setiap `0,1` detik (10 FPS); **buang 1 detik pertama setiap kondisi** (10 frame awal), ambil 20 frame aktif (`21 titik tangan × 3 koordinat = 63 fitur` & potongan citra tangan); pencerminan sumbu-$x$ tangan kiri; normalisasi Z-score dihitung **dari data latih saja** |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop |
| Evaluasi data uji | Sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | Dihitung otomatis dari `data/MANIFEST.json` oleh `src.common.manifest` |