# Protokol Evaluasi Tim 29 — Ruang Cerdas 5 Volt

Diisi Penjaga Data & Evaluasi (`123450111` — Zailani Satria), digabung ke `main` lewat *pull request* yang disetujui pembimbing, lalu **dibekukan**.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Pengenalan angka bahasa isyarat (BISINDO 0–9) |
| Sumber dataset | Meja 5 Volt Ruang Cerdas — Tim 29 (`sesi-20261118-140239` & `sesi-20261118-145349`) |
| Jenis tugas (`TUGAS`) | `klasifikasi` (10 kelas angka `0..9`) |
| Target / label | Kolom `angka` (`0..9`) terverifikasi pada `data/periksa-isyarat.csv` |
| Ukuran data latih / validasi / uji | 180 kondisi (`P01`–`P06`) / 30 kondisi (`P07`) / 30 kondisi (`P08`) |
| Cara split | Pemisahan per partisipan (*subject-independent split*), seed `2026` |
| Metrik utama untuk komparasi | F1 per partisipan (`macro-F1`, rata-rata ± simpangan baku dari 3 seed), ukuran ONNX (KB), latensi (ms) |
| Praproses yang sama untuk semua model | Sampling citra `640×480` selama 3 detik per kondisi; **buang 1 detik pertama setiap kondisi**, ambil jendela aktif (`63` fitur citra spasial-gradien per frame aktif); normalisasi Z-score dihitung **dari data latih saja** |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop |
| Evaluasi data uji | Sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | `ce33052f74683c8d40c94c758a3cda61` (dihitung dari `data/MANIFEST.json`) |
