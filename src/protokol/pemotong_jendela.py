from __future__ import annotations
import numpy as np

FPS = 10
DURASI_KONDISI_DETIK = 3.0
BUANG_AWAL_DETIK = 1.0
TOTAL_FRAME_MENTAH = int(DURASI_KONDISI_DETIK * FPS)      # 30 frame
FRAME_DIBUANG = int(BUANG_AWAL_DETIK * FPS)               # 10 frame pertama dibuang
PANJANG_JENDELA = TOTAL_FRAME_MENTAH - FRAME_DIBUANG      # 20 frame aktif
JUMLAH_TITIK_TANGAN = 21
DIMENSI_KOORDINAT = 3


def cerminkan_tangan_kiri(landmarks: np.ndarray, adalah_tangan_kiri: bool) -> np.ndarray:
    hasil = np.array(landmarks, dtype=np.float32, copy=True)
    if adalah_tangan_kiri:
        hasil[..., 0] = 1.0 - hasil[..., 0]
    return hasil


def potong_jendela_kondisi(
    frames_landmarks: np.ndarray,
    frames_crop: np.ndarray | None = None,
    adalah_tangan_kiri: bool = False,
):
    if frames_landmarks.shape[0] < TOTAL_FRAME_MENTAH:
        raise ValueError(f"Frame ({frames_landmarks.shape[0]}) kurang dari {TOTAL_FRAME_MENTAH}.")
    win_lm = cerminkan_tangan_kiri(
        frames_landmarks[FRAME_DIBUANG : FRAME_DIBUANG + PANJANG_JENDELA],
        adalah_tangan_kiri,
    )
    win_crop = None
    if frames_crop is not None:
        win_crop = np.array(frames_crop[FRAME_DIBUANG : FRAME_DIBUANG + PANJANG_JENDELA], copy=True)
        if adalah_tangan_kiri:
            win_crop = (
                np.flip(win_crop, axis=-1)
                if win_crop.ndim == 4 and win_crop.shape[1] in (1, 3)
                else np.flip(win_crop, axis=-2)
            )
    return win_lm, win_crop


def buat_sesi_simulasi(
    folder_simulasi,
    jumlah_partisipan: int = 8,
    jumlah_kelas: int = 10,
    ulangan: int = 3,
    seed: int = 2026,
):
    folder_simulasi.mkdir(parents=True, exist_ok=True)
    berkas_sesi = folder_simulasi / "sesi_simulasi.npz"
    if berkas_sesi.exists():
        return berkas_sesi
    rng = np.random.default_rng(seed)
    prototipe = rng.uniform(0.2, 0.8, size=(jumlah_kelas, JUMLAH_TITIK_TANGAN, DIMENSI_KOORDINAT)).astype(np.float32)
    X_raw, y_list, pid_list, kiri_list = [], [], [], []
    for pid in range(jumlah_partisipan):
        bias_p = rng.normal(0.0, 0.03, size=(1, JUMLAH_TITIK_TANGAN, DIMENSI_KOORDINAT)).astype(np.float32)
        kiri = bool(pid == 0)
        for angka in range(jumlah_kelas):
            for _ in range(ulangan):
                transisi = rng.uniform(0.0, 1.0, size=(FRAME_DIBUANG, JUMLAH_TITIK_TANGAN, DIMENSI_KOORDINAT)).astype(np.float32)
                aktif = np.clip(
                    prototipe[angka][None, :, :]
                    + bias_p
                    + rng.normal(0.0, 0.02, size=(PANJANG_JENDELA, JUMLAH_TITIK_TANGAN, DIMENSI_KOORDINAT)).astype(np.float32),
                    0.0,
                    1.0,
                )
                if kiri:
                    aktif[..., 0] = 1.0 - aktif[..., 0]
                X_raw.append(np.concatenate([transisi, aktif], axis=0))
                y_list.append(angka)
                pid_list.append(pid)
                kiri_list.append(kiri)
    np.savez_compressed(
        berkas_sesi,
        landmarks=np.asarray(X_raw, dtype=np.float32),
        angka=np.asarray(y_list, dtype=np.int64),
        partisipan=np.asarray(pid_list, dtype=np.int64),
        tangan_kiri=np.asarray(kiri_list, dtype=bool),
    )
    return berkas_sesi