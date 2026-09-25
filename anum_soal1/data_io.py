"""
Pembacaan data matriks transisi T_N.csv.

Dua cara membaca disediakan:
1. baca_T_dense   : membaca seluruh CSV menjadi array N x N (dipakai untuk
                    validasi data dan solver dense).
2. baca_T_nonzero : membaca CSV baris per baris dan hanya menyimpan elemen
                    tak nol. Dipakai oleh jalur solver banded agar tidak
                    pernah ada array berukuran N x N di memori.
"""

import csv
import os

import numpy as np

UKURAN_N = (16, 32, 64, 128, 256, 512)


def path_T(data_dir, N):
    """Path berkas T_N.csv di dalam folder data."""
    return os.path.join(data_dir, f"T_{N}.csv")


def _baris_csv(path):
    """Generator baris CSV (list string), melewati baris kosong."""
    with open(path, newline="") as f:
        for baris in csv.reader(f):
            if not baris or all(s.strip() == "" for s in baris):
                continue
            yield baris


def baca_T_dense(path):
    """
    Membaca T_N.csv menjadi array float64 berukuran (baris, kolom).

    Setiap baris diperiksa panjangnya, sehingga CSV yang tidak persegi
    panjang (jumlah kolom tidak seragam) langsung terdeteksi.
    """
    data = []
    panjang = set()
    for baris in _baris_csv(path):
        nilai = [float(s) for s in baris]
        panjang.add(len(nilai))
        data.append(nilai)
    if len(panjang) != 1:
        raise ValueError(f"{path}: jumlah kolom tiap baris tidak seragam {sorted(panjang)}")
    return np.array(data, dtype=np.float64)


def baca_T_nonzero(path):
    """
    Membaca T_N.csv secara streaming dan hanya menyimpan elemen tak nol.

    Returns
    -------
    N : int
        Ukuran matriks.
    entri : list of (i, j, T_ij)
        Seluruh elemen tak nol (indeks 0-based). Ukurannya O(nnz), untuk
        matriks banded nnz = O((p + q + 1) N).
    """
    entri = []
    n_baris = 0
    n_kolom = None
    for i, baris in enumerate(_baris_csv(path)):
        if n_kolom is None:
            n_kolom = len(baris)
        elif len(baris) != n_kolom:
            raise ValueError(f"{path}: baris {i} memiliki {len(baris)} kolom, seharusnya {n_kolom}")
        for j, s in enumerate(baris):
            v = float(s)
            if v != 0.0:
                entri.append((i, j, v))
        n_baris += 1
    if n_baris != n_kolom:
        raise ValueError(f"{path}: matriks tidak persegi ({n_baris} x {n_kolom})")
    return n_baris, entri
