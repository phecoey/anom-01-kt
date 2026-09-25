"""
Identifikasi struktur matriks (Tugas iii).

Konvensi penyimpanan band (dipakai solver banded):
    W berukuran N x (2p + q + 1),  W[i, p + (j - i)] = B[i, j]
untuk offset diagonal -p <= j - i <= p + q. Artinya setiap KOLOM W adalah
satu vektor diagonal:
    kolom 0 .. p-1          : subdiagonal (offset -p .. -1)
    kolom p                 : diagonal utama
    kolom p+1 .. p+q        : superdiagonal asli (offset 1 .. q)
    kolom p+q+1 .. 2p+q     : ruang fill-in akibat partial pivoting
                              (offset q+1 .. p+q), awalnya nol
Tidak ada matriks N x N yang dibentuk.
"""

import numpy as np


def deteksi_bandwidth(M, tol=0.0):
    """
    Deteksi otomatis lower bandwidth p dan upper bandwidth q:
        p = max{ i - j : |M_ij| > tol },   q = max{ j - i : |M_ij| > tol }.
    """
    baris, kolom = np.nonzero(np.abs(M) > tol)
    if baris.size == 0:
        return 0, 0
    offset = kolom - baris
    p = int(max(0, -offset.min()))
    q = int(max(0, offset.max()))
    return p, q


def bandwidth_B_dari_nonzero(N, entri):
    """
    Menghitung (p, q) matriks B langsung dari elemen tak nol T, tanpa
    membentuk B. Karena B[i, j] = delta_ij - T[j, i] untuk i >= 1, elemen
    T[r, c] != 0 menempati posisi B[c, r] (offset kolom-baris = r - c).
    Baris 0 dari B diganti e_1^T sehingga elemen T[r, 0] diabaikan.
    """
    p = q = 0
    for r, c, _ in entri:
        if c == 0 or r == c:
            continue
        off = r - c
        if off < 0:
            p = max(p, -off)
        else:
            q = max(q, off)
    return p, q


def _alokasi_band(N, p, q):
    return np.zeros((N, 2 * p + q + 1), dtype=np.float64)


def band_B_dari_T(T, p, q):
    """Bangun penyimpanan band W untuk B langsung dari T (tanpa membentuk B)."""
    N = T.shape[0]
    W = _alokasi_band(N, p, q)
    W[0, p] = 1.0                      # baris pertama B = e_1^T
    for i in range(1, N):
        for j in range(max(0, i - p), min(N, i + q + 1)):
            W[i, p + j - i] = (1.0 if i == j else 0.0) - T[j, i]
    return W


def band_B_dari_nonzero(N, entri, p, q):
    """Bangun W untuk B dari daftar elemen tak nol T (jalur hemat memori)."""
    W = _alokasi_band(N, p, q)
    for i in range(1, N):
        W[i, p] = 1.0                  # kontribusi identitas pada diagonal
    for r, c, v in entri:              # T[r, c] -> B[c, r] = -T[r, c] (c >= 1)
        if c == 0:
            continue
        W[c, p + r - c] -= v
    W[0, p] = 1.0
    return W


def band_dari_dense(M, p, q):
    """Salin elemen band dari matriks dense M ke format W (untuk pengujian)."""
    N = M.shape[0]
    W = _alokasi_band(N, p, q)
    for i in range(N):
        for j in range(max(0, i - p), min(N, i + q + 1)):
            W[i, p + j - i] = M[i, j]
    return W


def kebutuhan_penyimpanan(N, p, q, bytes_per_elemen=8):
    """
    Perbandingan kebutuhan penyimpanan (jumlah elemen dan byte float64):
      dense        : N^2
      band_eksak   : banyaknya posisi di dalam band
                     = (p+q+1) N - p(p+1)/2 - q(q+1)/2
      band_vektor  : (p+q+1) vektor diagonal panjang N
      band_pivot   : (2p+q+1) vektor, termasuk ruang fill-in partial pivoting
    """
    dense = N * N
    band_eksak = (p + q + 1) * N - p * (p + 1) // 2 - q * (q + 1) // 2
    band_vektor = (p + q + 1) * N
    band_pivot = (2 * p + q + 1) * N
    return {
        "N": N,
        "p": p,
        "q": q,
        "dense_elemen": dense,
        "band_eksak_elemen": band_eksak,
        "band_vektor_elemen": band_vektor,
        "band_pivot_elemen": band_pivot,
        "dense_byte": dense * bytes_per_elemen,
        "band_eksak_byte": band_eksak * bytes_per_elemen,
        "band_pivot_byte": band_pivot * bytes_per_elemen,
        "rasio_dense_per_band_pivot": dense / band_pivot,
        "persen_band_eksak": 100.0 * band_eksak / dense,
    }
