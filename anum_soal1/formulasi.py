"""
Formulasi SPL dan penanganan singularitas (Tugas ii).

    T^T pi = pi,  1^T pi = 1,  pi >= 0
    A = I - T^T  (singular karena 1^T A = 0^T)
    B = A dengan baris pertama diganti e_1^T,  b = e_1
    B z = b,  lalu  pi = z / (1^T z)
"""

import numpy as np


def bentuk_A(T):
    """A = I - T^T (matriks singular)."""
    N = T.shape[0]
    return np.eye(N) - T.T


def bentuk_B_b(T):
    """
    Membentuk matriks nonsingular B dan vektor ruas kanan b:
        B[0, :]  = [1, 0, ..., 0]
        B[i, :]  = A[i, :],   i = 1, ..., N-1   (indeks 0-based)
        b        = [1, 0, ..., 0]^T
    """
    B = bentuk_A(T)
    B[0, :] = 0.0
    B[0, 0] = 1.0
    b = np.zeros(T.shape[0])
    b[0] = 1.0
    return B, b


def normalisasi(z):
    """pi = z / (1^T z) agar jumlah seluruh komponen tepat 1."""
    return z / np.sum(z)
