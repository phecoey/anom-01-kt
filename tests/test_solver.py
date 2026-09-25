"""
Uji kebenaran solver pada ukuran kecil (bagian dari Tugas iv).

Jalankan dari folder proyek:
    python -m tests.test_solver          (tanpa pytest)
    python -m pytest tests -q            (jika pytest tersedia)

Semua pembanding memakai perhitungan analitik atau residual
(perkalian matriks-vektor), bukan solver pustaka.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anum_soal1 import (  # noqa: E402
    baca_T_dense, bentuk_A, bentuk_B_b, deteksi_bandwidth, band_dari_dense,
    lu_dense_faktorisasi, lu_dense_solve, lu_band_faktorisasi, lu_band_solve,
    rekonstruksi_PLU_band, steady_state_dense, steady_state_band, validasi_T,
    MatriksSingularError,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "A")
EPS = np.finfo(float).eps


def rantai_birth_death(N=6):
    """Rantai birth-death (tridiagonal) dengan pi analitik via detailed balance."""
    naik = np.linspace(0.30, 0.10, N - 1)      # T[i, i+1]
    turun = np.linspace(0.15, 0.35, N - 1)     # T[i+1, i]
    T = np.zeros((N, N))
    for i in range(N - 1):
        T[i, i + 1] = naik[i]
        T[i + 1, i] = turun[i]
    for i in range(N):
        T[i, i] = 1.0 - T[i].sum()
    pi = np.ones(N)
    for i in range(N - 1):                      # pi_{i+1} T_{i+1,i} = pi_i T_{i,i+1}
        pi[i + 1] = pi[i] * naik[i] / turun[i]
    return T, pi / pi.sum()


def rantai_doubly_stochastic(N=8):
    """T simetris (baris & kolom berjumlah 1) => pi = 1/N (seragam)."""
    T = np.zeros((N, N))
    for i in range(N - 1):
        T[i, i + 1] = T[i + 1, i] = 0.2
    for i in range(N):
        T[i, i] = 1.0 - T[i].sum()
    return T, np.full(N, 1.0 / N)


def matriks_band_acak(N, p, q, seed):
    """Matriks band acak dengan diagonal kecil agar partial pivoting aktif."""
    rng = np.random.default_rng(seed)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(max(0, i - p), min(N, i + q + 1)):
            M[i, j] = rng.uniform(-1.0, 1.0)
        M[i, i] *= 0.05
    return M


def test_birth_death_analitik():
    T, pi_eksak = rantai_birth_death(6)
    assert validasi_T(T, 6)["valid"]
    for pi in (steady_state_dense(T)["pi"], steady_state_band(T=T)["pi"]):
        assert np.max(np.abs(pi - pi_eksak)) < 10 * EPS


def test_doubly_stochastic_seragam():
    T, pi_eksak = rantai_doubly_stochastic(8)
    for pi in (steady_state_dense(T)["pi"], steady_state_band(T=T)["pi"]):
        assert np.max(np.abs(pi - pi_eksak)) < 10 * EPS


def _cek_pivoting(p, q, seed):
    N = 10
    M = matriks_band_acak(N, p, q, seed)
    assert deteksi_bandwidth(M) == (p, q)
    x_benar = np.arange(1.0, N + 1.0)
    b = M @ x_benar
    # solver dense
    LU, perm, info_d = lu_dense_faktorisasi(M)
    x_d = lu_dense_solve(LU, perm, b)
    L = np.tril(LU, -1) + np.eye(N)
    U = np.triu(LU)
    assert np.max(np.abs(M[perm] - L @ U)) < 50 * EPS
    # solver band
    Wf, piv, info_b = lu_band_faktorisasi(band_dari_dense(M, p, q), p, q)
    x_b = lu_band_solve(Wf, piv, p, q, b)
    P, Lb, Ub = rekonstruksi_PLU_band(Wf, piv, p, q)
    assert np.max(np.abs(P @ M - Lb @ Ub)) < 50 * EPS
    assert info_d["jumlah_tukar"] > 0 and info_b["jumlah_tukar"] > 0
    assert info_d["jumlah_tukar"] == info_b["jumlah_tukar"]
    assert np.max(np.abs(x_d - x_b)) < 1e-10
    assert np.max(np.abs(x_b - x_benar)) < 1e-9
    return info_d["jumlah_tukar"]


def test_pivoting_p1_q2():
    _cek_pivoting(1, 2, seed=7)


def test_pivoting_p2_q1():
    _cek_pivoting(2, 1, seed=11)


def test_data_A_N16():
    T = baca_T_dense(os.path.join(DATA, "T_16.csv"))
    B, b = bentuk_B_b(T)
    hd = steady_state_dense(T)
    hb = steady_state_band(path=os.path.join(DATA, "T_16.csv"))
    assert (hb["p"], hb["q"]) == deteksi_bandwidth(B) == (1, 2)
    for h in (hd, hb):
        assert abs(h["pi"].sum() - 1.0) < 10 * EPS
        assert np.all(h["pi"] > 0)
        assert np.max(np.abs(B @ h["z"] - b)) < 10 * EPS
        assert np.max(np.abs(T.T @ h["pi"] - h["pi"])) < 10 * EPS
    assert np.max(np.abs(hd["pi"] - hb["pi"])) < 10 * EPS
    P, L, U = rekonstruksi_PLU_band(hb["W"], hb["piv"], hb["p"], hb["q"])
    assert np.max(np.abs(P @ B - L @ U)) < 10 * EPS


def test_A_singular():
    T = baca_T_dense(os.path.join(DATA, "T_16.csv"))
    A = bentuk_A(T)
    assert np.max(np.abs(A.sum(axis=0))) < 10 * EPS       # 1^T A = 0^T
    try:
        LU, perm, info = lu_dense_faktorisasi(A)
        assert info["pivot"][-1] < 1e-12                  # pivot terakhir ~ 0
    except MatriksSingularError:
        pass


if __name__ == "__main__":
    daftar = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in daftar:
        f()
        print(f"LULUS  {f.__name__}")
    print(f"{len(daftar)} uji lulus")
