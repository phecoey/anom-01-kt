"""
Implementasi solver SPL B z = b (Tugas iv), tanpa pustaka SPL/faktorisasi.

(a) LU dense dengan partial pivoting:  P B = L U
    - B disimpan penuh (N x N). Faktor L (unit lower) dan U ditimpa
      di tempat pada array yang sama (format "packed").
    - P disimpan sebagai vektor permutasi perm, (P B)[i, :] = B[perm[i], :].

(b) Solver banded bergaya algoritma Thomas dengan partial pivoting.
    - Hanya menyimpan vektor-vektor diagonal (kolom array W, lihat
      struktur.py) ditambah vektor ruas kanan dan vektor indeks pivot.
    - Pivot di kolom k hanya dicari pada baris k .. k+p (elemen tak nol
      di bawah diagonal paling jauh p baris).
    - Pertukaran baris dapat menggeser elemen U sampai offset p + q
      (fill-in), sehingga disediakan p vektor superdiagonal tambahan.
    - Untuk p = q = 1 algoritma ini tepat menjadi algoritma Thomas dengan
      partial pivoting (setara LAPACK gtsv): vektor sub, diag, super,
      dan satu vektor fill-in.

Setiap fungsi faktorisasi mengembalikan statistik: jumlah pertukaran
baris, jumlah operasi floating point (flop: +, -, *, /), dan pivot.
"""

import numpy as np

from .formulasi import bentuk_B_b, normalisasi
from .struktur import band_B_dari_T, band_B_dari_nonzero, bandwidth_B_dari_nonzero
from .data_io import baca_T_nonzero


class MatriksSingularError(ArithmeticError):
    """Pivot bernilai nol: matriks singular (secara eksak)."""


# ---------------------------------------------------------------------------
# (a) LU dense dengan partial pivoting
# ---------------------------------------------------------------------------
def lu_dense_faktorisasi(B, izinkan_pivot_nol=False):
    """
    Faktorisasi P B = L U dengan partial pivoting (eliminasi Gauss).

    Pseudocode (0-based):
        perm = [0, 1, ..., N-1]
        untuk k = 0 .. N-2:
            r = argmax_{i >= k} |LU[i, k]|          # partial pivoting
            tukar baris k dan r (seluruh kolom) serta perm[k], perm[r]
            LU[k+1:, k]    = LU[k+1:, k] / LU[k, k]   # pengali (kolom L)
            LU[k+1:, k+1:] -= LU[k+1:, k] * LU[k, k+1:]  # update rank-1

    Returns
    -------
    LU : ndarray (N, N)   L (di bawah diagonal, diagonal 1 implisit) dan U
    perm : ndarray (N,)   vektor permutasi, P B = B[perm]
    info : dict           jumlah_tukar, flop, pivot (|u_kk|)
    """
    LU = np.array(B, dtype=np.float64, copy=True)
    N = LU.shape[0]
    perm = np.arange(N)
    jumlah_tukar = 0
    flop = 0
    for k in range(N - 1):
        r = k + int(np.argmax(np.abs(LU[k:, k])))
        if LU[r, k] == 0.0:
            if izinkan_pivot_nol:
                continue
            raise MatriksSingularError(f"pivot nol pada kolom {k}")
        if r != k:
            LU[[k, r], :] = LU[[r, k], :]
            perm[[k, r]] = perm[[r, k]]
            jumlah_tukar += 1
        m = N - k - 1
        LU[k + 1:, k] /= LU[k, k]
        LU[k + 1:, k + 1:] -= np.outer(LU[k + 1:, k], LU[k, k + 1:])
        flop += m + 2 * m * m
    if LU[N - 1, N - 1] == 0.0 and not izinkan_pivot_nol:
        raise MatriksSingularError(f"pivot nol pada kolom {N - 1}")
    pivot = np.abs(np.array([LU[i, i] for i in range(N)]))
    info = {"jumlah_tukar": jumlah_tukar, "flop": flop, "pivot": pivot}
    return LU, perm, info


def lu_dense_solve(LU, perm, b, info=None):
    """
    Menyelesaikan B z = b memakai faktor P B = L U:
        L y = P b   (substitusi maju, L unit lower triangular)
        U z = y     (substitusi mundur)
    """
    N = LU.shape[0]
    y = np.array(b, dtype=np.float64)[perm]
    flop = 0
    for i in range(1, N):
        y[i] -= LU[i, :i] @ y[:i]
        flop += 2 * i
    for i in range(N - 1, -1, -1):
        y[i] = (y[i] - LU[i, i + 1:] @ y[i + 1:]) / LU[i, i]
        flop += 2 * (N - 1 - i) + 1
    if info is not None:
        info["flop_solve"] = flop
    return y


# ---------------------------------------------------------------------------
# (b) Solver banded bergaya Thomas dengan partial pivoting
# ---------------------------------------------------------------------------
def lu_band_faktorisasi(W, p, q, inplace=False):
    """
    Faktorisasi LU banded dengan partial pivoting pada penyimpanan vektor
    diagonal W (N x (2p+q+1)), W[i, p + j - i] = B[i, j].

    Pseudocode (0-based, m = p + q):
        untuk k = 0 .. N-1:
            r = argmax_{k <= i <= min(k+p, N-1)} |B[i, k]|   # pivot lokal
            jika r != k: tukar B[k, k..k+m] <-> B[r, k..k+m]
            piv[k] = r
            untuk i = k+1 .. min(k+p, N-1):
                l = B[i, k] / B[k, k];  B[i, k] = l           # simpan L
                B[i, k+1..k+m] -= l * B[k, k+1..k+m]         # update band
    Akses B[i, j] selalu melalui W[i, p + j - i].

    Returns
    -------
    W : ndarray  faktor L (kolom < p) dan U (kolom >= p, lebar p+q)
    piv : ndarray (N,)  piv[k] = baris yang ditukar dengan baris k
    info : dict   jumlah_tukar, flop, pivot
    """
    if not inplace:
        W = W.copy()
    N = W.shape[0]
    m = p + q
    piv = np.arange(N)
    jumlah_tukar = 0
    flop = 0
    pivot = np.empty(N)
    for k in range(N):
        akhir = min(k + p, N - 1)
        jmax = min(k + m, N - 1)
        # --- partial pivoting: bandingkan paling banyak p+1 kandidat
        r = k
        amax = abs(W[k, p])
        for i in range(k + 1, akhir + 1):
            v = abs(W[i, p + k - i])
            if v > amax:
                amax = v
                r = i
        if amax == 0.0:
            raise MatriksSingularError(f"pivot nol pada kolom {k}")
        piv[k] = r
        if r != k:
            for j in range(k, jmax + 1):
                a, c = p + j - k, p + j - r
                W[k, a], W[r, c] = W[r, c], W[k, a]
            jumlah_tukar += 1
        # --- eliminasi baris di bawah pivot (paling banyak p baris)
        ukk = W[k, p]
        pivot[k] = abs(ukk)
        for i in range(k + 1, akhir + 1):
            ci = p + k - i
            l = W[i, ci] / ukk
            W[i, ci] = l
            for j in range(k + 1, jmax + 1):
                W[i, p + j - i] -= l * W[k, p + j - k]
            flop += 1 + 2 * (jmax - k)
    info = {"jumlah_tukar": jumlah_tukar, "flop": flop, "pivot": pivot}
    return W, piv, info


def lu_band_solve(W, piv, p, q, b, info=None):
    """
    Menyelesaikan B z = b dari faktor banded (gaya Thomas):
      sapuan maju  : untuk k = 0..N-1, terapkan tukar (k, piv[k]) lalu
                     z[i] -= l_ik z[k] untuk i = k+1..k+p
      sapuan mundur: z[i] = (z[i] - sum_{j=i+1}^{i+p+q} u_ij z[j]) / u_ii
    """
    N = W.shape[0]
    m = p + q
    z = np.array(b, dtype=np.float64, copy=True)
    flop = 0
    for k in range(N):
        r = piv[k]
        if r != k:
            z[k], z[r] = z[r], z[k]
        zk = z[k]
        for i in range(k + 1, min(k + p, N - 1) + 1):
            z[i] -= W[i, p + k - i] * zk
            flop += 2
    for i in range(N - 1, -1, -1):
        s = z[i]
        for j in range(i + 1, min(i + m, N - 1) + 1):
            s -= W[i, p + j - i] * z[j]
            flop += 2
        z[i] = s / W[i, p]
        flop += 1
    if info is not None:
        info["flop_solve"] = flop
    return z


def rekonstruksi_PLU_band(W, piv, p, q):
    """
    HANYA untuk verifikasi pada N kecil: membentuk P, L, U dense dari faktor
    banded sehingga dapat dicek P B = L U. Pengali L yang tersimpan di W
    tidak ikut ditukar saat faktorisasi (gaya LAPACK), maka pertukaran
    baris diputar ulang di sini agar diperoleh L unit lower triangular.
    """
    N = W.shape[0]
    m = p + q
    L = np.zeros((N, N))
    perm = np.arange(N)
    for k in range(N):
        r = piv[k]
        if r != k:
            L[[k, r], :k] = L[[r, k], :k]
            perm[[k, r]] = perm[[r, k]]
        for i in range(k + 1, min(k + p, N - 1) + 1):
            L[i, k] = W[i, p + k - i]
    for i in range(N):
        L[i, i] = 1.0
    U = np.zeros((N, N))
    for i in range(N):
        for j in range(i, min(i + m, N - 1) + 1):
            U[i, j] = W[i, p + j - i]
    P = np.eye(N)[perm]
    return P, L, U


# ---------------------------------------------------------------------------
# Pembungkus tingkat tinggi: T -> pi
# ---------------------------------------------------------------------------
def steady_state_dense(T):
    """Hitung pi memakai solver LU dense. Mengembalikan dict hasil."""
    B, b = bentuk_B_b(T)
    LU, perm, info = lu_dense_faktorisasi(B)
    z = lu_dense_solve(LU, perm, b, info)
    return {"pi": normalisasi(z), "z": z, "LU": LU, "perm": perm, "info": info}


def siapkan_band(T):
    """
    Pra-proses untuk solver banded dari T dense yang sudah dibaca:
    deteksi (p, q) dari pola tak nol, bangun vektor-vektor diagonal W, dan b = e_1.
    Berguna untuk eksperimen waktu: ukur lu_band_faktorisasi + lu_band_solve saja.
    """
    N = T.shape[0]
    baris, kolom = np.nonzero(T)
    entri = [(int(r), int(c), 0.0) for r, c in zip(baris, kolom)]
    p, q = bandwidth_B_dari_nonzero(N, entri)
    W = band_B_dari_T(T, p, q)
    b = np.zeros(N)
    b[0] = 1.0
    return W, p, q, b


def steady_state_band(T=None, path=None):
    """
    Hitung pi memakai solver banded. Masukan salah satu:
      T    : matriks transisi dense (sudah dibaca), atau
      path : path CSV; dibaca streaming sehingga tidak ada array N x N.
    p dan q ditentukan otomatis dari pola tak nol B.
    """
    if path is not None:
        N, entri = baca_T_nonzero(path)
        p, q = bandwidth_B_dari_nonzero(N, entri)
        W = band_B_dari_nonzero(N, entri, p, q)
        b = np.zeros(N)
        b[0] = 1.0
    elif T is not None:
        W, p, q, b = siapkan_band(T)
    else:
        raise ValueError("berikan T atau path")
    Wf, piv, info = lu_band_faktorisasi(W, p, q, inplace=True)
    z = lu_band_solve(Wf, piv, p, q, b, info)
    return {"pi": normalisasi(z), "z": z, "W": Wf, "piv": piv, "p": p, "q": q, "info": info}
