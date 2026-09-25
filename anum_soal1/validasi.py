"""
Validasi data matriks transisi (Tugas i).

Pemeriksaan yang dilakukan untuk setiap T:
1. Dimensi     : persegi N x N dan N sesuai nama berkas.
2. Nilai hingga: tidak ada NaN / inf.
3. Nonnegatif  : T_ij >= 0 untuk semua i, j.
4. Stokastik   : setiap baris berjumlah 1 (dengan toleransi pembulatan).
5. Syarat steady state tunggal:
   - irreducible  : graf transisi (i -> j jika T_ij > 0) terhubung kuat,
                    sehingga hanya ada satu kelas komunikasi. Ini menjamin
                    distribusi stasioner tunggal dengan pi_i > 0.
   - aperiodik    : periode rantai = 1 (dihitung eksak via BFS). Bersama
                    irreducible, ini menjamin distribusi setelah waktu lama
                    benar-benar konvergen ke pi dari kondisi awal mana pun.
"""

from math import gcd

import numpy as np


def _tetangga(T):
    """Daftar ketetanggaan graf berarah: i -> j jika T_ij > 0."""
    N = T.shape[0]
    return [np.nonzero(T[i] > 0.0)[0].tolist() for i in range(N)]


def _bfs(adj, sumber=0):
    """BFS; mengembalikan level (jarak) tiap simpul, -1 jika tak terjangkau."""
    N = len(adj)
    level = [-1] * N
    level[sumber] = 0
    antrean = [sumber]
    kepala = 0
    while kepala < len(antrean):
        u = antrean[kepala]
        kepala += 1
        for v in adj[u]:
            if level[v] < 0:
                level[v] = level[u] + 1
                antrean.append(v)
    return level


def cek_irreducible(T):
    """
    T irreducible  <=>  dari simpul 0 semua simpul terjangkau (graf T) dan
    simpul 0 terjangkau dari semua simpul (graf transpos / busur dibalik).
    """
    adj = _tetangga(T)
    maju = _bfs(adj, 0)
    adj_balik = [[] for _ in range(len(adj))]
    for u, daftar in enumerate(adj):
        for v in daftar:
            adj_balik[v].append(u)
    mundur = _bfs(adj_balik, 0)
    return all(x >= 0 for x in maju) and all(x >= 0 for x in mundur)


def periode_rantai(T):
    """
    Periode rantai Markov irreducible:
        d = gcd{ level(u) + 1 - level(v) : T_uv > 0 }
    dengan level = jarak BFS dari simpul 0. d = 1 berarti aperiodik.
    """
    adj = _tetangga(T)
    level = _bfs(adj, 0)
    d = 0
    for u, daftar in enumerate(adj):
        for v in daftar:
            d = gcd(d, abs(level[u] + 1 - level[v]))
    return d


def aliran_potongan_min(T):
    """
    Indikator 'hampir tereduksi' (nearly reducible). Untuk setiap potongan k
    (antara halte k dan k+1, 1-based) dihitung peluang total menyeberang:
        maju_k   = sum_{i <= k < j} T_ij,   mundur_k = sum_{j <= k < i} T_ij.
    Jika salah satu bernilai 0, koridor terputus (reducible). Jika sangat
    kecil, rantai tetap irreducible tetapi hampir terputus sehingga masalah
    steady state menjadi sensitif (ill-conditioned).
    """
    N = T.shape[0]
    maju = np.array([T[:k + 1, k + 1:].sum() for k in range(N - 1)])
    mundur = np.array([T[k + 1:, :k + 1].sum() for k in range(N - 1)])
    km, kb = int(np.argmin(maju)), int(np.argmin(mundur))
    return {
        "maju_min": float(maju[km]), "potongan_maju_min": km + 1,
        "mundur_min": float(mundur[kb]), "potongan_mundur_min": kb + 1,
    }


def validasi_T(T, N_diharapkan=None, tol_jumlah=1e-12):
    """
    Menjalankan seluruh pemeriksaan pada matriks transisi T.

    Returns
    -------
    dict berisi hasil tiap pemeriksaan dan flag 'valid' keseluruhan.
    """
    hasil = {}
    n_baris, n_kolom = T.shape
    hasil["dimensi"] = (n_baris, n_kolom)
    hasil["persegi"] = n_baris == n_kolom
    hasil["sesuai_nama_berkas"] = (N_diharapkan is None) or (n_baris == n_kolom == N_diharapkan)

    hasil["semua_hingga"] = bool(np.all(np.isfinite(T)))

    hasil["min_elemen"] = float(T.min())
    hasil["jumlah_negatif"] = int(np.sum(T < 0.0))
    hasil["nonnegatif"] = hasil["jumlah_negatif"] == 0

    jumlah_baris = T.sum(axis=1)
    hasil["maks_dev_jumlah_baris"] = float(np.max(np.abs(jumlah_baris - 1.0)))
    hasil["stokastik_baris"] = hasil["maks_dev_jumlah_baris"] <= tol_jumlah

    hasil["nnz"] = int(np.count_nonzero(T))
    diag = np.array([T[i, i] for i in range(min(n_baris, n_kolom))])
    hasil["diagonal_positif"] = int(np.sum(diag > 0.0))
    hasil["maks_diagonal"] = float(diag.max())

    hasil["irreducible"] = bool(cek_irreducible(T)) if hasil["persegi"] else False
    hasil["periode"] = periode_rantai(T) if hasil["irreducible"] else None
    hasil["aperiodik"] = hasil["periode"] == 1
    if hasil["persegi"]:
        hasil.update(aliran_potongan_min(T))

    hasil["steady_state_tunggal"] = (
        hasil["persegi"] and hasil["semua_hingga"] and hasil["nonnegatif"]
        and hasil["stokastik_baris"] and hasil["irreducible"]
    )
    hasil["valid"] = (
        hasil["sesuai_nama_berkas"] and hasil["steady_state_tunggal"] and hasil["aperiodik"]
    )
    return hasil
