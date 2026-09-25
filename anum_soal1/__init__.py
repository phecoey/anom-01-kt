"""
Paket kode Soal 1 TK 1 Analisis Numerik (Gasal 2026/2027).

Analisis perpindahan penumpang antarhalte: distribusi steady state
rantai Markov dengan matriks transisi banded.

Modul:
    data_io      : membaca T_N.csv (dense maupun streaming khusus elemen band)
    validasi     : pemeriksaan dimensi, nonnegativitas, jumlah baris,
                   irreducibility, dan aperiodisitas
    formulasi    : pembentukan A = I - T^T, matriks B, vektor b, normalisasi
    struktur     : deteksi otomatis bandwidth (p, q), penyimpanan band,
                   perbandingan kebutuhan memori dense vs band
    solver       : (a) LU dense dengan partial pivoting (PB = LU)
                   (b) solver banded bergaya algoritma Thomas dengan
                       partial pivoting (hanya menyimpan vektor diagonal)

Catatan: tidak ada fungsi pustaka untuk menyelesaikan SPL maupun
faktorisasi matriks (np.linalg / scipy.linalg tidak dipakai sama sekali).
NumPy hanya dipakai sebagai wadah array dan operasi aritmetika dasar.
"""

from .data_io import UKURAN_N, path_T, baca_T_dense, baca_T_nonzero
from .validasi import validasi_T
from .formulasi import bentuk_A, bentuk_B_b, normalisasi
from .struktur import (
    deteksi_bandwidth,
    bandwidth_B_dari_nonzero,
    band_B_dari_T,
    band_B_dari_nonzero,
    band_dari_dense,
    kebutuhan_penyimpanan,
)
from .solver import (
    MatriksSingularError,
    lu_dense_faktorisasi,
    lu_dense_solve,
    lu_band_faktorisasi,
    lu_band_solve,
    rekonstruksi_PLU_band,
    siapkan_band,
    steady_state_dense,
    steady_state_band,
)

__all__ = [
    "UKURAN_N", "path_T", "baca_T_dense", "baca_T_nonzero",
    "validasi_T",
    "bentuk_A", "bentuk_B_b", "normalisasi",
    "deteksi_bandwidth", "bandwidth_B_dari_nonzero", "band_B_dari_T",
    "band_B_dari_nonzero", "band_dari_dense", "kebutuhan_penyimpanan",
    "MatriksSingularError", "lu_dense_faktorisasi", "lu_dense_solve",
    "lu_band_faktorisasi", "lu_band_solve", "rekonstruksi_PLU_band",
    "siapkan_band", "steady_state_dense", "steady_state_band",
]
