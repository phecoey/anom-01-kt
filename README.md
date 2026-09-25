# TK 1 Analisis Numerik, Soal 1 (Kelompok C11)

Analisis Perpindahan Penumpang Antarhalte: distribusi steady state rantai Markov
dengan solver LU dense dan solver banded bergaya algoritma Thomas.

Kelompok C11 bernomor ganjil sehingga memakai **data kode A** (`data/A/T_N.csv`,
N = 16, 32, 64, 128, 256, 512).

Status pengerjaan:

| Bagian | Isi | Status |
|---|---|---|
| i | Validasi data | selesai |
| ii | Formulasi dan penanganan singularitas | selesai |
| iii | Identifikasi struktur matriks (p, q, penyimpanan) | selesai |
| iv | Implementasi 2 solver + uji ukuran kecil | selesai |
| v sampai ix | Eksperimen kinerja, kompleksitas, kondisi dan galat, interpretasi, rekomendasi | dilanjutkan |

## Kebutuhan

- Python 3.9 atau lebih baru
- `numpy`, `matplotlib` (lihat `requirements.txt`)

```bash
pip install -r requirements.txt
```

## Cara menjalankan

Semua perintah dijalankan dari folder ini (folder yang berisi README.md).

```bash
# 1. Uji kebenaran solver pada ukuran kecil (harus muncul "6 uji lulus")
python -m tests.test_solver

# 2. Hasil bagian i sampai iv (tabel, ringkasan JSON, gambar)
python bagian_i_iv.py              # default memakai data/A
python bagian_i_iv.py --data data/A
```

Keluaran `bagian_i_iv.py`:

| Berkas | Isi |
|---|---|
| `hasil/validasi.csv` | Tugas i: dimensi, min elemen, deviasi jumlah baris, irreducible, periode, peluang lintas potongan minimum |
| `hasil/formulasi.csv` | Tugas ii: bukti singularitas A, pivot terakhir, z1, jumlah z, normalisasi |
| `hasil/struktur.csv` | Tugas iii: (p, q) untuk T, A, B, B streaming, kebutuhan penyimpanan |
| `hasil/uji_N16.csv` | Tugas iv: pi dari kedua solver untuk N = 16 |
| `hasil/ringkasan.json` | Semua angka di atas + uji analitik dan uji pivoting |
| `gambar/pola_band_N16.png` | Pola elemen tak nol T dan B (N = 16) |
| `gambar/penyimpanan.png` | Memori dense vs band |

Waktu jalan seluruhnya sekitar 2 sampai 3 detik.

## Struktur kode

```
anum_soal1/
  data_io.py     baca_T_dense(path), baca_T_nonzero(path) (streaming, hanya elemen tak nol)
  validasi.py    validasi_T(T, N): dimensi, nonnegatif, jumlah baris, irreducible (BFS), periode
  formulasi.py   bentuk_A(T), bentuk_B_b(T), normalisasi(z)
  struktur.py    deteksi_bandwidth(M), bandwidth_B_dari_nonzero, band_B_dari_T,
                 band_B_dari_nonzero, band_dari_dense, kebutuhan_penyimpanan(N, p, q)
  solver.py      lu_dense_faktorisasi / lu_dense_solve         (solver a, PB = LU)
                 lu_band_faktorisasi / lu_band_solve           (solver b, gaya Thomas)
                 steady_state_dense(T), steady_state_band(T=... atau path=...)
tests/test_solver.py   uji analitik, uji pivoting, uji data N = 16
bagian_i_iv.py         skrip bagian i sampai iv
```

Aturan tugas: solver **tidak** memakai `np.linalg` maupun `scipy.linalg`.
NumPy hanya dipakai sebagai wadah array dan operasi aritmetika elementer
(`np.argmax` untuk memilih pivot, `np.outer` untuk update rank-1 pada LU dense).

## Catatan lanjutan untuk bagian v sampai ix

Fungsi yang bisa langsung dipakai:

```python
from anum_soal1 import *

T = baca_T_dense("data/A/T_512.csv")

hd = steady_state_dense(T)                     # solver a
hb = steady_state_band(path="data/A/T_512.csv")  # solver b, tanpa array N x N
# atau: hb = steady_state_band(T=T)

hd["pi"], hb["pi"]             # distribusi steady state (sudah dinormalkan)
hd["info"]["flop"]             # flop faktorisasi; ["flop_solve"] flop substitusi
hd["info"]["jumlah_tukar"]     # jumlah pertukaran baris (untuk data A selalu 0)
hb["p"], hb["q"]               # bandwidth hasil deteksi otomatis (1, 2)
hd["LU"], hd["perm"]           # faktor dense, bisa dipakai ulang
```

Beberapa catatan yang mungkin berguna:

1. **Eksperimen kinerja (v).** Supaya adil, baca T dan siapkan matriks di luar
   pengukuran, lalu ukur tahap faktorisasi + solve saja (misalnya median dari
   beberapa pengulangan). `steady_state_band(path=...)` ikut menghitung parsing
   CSV dan deteksi (p, q), jadi jangan dipakai untuk timing solver murni.

   ```python
   import time
   B, b = bentuk_B_b(T)              # untuk solver dense
   W, p, q, b = siapkan_band(T)      # untuk solver banded (tanpa array N x N)

   t0 = time.perf_counter()
   LU, perm, info = lu_dense_faktorisasi(B); z = lu_dense_solve(LU, perm, b)
   t_dense = time.perf_counter() - t0

   t0 = time.perf_counter()
   Wf, piv, info = lu_band_faktorisasi(W, p, q); z = lu_band_solve(Wf, piv, p, q, b)
   t_band = time.perf_counter() - t0
   ```

   Memori bisa dilaporkan secara teoretis (`kebutuhan_penyimpanan(N, p, q)`:
   dense N^2, band (2p+q+1)N elemen) dan/atau diukur dengan `tracemalloc`.
2. **Kompleksitas (vi).** Jumlah flop tiap solver sudah dihitung eksak di `info`,
   jadi bisa langsung dibandingkan dengan rumus teoretis.
3. **Kondisi dan galat (vii).** Invers B bisa dihitung kolom per kolom dengan
   `lu_dense_solve(LU, perm, e_j)` memakai faktor dari `lu_dense_faktorisasi(B)`
   (tetap tanpa library). Pakai norma yang sama untuk semua N.
   Pivot terakhir U kira-kira 0.1/N (N = 16: 6.5e-3, N = 512: 2.2e-4), kecuali
   **N = 256** yang turun ke 1.19e-7 karena koridornya hampir terputus di antara
   halte 128 dan 129 (T[128,129] sekitar 1.19e-7). Jadi condition number N = 256
   kemungkinan jauh lebih besar daripada ukuran lain.
4. **Interpretasi (viii).** Plot `pi` terhadap nomor halte 1..N untuk tiap N.
5. **Pengumpulan.** Jangan lupa Pakta Integritas beserta tanda tangan semua
   anggota di halaman pertama laporan, format nama berkas sesuai petunjuk, dan
   README ini ikut di-zip bersama kode.
