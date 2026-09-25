"""
Soal 1 TK 1 Anum, bagian i sampai iv.

    i)   Validasi data
    ii)  Formulasi dan penanganan singularitas
    iii) Identifikasi struktur matriks (bandwidth p, q dan penyimpanan)
    iv)  Implementasi solver dan uji pada ukuran kecil

Jalankan dari folder proyek:
    python bagian_i_iv.py            (data default: data/A)
    python bagian_i_iv.py --data data/A

Keluaran:
    hasil/validasi.csv, hasil/formulasi.csv, hasil/struktur.csv,
    hasil/uji_N16.csv, hasil/ringkasan.json
    gambar/pola_band_N16.png, gambar/penyimpanan.png
"""

import argparse
import csv
import json
import os

import numpy as np

from anum_soal1 import (
    UKURAN_N, path_T, baca_T_dense, baca_T_nonzero, validasi_T,
    bentuk_A, bentuk_B_b, deteksi_bandwidth, bandwidth_B_dari_nonzero,
    band_B_dari_T, band_dari_dense, kebutuhan_penyimpanan,
    lu_dense_faktorisasi, lu_dense_solve, lu_band_faktorisasi, lu_band_solve,
    rekonstruksi_PLU_band, steady_state_dense, steady_state_band,
)
from tests.test_solver import rantai_birth_death, rantai_doubly_stochastic, matriks_band_acak

ROOT = os.path.dirname(os.path.abspath(__file__))


def simpan_csv(path, baris):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0].keys()))
        w.writeheader()
        w.writerows(baris)


def maks_abs(x):
    return float(np.max(np.abs(x)))


# ---------------------------------------------------------------------------
def tugas_i(data_dir):
    tabel = []
    for N in UKURAN_N:
        T = baca_T_dense(path_T(data_dir, N))
        v = validasi_T(T, N)
        tabel.append({
            "N": N, "dimensi": f"{v['dimensi'][0]}x{v['dimensi'][1]}",
            "semua_hingga": v["semua_hingga"], "min_elemen": v["min_elemen"],
            "jumlah_negatif": v["jumlah_negatif"],
            "maks_dev_jumlah_baris": v["maks_dev_jumlah_baris"],
            "nnz": v["nnz"], "diagonal_positif": v["diagonal_positif"],
            "irreducible": v["irreducible"], "periode": v["periode"],
            "maju_min": v["maju_min"], "potongan_maju_min": v["potongan_maju_min"],
            "mundur_min": v["mundur_min"], "potongan_mundur_min": v["potongan_mundur_min"],
            "valid": v["valid"],
        })
    return tabel


def tugas_ii(data_dir):
    tabel = []
    contoh = {}
    for N in UKURAN_N:
        T = baca_T_dense(path_T(data_dir, N))
        A = bentuk_A(T)
        # 1) 1^T A = 0^T  -> baris-baris A bergantung linear
        jumlah_kolom = maks_abs(A.sum(axis=0))
        # 2) jika LU dipaksakan pada A, pivot terakhir ~ 0
        _, _, info_A = lu_dense_faktorisasi(A, izinkan_pivot_nol=True)
        # 3) sistem B z = b
        B, b = bentuk_B_b(T)
        LU, perm, info_B = lu_dense_faktorisasi(B)
        z = lu_dense_solve(LU, perm, b)
        s = float(np.sum(z))
        pi = z / s
        tabel.append({
            "N": N,
            "maks_abs_jumlah_kolom_A": jumlah_kolom,
            "pivot_terakhir_A": float(info_A["pivot"][-1]),
            "pivot_min_lain_A": float(info_A["pivot"][:-1].min()),
            "pivot_min_B": float(info_B["pivot"].min()),
            "z1": float(z[0]), "jumlah_z": s,
            "jumlah_pi_minus_1": abs(float(np.sum(pi)) - 1.0),
            "pi_min": float(pi.min()),
        })
        if N == 16:
            contoh = {
                "B_6x6": B[:6, :6].tolist(), "A_baris1": A[0, :5].tolist(),
                "z": z.tolist(), "pi": pi.tolist(), "jumlah_z": s,
            }
    return tabel, contoh


def tugas_iii(data_dir):
    tabel = []
    for N in UKURAN_N:
        path = path_T(data_dir, N)
        T = baca_T_dense(path)
        A = bentuk_A(T)
        B, _ = bentuk_B_b(T)
        pT, qT = deteksi_bandwidth(T)
        pA, qA = deteksi_bandwidth(A)
        pB, qB = deteksi_bandwidth(B)
        # jalur streaming (tanpa membentuk B) harus memberi hasil yang sama
        n, entri = baca_T_nonzero(path)
        pS, qS = bandwidth_B_dari_nonzero(n, entri)
        # pembanding: jika baris 1 diganti 1^T (bukan e_1^T), band rusak
        B_alt = B.copy()
        B_alt[0, :] = 1.0
        pX, qX = deteksi_bandwidth(B_alt)
        s = kebutuhan_penyimpanan(N, pB, qB)
        tabel.append({
            "N": N, "p_T": pT, "q_T": qT, "p_A": pA, "q_A": qA,
            "p_B": pB, "q_B": qB, "p_B_stream": pS, "q_B_stream": qS,
            "p_B_baris1_satu": pX, "q_B_baris1_satu": qX,
            "nnz_B": int(np.count_nonzero(B)),
            **{k: v for k, v in s.items() if k not in ("N", "p", "q")},
        })
    return tabel


def tugas_iv(data_dir):
    hasil = {}
    # --- (1) data A, N = 16 --------------------------------------------------
    T = baca_T_dense(path_T(data_dir, 16))
    B, b = bentuk_B_b(T)
    hd = steady_state_dense(T)
    hb = steady_state_band(path=path_T(data_dir, 16))
    L = np.tril(hd["LU"], -1) + np.eye(16)
    U = np.triu(hd["LU"])
    P, Lb, Ub = rekonstruksi_PLU_band(hb["W"], hb["piv"], hb["p"], hb["q"])
    W0 = band_B_dari_T(T, hb["p"], hb["q"])
    hasil["N16"] = {
        "p": hb["p"], "q": hb["q"],
        "pi_dense": hd["pi"].tolist(), "pi_band": hb["pi"].tolist(),
        "maks_selisih_pi": maks_abs(hd["pi"] - hb["pi"]),
        "residual_Bz_dense": maks_abs(B @ hd["z"] - b),
        "residual_Bz_band": maks_abs(B @ hb["z"] - b),
        "PB_LU_dense": maks_abs(B[hd["perm"]] - L @ U),
        "PB_LU_band": maks_abs(P @ B - Lb @ Ub),
        "tukar_dense": hd["info"]["jumlah_tukar"], "tukar_band": hb["info"]["jumlah_tukar"],
        "flop_faktor_dense": hd["info"]["flop"], "flop_solve_dense": hd["info"]["flop_solve"],
        "flop_faktor_band": hb["info"]["flop"], "flop_solve_band": hb["info"]["flop_solve"],
        "residual_TTpi_dense": maks_abs(T.T @ hd["pi"] - hd["pi"]),
        "residual_TTpi_band": maks_abs(T.T @ hb["pi"] - hb["pi"]),
        "W_awal_6baris": W0[:6].tolist(),
        "elemen_simpan_dense": 16 * 16, "elemen_simpan_band": int(W0.size),
    }
    # --- (2) uji analitik -------------------------------------------------------
    uji = []
    for nama, (Tt, pi_eksak) in {
        "birth-death N=6 (pi via detailed balance)": rantai_birth_death(6),
        "doubly stochastic N=8 (pi = 1/N)": rantai_doubly_stochastic(8),
    }.items():
        e_d = maks_abs(steady_state_dense(Tt)["pi"] - pi_eksak)
        e_b = maks_abs(steady_state_band(T=Tt)["pi"] - pi_eksak)
        uji.append({"kasus": nama, "galat_dense": e_d, "galat_band": e_b})
    hasil["uji_analitik"] = uji
    # --- (3) uji pivoting (baris harus ditukar) ------------------------------
    piv_uji = []
    for p, q, seed in [(1, 2, 7), (2, 1, 11)]:
        N = 10
        M = matriks_band_acak(N, p, q, seed)
        x_benar = np.arange(1.0, N + 1.0)
        rhs = M @ x_benar
        LU, perm, i_d = lu_dense_faktorisasi(M)
        x_d = lu_dense_solve(LU, perm, rhs)
        Wf, piv, i_b = lu_band_faktorisasi(band_dari_dense(M, p, q), p, q)
        x_b = lu_band_solve(Wf, piv, p, q, rhs)
        Pm, Lm, Um = rekonstruksi_PLU_band(Wf, piv, p, q)
        piv_uji.append({
            "p": p, "q": q, "N": N,
            "tukar_dense": i_d["jumlah_tukar"], "tukar_band": i_b["jumlah_tukar"],
            "PB_LU_dense": maks_abs(M[perm] - (np.tril(LU, -1) + np.eye(N)) @ np.triu(LU)),
            "PB_LU_band": maks_abs(Pm @ M - Lm @ Um),
            "galat_x_dense": maks_abs(x_d - x_benar), "galat_x_band": maks_abs(x_b - x_benar),
            "q_U_setelah_pivot": deteksi_bandwidth(Um)[1],
        })
    hasil["uji_pivoting"] = piv_uji
    # --- (4) cek jalan untuk semua N (bukan eksperimen waktu) -----------------
    semua = []
    for N in UKURAN_N:
        Tn = baca_T_dense(path_T(data_dir, N))
        a = steady_state_dense(Tn)
        c = steady_state_band(path=path_T(data_dir, N))
        semua.append({
            "N": N, "maks_selisih_pi": maks_abs(a["pi"] - c["pi"]),
            "tukar_dense": a["info"]["jumlah_tukar"], "tukar_band": c["info"]["jumlah_tukar"],
            "flop_faktor_dense": a["info"]["flop"], "flop_faktor_band": c["info"]["flop"],
            "jumlah_pi_minus_1_band": abs(float(np.sum(c["pi"])) - 1.0),
        })
    hasil["cek_semua_N"] = semua
    return hasil


# ---------------------------------------------------------------------------
def buat_gambar(data_dir, tabel_iii):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    BIRU, ORANYE, TEKS, TEKS2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 9,
        "text.color": TEKS, "axes.labelcolor": TEKS2, "xtick.color": TEKS2, "ytick.color": TEKS2,
        "axes.edgecolor": GRID,
    })
    os.makedirs(os.path.join(ROOT, "gambar"), exist_ok=True)

    # --- Gambar 1: pola tak nol T dan B (N = 16) --------------------------------
    T = baca_T_dense(path_T(data_dir, 16))
    B, _ = bentuk_B_b(T)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    for ax, M, judul in [(axes[0], T, "T"), (axes[1], B, "B")]:
        N = M.shape[0]
        p, q = deteksi_bandwidth(M)
        for i in range(N):
            for j in range(N):
                if M[i, j] != 0.0:
                    ax.add_patch(Rectangle((j + 0.08, i + 0.08), 0.84, 0.84, color=BIRU, lw=0))
        # batas band berbentuk tangga: kiri x = i - p, kanan x = i + q + 1
        bawah = [(0, p + 1)]
        for i in range(p + 1, N):
            bawah += [(i - p, i), (i - p, i + 1)]
        atas = []
        for i in range(N - q - 1):
            atas += [(i + q + 1, i), (i + q + 1, i + 1)]
        atas.append((N, N - q - 1))
        for pts in (bawah, atas):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, ls=(0, (3, 2)), lw=1, color=TEKS2)
        ax.set_xlim(0, N)
        ax.set_ylim(N, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(N) + 0.5, [str(k + 1) if (k + 1) in (1, 4, 8, 12, 16) else "" for k in range(N)])
        ax.set_yticks(np.arange(N) + 0.5, [str(k + 1) if (k + 1) in (1, 4, 8, 12, 16) else "" for k in range(N)])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(f"{judul} (N = 16):  p = {p},  q = {q}", fontsize=10, color=TEKS)
        ax.set_xlabel("kolom j")
        ax.set_ylabel("baris i")
    axes[1].add_patch(Rectangle((0.02, 0.02), 15.96, 0.96, fill=False, ec=ORANYE, lw=1.6))
    axes[1].annotate("baris 1 diganti [1 0 ... 0]", xy=(11.0, 1.0), xytext=(8.6, 3.6),
                     color=TEKS, fontsize=8.5, va="center",
                     arrowprops=dict(arrowstyle="-", color=ORANYE, lw=1, shrinkA=2, shrinkB=0))
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "gambar", "pola_band_N16.png"), dpi=200)
    plt.close(fig)

    # --- Gambar 2: kebutuhan penyimpanan dense vs band ---------------------------
    Ns = [r["N"] for r in tabel_iii]
    dense = [r["dense_byte"] / 1024 for r in tabel_iii]
    band = [r["band_pivot_byte"] / 1024 for r in tabel_iii]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(Ns, dense, color=BIRU, lw=2, marker="o", ms=5, label="Dense  N x N")
    ax.plot(Ns, band, color=ORANYE, lw=2, marker="o", ms=5, label="Band  (2p+q+1) N, termasuk fill-in")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.set_xlabel("Ukuran matriks N")
    ax.set_ylabel("Memori matriks koefisien (KiB)")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate(f"{dense[-1]:.0f} KiB", (Ns[-1], dense[-1]), xytext=(-6, 8),
                textcoords="offset points", ha="right", fontsize=8.5, color=TEKS)
    ax.annotate(f"{band[-1]:.0f} KiB", (Ns[-1], band[-1]), xytext=(-6, 8),
                textcoords="offset points", ha="right", fontsize=8.5, color=TEKS)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "gambar", "penyimpanan.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "A"))
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "hasil"), exist_ok=True)

    t_i = tugas_i(args.data)
    t_ii, contoh_ii = tugas_ii(args.data)
    t_iii = tugas_iii(args.data)
    t_iv = tugas_iv(args.data)

    simpan_csv(os.path.join(ROOT, "hasil", "validasi.csv"), t_i)
    simpan_csv(os.path.join(ROOT, "hasil", "formulasi.csv"), t_ii)
    simpan_csv(os.path.join(ROOT, "hasil", "struktur.csv"), t_iii)
    simpan_csv(os.path.join(ROOT, "hasil", "uji_N16.csv"), [
        {"halte": i + 1, "pi_dense": a, "pi_band": c, "selisih": abs(a - c)}
        for i, (a, c) in enumerate(zip(t_iv["N16"]["pi_dense"], t_iv["N16"]["pi_band"]))
    ])
    with open(os.path.join(ROOT, "hasil", "ringkasan.json"), "w") as f:
        json.dump({"i": t_i, "ii": t_ii, "ii_contoh": contoh_ii, "iii": t_iii, "iv": t_iv},
                  f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    buat_gambar(args.data, t_iii)

    print("== i) Validasi data")
    for r in t_i:
        print(f"  N={r['N']:>3}  {r['dimensi']:>7}  min={r['min_elemen']:.1e}  "
              f"dev_jml_baris={r['maks_dev_jumlah_baris']:.1e}  irreducible={r['irreducible']}  "
              f"periode={r['periode']}  maju_min={r['maju_min']:.2e}  mundur_min={r['mundur_min']:.2e}  "
              f"valid={r['valid']}")
    print("== ii) Singularitas A dan normalisasi")
    for r in t_ii:
        print(f"  N={r['N']:>3}  |1^T A|max={r['maks_abs_jumlah_kolom_A']:.1e}  "
              f"pivot_akhir_A={r['pivot_terakhir_A']:.1e}  pivot_min_B={r['pivot_min_B']:.2e}  "
              f"z1={r['z1']:.1f}  1^T z={r['jumlah_z']:.4f}  |1^T pi - 1|={r['jumlah_pi_minus_1']:.1e}")
    print("== iii) Bandwidth dan penyimpanan")
    for r in t_iii:
        print(f"  N={r['N']:>3}  T(p,q)=({r['p_T']},{r['q_T']})  B(p,q)=({r['p_B']},{r['q_B']})  "
              f"dense={r['dense_elemen']}  band_eksak={r['band_eksak_elemen']}  "
              f"band+fill={r['band_pivot_elemen']}  rasio={r['rasio_dense_per_band_pivot']:.1f}x")
    print("== iv) Uji solver")
    n16 = t_iv["N16"]
    print(f"  N=16: maks|pi_dense - pi_band|={n16['maks_selisih_pi']:.1e}  "
          f"|PB-LU| dense={n16['PB_LU_dense']:.1e} band={n16['PB_LU_band']:.1e}  "
          f"tukar={n16['tukar_dense']}/{n16['tukar_band']}")
    for r in t_iv["uji_analitik"]:
        print(f"  {r['kasus']}: galat dense={r['galat_dense']:.1e} band={r['galat_band']:.1e}")
    for r in t_iv["uji_pivoting"]:
        print(f"  pivot p={r['p']} q={r['q']}: tukar={r['tukar_dense']}/{r['tukar_band']}  "
              f"|PB-LU|={r['PB_LU_dense']:.1e}/{r['PB_LU_band']:.1e}  "
              f"galat x={r['galat_x_dense']:.1e}/{r['galat_x_band']:.1e}  q(U)={r['q_U_setelah_pivot']}")
    for r in t_iv["cek_semua_N"]:
        print(f"  N={r['N']:>3}: maks|pi_dense - pi_band|={r['maks_selisih_pi']:.1e}  "
              f"tukar={r['tukar_dense']}/{r['tukar_band']}")


if __name__ == "__main__":
    main()
