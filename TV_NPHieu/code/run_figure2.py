"""Tái hiện Hình 2 của paper (Mục 6): Forest-RC (L=3) trên Breast Cancer, cho số đặc trưng F
thay đổi từ 1 đến 25; với mỗi F đo strength, correlation, test error và OOB error.
Paper lặp 80 lần với mọi F từ 1 đến 25; ở đây rút gọn lưới F và số lần lặp để chạy nhanh.
Chạy song song trên nhiều nhân CPU (joblib)."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from data_loader import load_breast_cancer_original
from protocol import split_and_impute
from forest_utils import majority_vote, oob_error, strength_correlation
from forest_rc import fit_forest_rc, tree_votes_rc

N_REPEATS = 30          # paper: 80. Tăng lên nếu còn thời gian
F_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
N_TREES = 100
BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR, FIG_DIR = BASE / "results", BASE / "figures"


def one_repeat(seed, X, y):
    X_tr, X_te, y_tr, y_te = split_and_impute(X, y, seed)
    out = []
    for F in F_GRID:
        trees, mask = fit_forest_rc(X_tr, y_tr, F, 3, N_TREES, seed)
        votes_tr = tree_votes_rc(trees, X_tr)
        s, rho = strength_correlation(votes_tr, mask, y_tr)
        out.append({"seed": seed, "F": F, "strength": s, "correlation": rho,
                    "test_error": np.mean(majority_vote(tree_votes_rc(trees, X_te)) != y_te),
                    "oob_error": oob_error(votes_tr, mask, y_tr)})
    return out


if __name__ == "__main__":
    X, y, names = load_breast_cancer_original()
    start = time.time()
    results = Parallel(n_jobs=-1, verbose=5)(delayed(one_repeat)(s, X, y) for s in range(N_REPEATS))
    df = pd.DataFrame([r for rep in results for r in rep])
    df.to_csv(RESULTS_DIR / "figure2_rc_breast_runs.csv", index=False)
    print(f"Xong sau {time.time() - start:.0f}s")

    m = df.groupby("F").mean(numeric_only=True)
    print("\n  F      s    rho   test%   oob%")
    for F, r in m.iterrows():
        print(f"{F:3d}  {r.strength:.3f}  {r.correlation:.3f}  {100 * r.test_error:5.2f}  {100 * r.oob_error:5.2f}")

    # ---------- vẽ hình: 2 khung, mỗi khung 1 trục y ----------
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True, facecolor=SURF)
    panels = [
        (axes[0], [("strength", "Strength", BLUE), ("correlation", "Correlation", ORANGE)], 1,
         "Strength và Correlation", None),
        (axes[1], [("test_error", "Test error", BLUE), ("oob_error", "OOB error", ORANGE)], 100,
         "Sai số (%)", None),
    ]
    for ax, series, scale, ylabel, _ in panels:
        ax.set_facecolor(SURF)
        for col, label, color in series:
            vals = scale * m[col]
            ax.plot(m.index, vals, color=color, lw=2, marker="o", ms=6,
                    mec=SURF, mew=1.5, label=label, zorder=3)
            ax.annotate(label, (m.index[-1], vals.iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="best")
    axes[1].set_xlabel("Số đặc trưng F (mỗi đặc trưng là tổ hợp của L = 3 biến)")
    axes[0].set_title(f"Forest-RC trên Breast Cancer ({N_REPEATS} lần lặp, 100 cây)",
                      color=INK, loc="left")
    axes[1].set_xlim(0, max(F_GRID) + 4)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "figure2_rc_breast.png", dpi=200, facecolor=SURF)
    print("Da luu hinh:", FIG_DIR / "figure2_rc_breast.png")