"""Đo strength (s), correlation (rho), c/s^2 và cận sai số rho(1-s^2)/s^2 (Định lý 2.3)
cho 4 cấu hình: Forest-RI (F=1, F=4) và Forest-RC (F=2, F=8). Ước lượng bằng OOB (Phụ lục II)."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import load_breast_cancer_original
from protocol import split_and_impute
from forest_utils import (fit_forest_ri, tree_votes, majority_vote,
                          oob_mask, oob_error, strength_correlation)
from forest_rc import fit_forest_rc, tree_votes_rc

N_REPEATS = 100
N_TREES = 100
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CONFIGS = [("RI", 1), ("RI", 4), ("RC", 2), ("RC", 8)]

X, y, names = load_breast_cancer_original()

rows = []
start = time.time()
for seed in range(N_REPEATS):
    X_tr, X_te, y_tr, y_te = split_and_impute(X, y, seed)
    for method, F in CONFIGS:
        if method == "RI":
            rf = fit_forest_ri(X_tr, y_tr, F, N_TREES, seed)
            votes_tr, votes_te = tree_votes(rf, X_tr), tree_votes(rf, X_te)
            mask = oob_mask(rf, len(y_tr))
        else:
            trees, mask = fit_forest_rc(X_tr, y_tr, F, 3, N_TREES, seed)
            votes_tr, votes_te = tree_votes_rc(trees, X_tr), tree_votes_rc(trees, X_te)
        s, rho = strength_correlation(votes_tr, mask, y_tr)
        rows.append({"seed": seed, "method": method, "F": F, "strength": s, "correlation": rho,
                     "test_error": np.mean(majority_vote(votes_te) != y_te),
                     "oob_error": oob_error(votes_tr, mask, y_tr)})
    if (seed + 1) % 10 == 0:
        print(f"  xong {seed + 1}/{N_REPEATS} lan ({time.time() - start:.0f}s)")

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "strength_correlation_runs.csv", index=False)

summary = df.groupby(["method", "F"], sort=False).mean(numeric_only=True)
summary["c_s2"] = summary["correlation"] / summary["strength"] ** 2
summary["bound"] = summary["correlation"] * (1 - summary["strength"] ** 2) / summary["strength"] ** 2
print("\nSTRENGTH / CORRELATION tren Breast Cancer (trung binh 100 lan)")
print(f"{'Cau hinh':10s} {'s':>7s} {'rho':>7s} {'c/s2':>7s} {'Can PE*':>8s} {'Test %':>7s} {'OOB %':>7s}")
for (method, F), r in summary.iterrows():
    print(f"{method + ' F=' + str(F):10s} {r.strength:7.3f} {r.correlation:7.3f} {r.c_s2:7.3f} "
          f"{100 * r.bound:7.1f}% {100 * r.test_error:7.2f} {100 * r.oob_error:7.2f}")