"""Tái hiện Bảng 2 của paper (Forest-RI) trên Breast Cancer.
Quy trình (Mục 4): giữ 10% làm test, chạy rừng 100 cây với F=1 và F=int(log2 M + 1),
chọn F có OOB error thấp hơn, lặp 100 lần, lấy trung bình."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import load_breast_cancer_original
from protocol import split_and_impute
from forest_utils import (fit_forest_ri, tree_votes, majority_vote,
                          oob_mask, oob_error, one_tree_error)

N_REPEATS = 100
N_TREES = 100
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

X, y, names = load_breast_cancer_original()
M = X.shape[1]
F_VALUES = [1, int(np.log2(M) + 1)]   # Breast: [1, 4]
print("M =", M, "| F thu nghiem:", F_VALUES)

rows = []
start = time.time()
for seed in range(N_REPEATS):
    X_tr, X_te, y_tr, y_te = split_and_impute(X, y, seed)
    row = {"seed": seed}
    for F in F_VALUES:
        rf = fit_forest_ri(X_tr, y_tr, F, N_TREES, seed)
        votes_tr = tree_votes(rf, X_tr)
        mask = oob_mask(rf, len(y_tr))
        row[f"test_F{F}"] = np.mean(majority_vote(tree_votes(rf, X_te)) != y_te)
        row[f"oob_F{F}"] = oob_error(votes_tr, mask, y_tr)
        row[f"onetree_F{F}"] = one_tree_error(votes_tr, mask, y_tr)

    # Cột Selection: lấy test error của F có OOB error thấp hơn (hòa thì lấy F nhỏ)
    best_F = min(F_VALUES, key=lambda F: row[f"oob_F{F}"])
    row["best_F"] = best_F
    row["selection"] = row[f"test_F{best_F}"]
    row["single_input"] = row["test_F1"]
    row["one_tree"] = row[f"onetree_F{best_F}"]
    rows.append(row)
    if (seed + 1) % 10 == 0:
        print(f"  xong {seed + 1}/{N_REPEATS} lan ({time.time() - start:.0f}s)")

df = pd.DataFrame(rows)
RESULTS_DIR.mkdir(exist_ok=True)
df.to_csv(RESULTS_DIR / "table2_forest_ri_runs.csv", index=False)

paper = {"selection": 2.9, "single_input": 2.7, "one_tree": 6.3}
print("\nBANG 2 - Forest-RI tren Breast Cancer (% loi, trung binh 100 lan)")
print(f"{'Cot':14s} {'Nhom':>8s} {'Do lech chuan':>14s} {'Paper':>7s}")
for col in ["selection", "single_input", "one_tree"]:
    print(f"{col:14s} {100 * df[col].mean():8.2f} {100 * df[col].std():14.2f} {paper[col]:7.1f}")
print("\nSo lan OOB chon F=1:", (df.best_F == 1).sum(), "/", N_REPEATS)
print("Test error trung binh: F=1 ->", round(100 * df[f"test_F{F_VALUES[0]}"].mean(), 2),
      "| F=4 ->", round(100 * df[f"test_F{F_VALUES[1]}"].mean(), 2))