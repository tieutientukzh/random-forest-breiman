"""Tái hiện Bảng 3 của paper (Forest-RC) trên Breast Cancer.
Quy trình (Mục 5): L=3, F=2 và F=8, chọn F có OOB error thấp hơn, 100 cây, lặp 100 lần.
Không chuẩn hóa dữ liệu vì cả 9 biến của Breast Cancer cùng thang điểm 1-10."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import load_breast_cancer_original
from protocol import split_and_impute
from forest_utils import majority_vote, oob_error, one_tree_error
from forest_rc import fit_forest_rc, tree_votes_rc

N_REPEATS = 100
N_TREES = 100
L = 3
F_VALUES = [2, 8]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

X, y, names = load_breast_cancer_original()

rows = []
start = time.time()
for seed in range(N_REPEATS):
    X_tr, X_te, y_tr, y_te = split_and_impute(X, y, seed)
    row = {"seed": seed}
    for F in F_VALUES:
        trees, mask = fit_forest_rc(X_tr, y_tr, F, L, N_TREES, seed)
        votes_tr = tree_votes_rc(trees, X_tr)
        row[f"test_F{F}"] = np.mean(majority_vote(tree_votes_rc(trees, X_te)) != y_te)
        row[f"oob_F{F}"] = oob_error(votes_tr, mask, y_tr)
        row[f"onetree_F{F}"] = one_tree_error(votes_tr, mask, y_tr)

    best_F = min(F_VALUES, key=lambda F: row[f"oob_F{F}"])
    row["best_F"] = best_F
    row["selection"] = row[f"test_F{best_F}"]
    row["two_features"] = row["test_F2"]
    row["one_tree"] = row[f"onetree_F{best_F}"]
    rows.append(row)
    if (seed + 1) % 10 == 0:
        print(f"  xong {seed + 1}/{N_REPEATS} lan ({time.time() - start:.0f}s)")

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "table3_forest_rc_runs.csv", index=False)

paper = {"selection": 3.1, "two_features": 2.9, "one_tree": 5.8}
print("\nBANG 3 - Forest-RC tren Breast Cancer (% loi, trung binh 100 lan)")
print(f"{'Cot':14s} {'Nhom':>8s} {'SE':>6s} {'Paper':>7s}")
for col in ["selection", "two_features", "one_tree"]:
    print(f"{col:14s} {100 * df[col].mean():8.2f} {100 * df[col].std() / np.sqrt(N_REPEATS):6.2f} {paper[col]:7.1f}")
print("\nSo lan OOB chon F=2:", (df.best_F == 2).sum(), "/", N_REPEATS)
print("Test error trung binh: F=2 ->", round(100 * df["test_F2"].mean(), 2),
      "| F=8 ->", round(100 * df["test_F8"].mean(), 2))

# So sánh cặp với Forest-RI (cùng seed = cùng tập train/test)
ri = pd.read_csv(RESULTS_DIR / "table2_forest_ri_runs.csv")
diff = df["selection"] - ri["selection"]
print(f"Forest-RC - Forest-RI (Selection): {100 * diff.mean():.2f} (SE {100 * diff.std() / np.sqrt(N_REPEATS):.2f})")