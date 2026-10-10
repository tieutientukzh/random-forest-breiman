"""Chạy Adaboost (50 cây) theo cùng quy trình với Bảng 2, để so sánh với Forest-RI.
Paper không nói rõ độ sâu cây trong Adaboost, nên thử vài độ sâu và báo cáo tất cả."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from data_loader import load_breast_cancer_original
from protocol import split_and_impute

N_REPEATS = 100
N_TREES = 50                  # paper: Adaboost kết hợp 50 cây
DEPTHS = [1, 3, 5, 8, None]   # None = cây mọc tối đa          
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

X, y, names = load_breast_cancer_original()

rows = []
start = time.time()
for seed in range(N_REPEATS):
    X_tr, X_te, y_tr, y_te = split_and_impute(X, y, seed)   # cùng cách chia với Bảng 2
    row = {"seed": seed}
    for d in DEPTHS:
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=d, random_state=seed),
            n_estimators=N_TREES,
            random_state=seed,
        )
        ada.fit(X_tr, y_tr)
        row[f"ada_depth{d}"] = np.mean(ada.predict(X_te) != y_te)
        row[f"n_trees_depth{d}"] = len(ada.estimators_)   # Adaboost có thể dừng sớm
    rows.append(row)
    if (seed + 1) % 20 == 0:
        print(f"  xong {seed + 1}/{N_REPEATS} lan ({time.time() - start:.0f}s)")

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "adaboost_runs.csv", index=False)

# So sánh cặp với Forest-RI: cùng seed nghĩa là cùng tập train/test
ri = pd.read_csv(RESULTS_DIR / "table2_forest_ri_runs.csv")
print("\nADABOOST tren Breast Cancer (% loi, trung binh 100 lan) - paper: 3.2")
print(f"{'Do sau cay':12s} {'Loi':>6s} {'SE':>6s} {'So cay TB':>10s} {'Forest-RI - Ada':>16s} {'SE hieu':>8s}")
for d in DEPTHS:
    err = df[f"ada_depth{d}"]
    diff = ri["selection"] - err                 # âm nghĩa là Forest-RI tốt hơn
    print(f"{str(d):<12s} {100 * err.mean():6.2f} {100 * err.std() / np.sqrt(N_REPEATS):6.2f} "
          f"{df[f'n_trees_depth{d}'].mean():10.1f} {100 * diff.mean():16.2f} "
          f"{100 * diff.std() / np.sqrt(N_REPEATS):8.2f}")