"""
main.py - Chay TRON BO thuc nghiem Sonar (TV3) va luu ket qua tung lan lap.

Dung chung rf_breiman.py (cung thu muc). Gom:
  - Bang 2: Adaboost | Forest-RI (Selection, Single Input F=1, One Tree)
  - Bang 3: Adaboost | Forest-RC (Selection, Two Features F=2, One Tree)
  - Bang 4: % tang loi khi them 5% nhieu nhan: Adaboost, Forest-RI, Forest-RC
  - Strength / correlation / c_s2 (nam trong cac cot cua Bang 2, 3)
  - Hinh 1: Forest-RI, F tu 1 den 50
  - Variable Importance (1000 cay, lap 10 seed)

Cach chay:
  python -u main.py                 # chay day du (n_rep=100)  -> lau
  python -u main.py 5               # chay thu nhanh (n_rep=5)
  python -u main.py 100 duong_dan\\sonar.all-data
"""
import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from rf_breiman import (
    BreimanForest, add_label_noise, load_sonar, one_repeat,
    plot_figure1, sweep_F, variable_importance, oob_report
)

# So lieu cua bai bao (Sonar) de doi chieu
PAPER = {
    "table2": dict(adaboost=15.6, selection=15.9, single=18.0, one_tree=31.7),
    "table3": dict(adaboost=15.6, selection=13.6, single=13.8, one_tree=31.7),
    "table4": dict(adaboost=15.1, RI=-6.6, RC=4.2),
}


# ------------------------------------------------------------------ Adaboost
class AdaBoostResample:
    """AdaBoost nhi phan dung trong so mau (sample_weight), cay CART max_depth=5, 50 cay.
    Cay vo dung (loi co trong so >= 0.5): bo cay, dat lai trong so deu roi thu tiep.
    Cay khop hoan hao (loi = 0): giu voi alpha = 1, dat lai trong so deu (tranh alpha ~ 11.5 at phieu)."""
    def __init__(self, n_estimators=50, random_state=42, max_depth=5):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)
        self.classes_ = classes
        if len(classes) != 2:
            raise ValueError("AdaBoostResample chi ho tro bai toan 2 lop.")
        y_signed = np.where(y == classes[1], 1, -1)
        n = len(y)
        w = np.ones(n) / n
        self.models, self.alphas = [], []
        tries = 0
        while len(self.models) < self.n_estimators and tries < 3 * self.n_estimators:
            tries += 1
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=self.random_state + tries)
            tree.fit(X, y_signed, sample_weight=w)
            pred = tree.predict(X)
            err = np.sum(w[pred != y_signed])
            if err >= 0.5:
                w = np.ones(n) / n
                continue
            if err <= 1e-10:
                self.models.append(tree)
                self.alphas.append(1.0)
                w = np.ones(n) / n
                continue
            alpha = 0.5 * np.log((1 - err) / err)
            self.models.append(tree)
            self.alphas.append(alpha)
            w *= np.exp(-alpha * y_signed * pred)
            w /= w.sum()
        return self

    def predict(self, X):
        X = np.asarray(X)
        scores = np.zeros(len(X))
        for tree, alpha in zip(self.models, self.alphas):
            scores += alpha * tree.predict(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    def score_error(self, X, y):
        return np.mean(self.predict(X) != np.asarray(y)) * 100




# ------------------------------------------------------------------ Bang 2, 3


def table23_job(X, y, i, F2):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.1, random_state=42 + i
    )

    ada = AdaBoostResample(
        n_estimators=50,
        random_state=42 + i
    ).fit(Xtr, ytr)

    ada_err = ada.score_error(Xte, yte)

    ri = one_repeat(X, y, i, "RI", [1, F2])
    rc = one_repeat(X, y, i, "RC", [2, 8])

    return (
        dict(
            seed=i,
            adaboost=ada_err,
            n_ada_trees=len(ada.models),
            **ri
        ),
        dict(
            seed=i,
            adaboost=ada_err,
            n_ada_trees=len(ada.models),
            **rc
        )
    )



# ------------------------------------------------------------------ Bang 4

def noise_job(X, y, i, rate=0.05):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.1, random_state=42 + i
    )

    yn = add_label_noise(
        ytr, rate, np.random.default_rng(1000 + i)
    )

    rows = []

    ada_clean = AdaBoostResample(n_estimators=50, random_state=42 + i).fit(Xtr, ytr)
    ada_noisy = AdaBoostResample(n_estimators=50, random_state=42 + i).fit(Xtr, yn)

    for name, clean_model, noisy_model in [
        ("Adaboost", ada_clean, ada_noisy)
    ]:
        rows.append({
            "seed": i,
            "mode": name,
            "clean": np.mean(clean_model.predict(Xte) != yte) * 100,
            "noisy": np.mean(noisy_model.predict(Xte) != yte) * 100
        })

    
    for mode, F in [("RI", 6), ("RC", 2)]:
        clean_model = BreimanForest(
            mode, n_trees=100, F=F, seed=42 + i
        ).fit(Xtr, ytr)

        noisy_model = BreimanForest(
            mode, n_trees=100, F=F, seed=42 + i
        ).fit(Xtr, yn)

        rows.append({
            "seed": i,
            "mode": mode,
            "F": F,
            "clean": np.mean(clean_model.predict(Xte) != yte) * 100,
            "noisy": np.mean(noisy_model.predict(Xte) != yte) * 100
        })

    return rows





def summarize_noise(d):
    out = {}

    for mode, g in d.groupby("mode"):
        diff = g["noisy"] - g["clean"]

        relative_from_means = (
            100 * diff.mean() / g["clean"].mean()
        )

        valid = g["clean"] > 0
        relative_per_run = (
            100 * (
                (g.loc[valid, "noisy"] - g.loc[valid, "clean"])
                / g.loc[valid, "clean"]
            )
        ).mean()

        out[mode] = {
            "clean_mean": g["clean"].mean(),
            "noisy_mean": g["noisy"].mean(),
            "tang_tuong_doi": relative_from_means,
            "tang_tb_tung_lan": relative_per_run,
            "chenh_diem": diff.mean(),
            "SE": diff.std() / np.sqrt(len(diff))
        }

    return pd.DataFrame(out).T.round(2)




# ------------------------------------------------------------------ main
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        BASE_DIR, "data", "sonar.all-data"
    )

    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    quick = n_rep < 20

    X, y = load_sonar(path)
    F2 = int(np.log2(X.shape[1]) + 1)
    print(f"Sonar {X.shape}, n_rep = {n_rep}", flush=True)

    # ---- Bang 2, 3
    pairs = Parallel(n_jobs=-1)(
        delayed(table23_job)(X, y, i, F2) for i in range(n_rep)
    )

    t2 = pd.DataFrame([p[0] for p in pairs])
    t3 = pd.DataFrame([p[1] for p in pairs])

    t2.to_csv(os.path.join(RESULTS_DIR, "table2_forest_ri_runs.csv"), index=False)
    t3.to_csv(os.path.join(RESULTS_DIR, "table3_forest_rc_runs.csv"), index=False)

    t23 = pd.concat([
        t2[["seed", "strength", "correlation", "c_s2"]].assign(mode="RI"),
        t3[["seed", "strength", "correlation", "c_s2"]].assign(mode="RC")
    ], ignore_index=True)

    t23.to_csv(os.path.join(RESULTS_DIR, "strength_correlation_runs.csv"), index=False)

    cols = ["adaboost", "selection", "single", "one_tree"]

    for name, t, ref in (
        ("Bang 2 (Forest-RI)", t2, PAPER["table2"]),
        ("Bang 3 (Forest-RC)", t3, PAPER["table3"])
    ):
        comp = pd.DataFrame({
            "cua ban": t[cols].mean(),
            "SE": t[cols].std() / np.sqrt(len(t)),
            "bai bao": pd.Series(ref)
        }).round(2)

        print(f"\n{name}\n{comp}", flush=True)
        print(
            "strength/correlation/c_s2:",
            t[["strength", "correlation", "c_s2"]].mean().round(3).to_dict()
        )

    print("So cay Adaboost trung binh:", round(t2["n_ada_trees"].mean(), 1), flush=True)

    # ---- Bang 4
    n4 = 50 if n_rep >= 100 else n_rep

    d4 = pd.DataFrame([
        r
        for rows in Parallel(n_jobs=-1)(
            delayed(noise_job)(X, y, i) for i in range(n4)
        )
        for r in rows
    ])

    d4.to_csv(os.path.join(RESULTS_DIR, "table4_noise_runs.csv"), index=False)

    s4 = summarize_noise(d4)
    s4["bai bao (%)"] = pd.Series({
        "Adaboost": PAPER["table4"]["adaboost"],
        "RI": PAPER["table4"]["RI"],
        "RC": PAPER["table4"]["RC"]
    })

    print("\nBang 4 (tang loi khi them 5% nhieu nhan)\n", s4, flush=True)

    # ---- Hinh 1
    Fs = list(range(1, 51))

    fig = sweep_F(X, y, Fs, n_rep=80)
    fig.to_csv(os.path.join(RESULTS_DIR, "figure1_sonar_ri.csv"))
    plot_figure1(fig, os.path.join(FIGURES_DIR, "hinh1_sonar.png"))

    print("\nHinh 1: da luu figures/hinh1_sonar.png", flush=True)

    # ---- Variable Importance
    n_vi, trees_vi = (2, 100) if quick else (10, 1000)

    vis = [
        variable_importance(
            BreimanForest("RI", trees_vi, 1, seed=42 + s).fit(X, y),
            seed=s
        )
        for s in range(n_vi)
    ]

    vi = pd.DataFrame(vis, columns=range(1, X.shape[1] + 1))
    vi.index.name = "run"
    vi.to_csv(os.path.join(RESULTS_DIR, "variable_importance_runs.csv"))

    print(
        "\nTop 10 bien quan trong (trung binh qua",
        n_vi,
        "lan chay, % tang loi):"
    )
    print(vi.mean().sort_values(ascending=False).head(10).round(1), flush=True)

    print("\nXONG. Ket qua nam trong thu muc results/ va figures/.")
