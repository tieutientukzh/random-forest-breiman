
"""Sonar - Breiman (2001): Tables 2, 3, 4, Figure 1 and Variable Importance."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rf_breiman import fit_forest, measure, test_error, perm_importance

ROOT = Path(__file__).resolve().parent.parent
PAPER_2 = {"adaboost": 15.6, "selection": 15.9, "single": 18.0, "one_tree": 31.7}
PAPER_3 = {"adaboost": 15.6, "selection": 13.6, "single": 13.8, "one_tree": 31.7}
PAPER_4 = {"Adaboost": 15.1, "RI": -6.6, "RC": 4.2}


def split(X, y, seed):
    return train_test_split(X, y, test_size=.10, random_state=int(seed))


def load_data(path):
    df = pd.read_csv(path, header=None)
    if df.shape != (208, 61):
        raise ValueError(f"Sonar needs (208, 61) raw columns; got {df.shape}")
    X = df.iloc[:, :60].apply(pd.to_numeric, errors="raise").to_numpy(float)
    label = df.iloc[:, 60].astype(str).str.strip().str.upper()
    if set(label) != {"R", "M"} or not np.isfinite(X).all():
        raise ValueError("Expected 60 numeric inputs and labels R/M")
    return X, (label == "M").to_numpy(dtype=int)


def fit_ada(X, y, seed, depth):
    tree = DecisionTreeClassifier(max_depth=depth, criterion="gini", random_state=seed)
    try:
        model = AdaBoostClassifier(estimator=tree, n_estimators=50, random_state=seed)
    except TypeError:
        model = AdaBoostClassifier(base_estimator=tree, n_estimators=50, random_state=seed)
    return model.fit(X, y)


def one_table23(X, y, i, args):
    seed = args.seed + i
    Xtr, Xte, ytr, yte = split(X, y, seed)
    ada = fit_ada(Xtr, ytr, seed, args.ada_depth)
    eada = np.mean(ada.predict(Xte) != yte) * 100
    outputs = []

    for mode, Fs in (("RI", [1, 6]), ("RC", [2, 8])):
        results = {
            f: measure(mode, f, Xtr, ytr, Xte, yte, seed, args.trees)
            for f in Fs
        }
        chosen = min(Fs, key=lambda f: results[f]["oob_error"])
        out = dict(
            seed=seed,
            adaboost=eada,
            n_ada_trees=len(ada.estimators_),
            selection=results[chosen]["test_error"],
            single=results[Fs[0]]["test_error"],
            one_tree=results[chosen]["one_tree"],
            selected_F=chosen,
            strength=results[chosen]["strength"],
            correlation=results[chosen]["correlation"],
            rho_over_s2=results[chosen]["rho_over_s2"],
            F_low_test=results[Fs[0]]["test_error"],
            F_high_test=results[Fs[1]]["test_error"],
            F_low_OOB=results[Fs[0]]["oob_error"],
            F_high_OOB=results[Fs[1]]["oob_error"]
        )
        outputs.append(out)
    return outputs


def table23(X, y, args, res, repeats):
    print(f"\n[BANG 2-3] {repeats} lan, OOB chon F cho tung lan chia", flush=True)
    pairs = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(one_table23)(X, y, i, args) for i in range(repeats)
    )
    df2 = pd.DataFrame([p[0] for p in pairs])
    df3 = pd.DataFrame([p[1] for p in pairs])

    for no, df, paper in [(2, df2, PAPER_2), (3, df3, PAPER_3)]:
        summary = pd.DataFrame({
            "mean_pct": df[list(paper)].mean(),
            "SE_pct": df[list(paper)].std(ddof=1) / np.sqrt(len(df)),
            "paper_pct": pd.Series(paper)
        })
        summary["difference_points"] = summary["mean_pct"] - summary["paper_pct"]

        if no == 3:
            summary.loc["F2_clean", "mean_pct"] = df.F_low_test.mean()
            summary.loc["F8_clean", "mean_pct"] = df.F_high_test.mean()

        filename = f"table{no}_forest_{'ri' if no == 2 else 'rc'}.csv"
        df.to_csv(res / filename, index=False)
        print(f"\nBANG {no}:\n{summary.round(2).to_string()}")
        print("F theo OOB:", df.selected_F.value_counts().to_dict())

    sc = pd.concat([
        df2.assign(mode="RI"),
        df3.assign(mode="RC")
    ], ignore_index=True)
    sc[[
        "seed", "mode", "selected_F", "strength",
        "correlation", "rho_over_s2"
    ]].to_csv(res / "strength_correlation.csv", index=False)
    return df3


def one_figure(X, y, i, args, fs):
    seed = args.seed + 10000 + i
    Xtr, Xte, ytr, yte = split(X, y, seed)
    return [
        dict(F=f, seed=seed, **measure(
            "RI", f, Xtr, ytr, Xte, yte, seed, args.trees
        ))
        for f in fs
    ]


def figure1(X, y, args, res, fig, repeats, fast=False):
    Fs = [1, 3, 6, 8, 10] if fast else range(1, 51)
    print(f"\n[HINH 1] F={list(Fs) if fast else '1..50'}; {repeats} lan", flush=True)

    all_runs = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(one_figure)(X, y, i, args, Fs)
        for i in range(repeats)
    )
    detail = pd.DataFrame(row for group in all_runs for row in group)
    summary = detail.groupby("F")[[
        "test_error", "oob_error", "strength",
        "correlation", "rho_over_s2"
    ]].mean()

    detail.to_csv(res / "figure1_sonar_ri.csv", index=False)

    axes_fig, ax = plt.subplots(2, 1, figsize=(7.5, 8.8), sharex=True)
    ax[0].plot(summary.index, summary.strength, label="Strength")
    ax[0].plot(summary.index, summary.correlation, label="Correlation")
    ax[0].set_ylabel("Value")
    ax[0].legend()
    ax[0].grid(alpha=.2)

    ax[1].plot(summary.index, summary.test_error, label="Test error (%)")
    ax[1].plot(summary.index, summary.oob_error, label="OOB error (%)")
    ax[1].set_xlabel("F (random inputs per node)")
    ax[1].set_ylabel("Error (%)")
    ax[1].legend()
    ax[1].grid(alpha=.2)

    axes_fig.suptitle(f"Breiman 2001 - Sonar Figure 1 ({repeats} repetitions)")
    axes_fig.tight_layout()
    axes_fig.savefig(fig / "hinh1_sonar.png", dpi=160)
    plt.close(axes_fig)

    best = int(summary.test_error.idxmin())
    print(
        f"Hinh 1: RI F={best} co mean clean test error "
        f"thap nhat: {summary.loc[best, 'test_error']:.2f}%"
    )
    return summary


def add_noise(y, seed):
    yn = y.copy()
    rng = np.random.default_rng(100000 + seed)
    indices = rng.choice(len(y), size=round(.05 * len(y)), replace=False)
    yn[indices] = 1 - yn[indices]
    return yn, len(indices)


def one_table4(X, y, i, args, f_ri, f_rc):
    seed = args.seed + 20000 + i
    Xtr, Xte, ytr, yte = split(X, y, seed)
    yn, n_flip = add_noise(ytr, seed)
    rows = []

    for mode, F in [("RI", f_ri), ("RC", f_rc)]:
        clean = fit_forest(mode, Xtr, ytr, F, seed, args.trees)
        noisy = fit_forest(mode, Xtr, yn, F, seed, args.trees)
        rows.append(dict(
            seed=seed, mode=mode, F=F, n_flip=n_flip,
            clean=test_error(mode, clean, Xte, yte),
            noisy=test_error(mode, noisy, Xte, yte)
        ))

    a0 = fit_ada(Xtr, ytr, seed, args.ada_depth)
    a1 = fit_ada(Xtr, yn, seed, args.ada_depth)
    rows.append(dict(
        seed=seed, mode="Adaboost", F=np.nan, n_flip=n_flip,
        clean=np.mean(a0.predict(Xte) != yte) * 100,
        noisy=np.mean(a1.predict(Xte) != yte) * 100,
        ada_clean_trees=len(a0.estimators_),
        ada_noisy_trees=len(a1.estimators_)
    ))
    return rows


def table4(X, y, args, res, repeats, f_ri, f_rc):
    print(f"\n[BANG 4] 5% nhieu, {repeats} lan, RI F={f_ri}, RC F={f_rc}", flush=True)
    all_rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(one_table4)(X, y, i, args, f_ri, f_rc)
        for i in range(repeats)
    )
    df = pd.DataFrame(row for group in all_rows for row in group)
    summary = []

    for mode, g in df.groupby("mode"):
        clean = float(g.clean.mean())
        noisy = float(g.noisy.mean())
        change = 100 * (noisy - clean) / clean
        diff = g.noisy - g.clean
        summary.append(dict(
            mode=mode,
            clean_mean=clean,
            noisy_mean=noisy,
            change_relative_pct=change,
            paper_pct=PAPER_4[mode],
            difference_vs_paper=change - PAPER_4[mode],
            mean_diff_points=float(diff.mean()),
            SE_diff_points=float(diff.std(ddof=1) / np.sqrt(len(diff))),
            improved=int((diff < 0).sum()),
            worsened=int((diff > 0).sum()),
            unchanged=int((diff == 0).sum())
        ))

    out = pd.DataFrame(summary).set_index("mode").reindex(["Adaboost", "RI", "RC"])
    df.to_csv(res / "table4_noise.csv", index=False)
    print("\nBANG 4:\n", out.round(2).to_string())

    ada = df[df["mode"] == "Adaboost"]
    print(
        "AdaBoost mean realized trees:",
        ada[["ada_clean_trees", "ada_noisy_trees"]].mean().round(2).to_dict()
    )


def importance(X, y, args, res, repeats, trees):
    print(f"\n[IMPORTANCE] {repeats} lan x {trees} cay", flush=True)
    runs = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(perm_importance)(
            X, y, args.seed + 30000 + i, trees, 1
        )
        for i in range(repeats)
    )
    detail = pd.DataFrame([
        dict(run=i, **row)
        for i, group in enumerate(runs)
        for row in group
    ])
    summary = detail.groupby("feature")["increase_relative_pct"].agg(["mean", "std"])
    detail.to_csv(res / "variable_importance.csv", index=False)
    print("Top 10 features:\n", summary.sort_values("mean", ascending=False).head(10).round(2))


def choose_features(res, fig_data, rc_data):
    if fig_data is None:
        path = res / "figure1_sonar_ri.csv"
        if not path.exists():
            raise ValueError("Can chay --mode figure1 truoc --mode table4")
        fig_data = pd.read_csv(path)

    if rc_data is None:
        path = res / "table3_forest_rc.csv"
        if not path.exists():
            raise ValueError("Can chay --mode tables23 truoc --mode table4")
        rc_data = pd.read_csv(path)

    f_ri = int(fig_data.groupby("F")["test_error"].mean().idxmin())
    f_rc = 2 if rc_data["F_low_test"].mean() <= rc_data["F_high_test"].mean() else 8
    print(f"Da chon F tu du lieu SACH: RI={f_ri}, RC={f_rc}")
    return f_ri, f_rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["all", "tables23", "figure1", "table4", "importance"],
        default="all"
    )
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trees", type=int, default=100)
    p.add_argument("--ada-depth", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.jobs == 0 or args.trees < 1:
        p.error("--jobs must be nonzero and --trees positive")

    res = ROOT / ("results_quick" if args.quick else "results")
    fig = ROOT / ("figures_quick" if args.quick else "figures")
    res.mkdir(exist_ok=True, parents=True)
    fig.mkdir(exist_ok=True, parents=True)

    data_folder = ROOT / "data"
    data_folder.mkdir(exist_ok=True, parents=True)
    data = data_folder / "sonar.all-data"

    if not data.is_file():
        raise SystemExit(
            "Hay chep sonar.all-data vao Demo/data/sonar.all-data "
            "roi chay lai."
        )

    X, y = load_data(data)
    print(f"Sonar: {X.shape}, R={(y == 0).sum()}, M={(y == 1).sum()}")

    repeats23 = 4 if args.quick else 100
    repeats_fig = 3 if args.quick else 30
    repeats4 = 5 if args.quick else 50
    repeats_importance = 1 if args.quick else 10

    rc_df, fig_df = None, None

    if args.mode in ("all", "tables23"):
        rc_df = table23(X, y, args, res, repeats23)

    if args.mode in ("all", "figure1"):
        fig_df = figure1(X, y, args, res, fig, repeats_fig, args.quick)

    if args.mode in ("all", "table4"):
        ri_F, rc_F = choose_features(res, fig_df, rc_df)
        table4(X, y, args, res, repeats4, ri_F, rc_F)

    if args.mode in ("all", "importance"):
        importance(X, y, args, res, repeats_importance, 100 if args.quick else 1000)

    print(f"\nXONG! Ket qua: {res}; Hinh: {fig}")


if __name__ == "__main__":
    main()
