from pathlib import Path
import numpy as np
import pandas as pd
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "breast-cancer-wisconsin.data"
COLUMNS =  ["id", "clump_thickness", "cell_size", "cell_shape", "marginal_adhesion", "epithelial_size",
            "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "label"]
def load_breast_cancer_original(path=DATA_PATH):
    df = pd.read_csv(path, header=None, names = COLUMNS, na_values="?")
    X = df.drop(columns=["id", "label"]).to_numpy(dtype=float)
    y = (df["label"] == 4).astype(int).to_numpy()
    feature_names = COLUMNS[1:-1]
    return X, y, feature_names
if __name__ == "__main__":
    X, y, names = load_breast_cancer_original()
    print("Kich thuoc X:", X.shape)
    print("Lanh tinh (0):", (y == 0).sum(), "| Ac tinh (1):", (y==1).sum())
    print("So gia tri thieu moi cot:")
    for n, k in zip(names, np.isnan(X).sum(axis = 0)):
        print(f" {n:20s} {k}")
    