import  numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_breast_cancer_original
def split_and_impute(X, y, seed, test_size = 0.1):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size = test_size, random_state = seed)
    median = np.nanmedian(X_tr, axis = 0)
    X_tr = np.where(np.isnan(X_tr), median, X_tr)
    X_te = np.where(np.isnan(X_te), median, X_te)
    return X_tr, X_te, y_tr, y_te
if __name__ == "__main__":
    X, y, names = load_breast_cancer_original()

    X_tr, X_te, y_tr, y_te = split_and_impute(X, y, seed = 0)
    print("Train:", X_tr.shape, "| Test:", X_te.shape)
    print("Con NaN trong train: ", np.isnan(X_tr).sum(), " | trong test: ", np.isnan(X_te).sum())
    print("trung vi bare_nuclei tren train: ", np.nanmedian(X[:, 5]))
    counts = [split_and_impute(X, y, seed=s)[3].sum() for s in range(100)]
    print("so ca ac tinh trong test (100 lan chia): min =", min(counts), "| max= ", max(counts), "| trung binh= ", np.mean(counts))
    