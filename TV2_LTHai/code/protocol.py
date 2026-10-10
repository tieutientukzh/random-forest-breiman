"""Quy trình chia train/test chung; median chỉ được học trên train."""
import numpy as np
from sklearn.model_selection import train_test_split


def split_and_impute(X, y, seed, test_size=0.1, zero_missing_indices=()):
    X = np.asarray(X, dtype=float).copy()
    y = np.asarray(y, dtype=int)
    for col in zero_missing_indices:
        X[X[:, col] == 0, col] = np.nan
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    med = np.nanmedian(X_tr, axis=0)
    if not np.isfinite(med).all():
        raise ValueError('Có cột không có median hợp lệ trong train')
    X_tr = np.where(np.isnan(X_tr), med, X_tr)
    X_te = np.where(np.isnan(X_te), med, X_te)
    return X_tr, X_te, y_tr, y_te
