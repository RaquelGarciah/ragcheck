"""Entrenamiento y validación cruzada sin fuga de documento.

La partición es `GroupKFold` agrupando por `source`: ninguna fuente aparece a
la vez en train y test. TF-IDF y SVD, si se usan, se ajustan solo con train.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GroupKFold

from ragcheck.config import MODELS_DIR, N_SPLITS


def cross_validate(
    model: BaseEstimator, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray
) -> dict:
    """Validación cruzada GroupKFold por `groups`.

    Devuelve las probabilidades fuera de muestra (`y_prob`, alineadas con X) y
    el índice de fold de cada fila (`fold`).
    """
    y = np.asarray(y)
    y_prob = np.zeros(len(y), dtype=float)
    fold = np.empty(len(y), dtype=int)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        y_prob[te] = m.predict_proba(X.iloc[te])[:, 1]
        fold[te] = k
    return {"y_prob": y_prob, "fold": fold}


def fit_final(model: BaseEstimator, X: pd.DataFrame, y: np.ndarray) -> BaseEstimator:
    """Entrena el modelo con todos los datos y lo devuelve listo para persistir."""
    m = clone(model)
    m.fit(X, np.asarray(y))
    return m


def save_model(model: BaseEstimator, name: str) -> None:
    """Serializa el modelo entrenado en `outputs/models/<name>.joblib`."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")
