"""Entrenamiento y validación cruzada sin fuga de documento.

La partición es `GroupKFold` agrupando por `source`: ninguna fuente aparece a
la vez en train y test. TF-IDF y SVD, si se usan, se ajustan solo con train.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, GroupKFold

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


def grid_search(
    estimator: BaseEstimator,
    param_grid: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    scoring: str = "f1",
) -> GridSearchCV:
    """Búsqueda en rejilla con GroupKFold por `groups`, maximizando `scoring`.

    Devuelve el `GridSearchCV` ya ajustado (con `best_estimator_` reentrenado
    sobre todo el train). Por defecto optimiza F1.
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    gs = GridSearchCV(estimator, param_grid, scoring=scoring, cv=gkf, n_jobs=-1)
    gs.fit(X, np.asarray(y), groups=groups)
    return gs


def top_configs(gs: GridSearchCV, k: int = 5) -> pd.DataFrame:
    """Las `k` mejores combinaciones de la rejilla, ordenadas por score CV."""
    res = pd.DataFrame(gs.cv_results_)
    cols = [c for c in res.columns if c.startswith("param_")]
    cols += ["mean_test_score", "std_test_score"]
    return res[cols].sort_values("mean_test_score", ascending=False).head(k).reset_index(drop=True)


def save_model(model: BaseEstimator, name: str) -> None:
    """Serializa el modelo entrenado en `outputs/models/<name>.joblib`."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")
