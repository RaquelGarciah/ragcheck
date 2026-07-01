"""Entrenamiento y validación cruzada sin fuga de documento.

La partición es `GroupKFold` agrupando por `source`: ninguna fuente aparece a
la vez en train y test. TF-IDF y SVD, si se usan, se ajustan solo con train.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def cross_validate(
    model: BaseEstimator, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray
) -> dict:
    """Validación cruzada GroupKFold por `groups`.

    Devuelve las probabilidades fuera de muestra y las métricas por fold.
    """
    raise NotImplementedError  # Fase 0


def fit_final(
    model: BaseEstimator, X: pd.DataFrame, y: np.ndarray
) -> BaseEstimator:
    """Entrena el modelo con todos los datos y lo devuelve listo para persistir."""
    raise NotImplementedError  # Fase 0


def save_model(model: BaseEstimator, name: str) -> None:
    """Serializa el modelo entrenado en `outputs/models/<name>.joblib`."""
    raise NotImplementedError  # Fase 0
