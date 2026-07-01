"""Inferencia rápida sobre el modelo ya entrenado.

Es la cara pública de la librería: cargar el modelo persistido y puntuar un
par (respuesta, fuente).
"""

from functools import lru_cache

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from ragcheck.config import DEFAULT_MODEL, MODELS_DIR
from ragcheck.features import extract_features


@lru_cache(maxsize=4)
def load_model(name: str = DEFAULT_MODEL) -> BaseEstimator:
    """Carga el modelo entrenado desde `outputs/models/<name>.joblib`."""
    return joblib.load(MODELS_DIR / f"{name}.joblib")


def score(response: str, source: str) -> float:
    """Probabilidad de que `response` contenga una alucinación respecto a `source`.

    Extrae las features del par y aplica el modelo por defecto. Devuelve un
    float en [0, 1].
    """
    X = extract_features(pd.DataFrame({"response": [response], "source": [source]}))
    return float(load_model().predict_proba(X)[0, 1])
