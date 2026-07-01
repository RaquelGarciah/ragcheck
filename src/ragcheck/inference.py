"""Inferencia rápida sobre el modelo ya entrenado.

Es la cara pública de la librería: cargar el modelo persistido y puntuar un
par (respuesta, fuente).
"""

from sklearn.base import BaseEstimator


def load_model(name: str = "default") -> BaseEstimator:
    """Carga el modelo entrenado desde `outputs/models/<name>.joblib`."""
    raise NotImplementedError  # Fase 0


def score(response: str, source: str) -> float:
    """Probabilidad de que `response` contenga una alucinación respecto a `source`.

    Extrae las features del par y aplica el modelo por defecto. Devuelve un
    float en [0, 1].
    """
    raise NotImplementedError  # Fase 0
