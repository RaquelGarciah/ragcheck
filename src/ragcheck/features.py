"""Features léxicas y estadísticas sobre el par (respuesta, fuente).

Notación: R = palabras únicas de la respuesta, F = palabras únicas de la
fuente. Todas las funciones reciben `(response, source)` y devuelven un float.
"""

import pandas as pd


def containment(response: str, source: str) -> float:
    """Precisión léxica: |R ∩ F| / |R|. Señal dominante del detector."""
    raise NotImplementedError  # Fase 0


def jaccard(response: str, source: str) -> float:
    """Solape simétrico: |R ∩ F| / |R ∪ F|."""
    raise NotImplementedError  # Fase 0


def tfidf_cos(response: str, source: str) -> float:
    """Coseno TF-IDF entre respuesta y fuente (Salton y Buckley, 1988)."""
    raise NotImplementedError  # Fase 0


def num_overlap(response: str, source: str) -> float:
    """Fracción de números de la respuesta presentes en la fuente."""
    raise NotImplementedError  # Fase 0


def answer_len(response: str, source: str) -> float:
    """Longitud de la respuesta en tokens (la fuente no interviene)."""
    raise NotImplementedError  # Fase 0


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las cinco features a cada fila (response, source) del dataframe.

    Devuelve un dataframe con una columna por feature, alineado por índice.
    """
    raise NotImplementedError  # Fase 0
