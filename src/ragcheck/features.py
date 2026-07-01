"""Features léxicas y estadísticas sobre el par (respuesta, fuente).

Notación: R = palabras únicas de la respuesta, F = palabras únicas de la
fuente. Todas las funciones reciben `(response, source)` y devuelven un float.
"""

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _tokens(text: str) -> list[str]:
    """Tokeniza a minúsculas quedándose con secuencias alfanuméricas."""
    return _TOKEN.findall(text.lower())


def _words(text: str) -> set[str]:
    """Conjunto de palabras únicas del texto."""
    return set(_tokens(text))


def _numbers(text: str) -> set[str]:
    """Conjunto de números (enteros o decimales) que aparecen en el texto."""
    return set(_NUMBER.findall(text))


def containment(response: str, source: str) -> float:
    """Precisión léxica: |R ∩ F| / |R|. Señal dominante del detector.

    Vale 0 si la respuesta no tiene palabras.
    """
    R = _words(response)
    if not R:
        return 0.0
    F = _words(source)
    return len(R & F) / len(R)


def jaccard(response: str, source: str) -> float:
    """Solape simétrico: |R ∩ F| / |R ∪ F|. Vale 0 si ambos están vacíos."""
    R = _words(response)
    F = _words(source)
    union = R | F
    if not union:
        return 0.0
    return len(R & F) / len(union)


def tfidf_cos(response: str, source: str) -> float:
    """Coseno TF-IDF entre respuesta y fuente (Salton y Buckley, 1988).

    Ajusta el vocabulario solo con este par, así que no hay fuga entre
    muestras. Vale 0 si alguno no aporta términos.
    """
    if not response.strip() or not source.strip():
        return 0.0
    try:
        X = TfidfVectorizer().fit_transform([response, source])
    except ValueError:  # vocabulario vacío tras tokenizar
        return 0.0
    return float(cosine_similarity(X[0], X[1])[0, 0])


def num_overlap(response: str, source: str) -> float:
    """Fracción de números de la respuesta presentes en la fuente.

    Caza fechas y cifras inventadas. Vale 1 si la respuesta no tiene números
    (no hay ninguno que pueda estar sin respaldo).
    """
    NR = _numbers(response)
    if not NR:
        return 1.0
    NF = _numbers(source)
    return len(NR & NF) / len(NR)


def answer_len(response: str, source: str) -> float:
    """Longitud de la respuesta en tokens (la fuente no interviene)."""
    return float(len(_tokens(response)))


# Orden canónico de las features en la matriz de diseño.
FEATURES = {
    "containment": containment,
    "jaccard": jaccard,
    "tfidf_cos": tfidf_cos,
    "num_overlap": num_overlap,
    "answer_len": answer_len,
}


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las cinco features a cada fila (response, source) del dataframe.

    Devuelve un dataframe con una columna por feature, alineado por índice.
    """
    data = {
        name: [f(r, s) for r, s in zip(df["response"], df["source"])]
        for name, f in FEATURES.items()
    }
    return pd.DataFrame(data, index=df.index)
