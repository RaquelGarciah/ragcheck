"""Features léxicas y estadísticas sobre el par (respuesta, fuente).

Notación: R = palabras únicas de la respuesta, F = palabras únicas de la
fuente. Todas las funciones reciben `(response, source)` y devuelven un float.
"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_SENT = re.compile(r"[^.!?\n]+")

# Tareas de RAGTruth; se codifican one-hot para que el modelo especialice.
TASKS = ("QA", "Summary", "Data2txt")


def _tokens(text: str) -> list[str]:
    """Tokeniza a minúsculas quedándose con secuencias alfanuméricas."""
    return _TOKEN.findall(text.lower())


def _sentences(text: str) -> list[str]:
    """Trocea en frases por signos de puntuación fuertes y saltos de línea.

    Descarta los fragmentos sin tokens alfanuméricos (puntuación suelta).
    """
    return [s for s in _SENT.findall(text) if _TOKEN.search(s)]


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


# Orden canónico de las features escalares por par (respuesta, fuente).
FEATURES = {
    "containment": containment,
    "jaccard": jaccard,
    "tfidf_cos": tfidf_cos,
    "num_overlap": num_overlap,
    "answer_len": answer_len,
}

# Columnas del bloque de nivel frase, en orden.
SENTENCE_FEATURES = (
    "sent_cont_min",
    "sent_cont_mean",
    "sent_cont_std",
    "sent_frac_low",
    "sent_sim_min",
    "sent_sim_mean",
)


def _sentence_features(response: str, source: str) -> dict[str, float]:
    """Agrega el soporte de cada frase de la respuesta contra la fuente.

    La intuición: una alucinación suele ser *una* frase sin respaldo dentro de
    una respuesta por lo demás correcta, y un promedio global la diluye. Aquí se
    mira frase a frase y se agrega con el mínimo (la frase peor sostenida delata)
    y la dispersión. Dos señales de soporte por frase:

    - `cont`: containment léxico de la frase contra el vocabulario de la fuente.
    - `sim`: mejor coseno TF-IDF de la frase contra alguna frase de la fuente
      (TF-IDF ajustado solo con este par, sin fuga entre muestras).

    Devuelve las seis columnas de `SENTENCE_FEATURES`. Respuesta sin frases con
    contenido: valores neutros (soporte pleno, dispersión nula).
    """
    neutral = dict.fromkeys(SENTENCE_FEATURES, 0.0)
    neutral["sent_cont_min"] = neutral["sent_cont_mean"] = 1.0
    neutral["sent_sim_min"] = neutral["sent_sim_mean"] = 1.0

    rs = _sentences(response)
    if not rs:
        return neutral

    F = _words(source)
    cont = np.array([len(w & F) / len(w) for s in rs if (w := _words(s))])

    ss = _sentences(source)
    if ss:
        try:
            V = TfidfVectorizer().fit(rs + ss)
            sim = cosine_similarity(V.transform(rs), V.transform(ss)).max(axis=1)
        except ValueError:  # vocabulario vacío tras tokenizar
            sim = np.ones(len(rs))
    else:
        sim = np.zeros(len(rs))

    return {
        "sent_cont_min": float(cont.min()),
        "sent_cont_mean": float(cont.mean()),
        "sent_cont_std": float(cont.std()),
        "sent_frac_low": float((cont < 0.5).mean()),
        "sent_sim_min": float(sim.min()),
        "sent_sim_mean": float(sim.mean()),
    }


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de diseño a partir de (response, source[, task_type]).

    Tres bloques, alineados por índice:
    - las cinco features escalares de `FEATURES`;
    - el bloque de nivel frase de `SENTENCE_FEATURES`;
    - one-hot de `task_type` (`task_<QA|Summary|Data2txt>`) si la columna existe.
    """
    data = {
        name: [f(r, s) for r, s in zip(df["response"], df["source"])]
        for name, f in FEATURES.items()
    }
    sent = [_sentence_features(r, s) for r, s in zip(df["response"], df["source"])]
    for col in SENTENCE_FEATURES:
        data[col] = [row[col] for row in sent]

    if "task_type" in df.columns:
        for task in TASKS:
            data[f"task_{task}"] = (df["task_type"] == task).astype(float).tolist()

    return pd.DataFrame(data, index=df.index)
