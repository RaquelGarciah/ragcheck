"""Parseo de los spans alucinados y sus distribuciones descriptivas.

Alimenta el análisis del Nivel 2 (posición, longitud, categoría gramatical,
tipo de entidad). `spacy` se usa solo como herramienta de análisis, nunca como
clasificador ni para generar features del modelo.
"""

import json
from functools import lru_cache

import pandas as pd


def parse_spans(labels) -> list[dict]:
    """Devuelve la lista de spans alucinados con sus offsets de carácter.

    El campo `hallucination_labels` es JSON; cada span trae `start`, `end`,
    `text` y `label_type`. Lista vacía si la respuesta está limpia o el campo
    no es parseable.
    """
    if isinstance(labels, list):
        return labels
    if not isinstance(labels, str) or labels.strip() in ("", "[]"):
        return []
    try:
        return json.loads(labels)
    except (ValueError, TypeError):
        return []


@lru_cache(maxsize=1)
def _nlp():
    """Carga perezosa del modelo spacy en inglés (solo para el descriptivo)."""
    import spacy

    return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])


def span_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla con una fila por span alucinado y sus atributos descriptivos.

    Columnas: task_type, label_type, position (0-1, inicio relativo en la
    respuesta), length_chars, length_tokens, pos (categoría gramatical
    dominante del span) y entity (primer tipo de entidad, o "NONE").
    """
    nlp = _nlp()
    rows = []
    for response, labels, task in zip(
        df["response"], df["hallucination_labels"], df["task_type"]
    ):
        n = max(len(response), 1)
        for sp in parse_spans(labels):
            text = sp.get("text", response[sp["start"] : sp["end"]])
            doc = nlp(text)
            pos = doc[0].pos_ if len(doc) else "NONE"
            entity = doc.ents[0].label_ if doc.ents else "NONE"
            rows.append(
                {
                    "task_type": task,
                    "label_type": sp.get("label_type", "NA"),
                    "position": sp["start"] / n,
                    "length_chars": sp["end"] - sp["start"],
                    "length_tokens": len(doc),
                    "pos": pos,
                    "entity": entity,
                }
            )
    return pd.DataFrame(rows)
