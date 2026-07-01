"""Carga y preparación de RAGTruth-processed.

Frontera de datos: aquí se valida el esquema y se parsea la etiqueta; el
resto del pipeline asume el dataframe ya limpio.
"""

import ast

import pandas as pd
from datasets import load_dataset

from ragcheck.config import DATASET

COLUMNS = ["id", "source", "response", "label", "task_type", "model"]


def has_hallucination(labels) -> int:
    """Convierte `hallucination_labels` en etiqueta binaria (1 = alucina).

    Maneja los tres formatos posibles del campo: lista, cadena "[]" y
    cadena "[...]" con spans.
    """
    if isinstance(labels, str):
        try:
            labels = ast.literal_eval(labels)
        except (ValueError, SyntaxError):
            return 1 if labels.strip() not in ("", "[]") else 0
    return 1 if (labels is not None and len(labels) > 0) else 0


def load_ragtruth(split: str, keep_spans: bool = False) -> pd.DataFrame:
    """Descarga RAGTruth y devuelve columnas [id, source, response, label,
    task_type, model].

    Renombra `context` -> `source` y `output` -> `response`, y añade `label`.
    `split` es "train" o "test". Con `keep_spans=True` conserva además la
    columna cruda `hallucination_labels` (la necesita el descriptivo de spans).
    """
    df = load_dataset(DATASET, split=split).to_pandas()
    df = df.rename(columns={"context": "source", "output": "response"})
    df["label"] = df["hallucination_labels"].map(has_hallucination)
    cols = COLUMNS + ["hallucination_labels"] if keep_spans else COLUMNS
    return df[cols].reset_index(drop=True)
