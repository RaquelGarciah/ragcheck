"""Carga y preparación de RAGTruth-processed.

Frontera de datos: aquí se valida el esquema y se parsea la etiqueta; el
resto del pipeline asume el dataframe ya limpio.
"""

import pandas as pd


def has_hallucination(labels) -> int:
    """Convierte `hallucination_labels` en etiqueta binaria (1 = alucina).

    Maneja los tres formatos posibles del campo: lista, cadena "[]" y
    cadena "[...]" con spans.
    """
    raise NotImplementedError  # Fase 0


def load_ragtruth(split: str) -> pd.DataFrame:
    """Descarga RAGTruth y devuelve columnas [id, source, response, label,
    task_type, model].

    Renombra `context` -> `source` y `output` -> `response`, y añade `label`.
    `split` es "train" o "test".
    """
    raise NotImplementedError  # Fase 0
