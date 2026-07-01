"""Parseo de los spans alucinados y sus distribuciones descriptivas.

Alimenta el análisis del Nivel 2 (posición, longitud, categoría gramatical,
tipo de entidad). `spacy` se usa solo como herramienta de análisis, nunca como
clasificador ni para generar features del modelo.
"""

import pandas as pd


def parse_spans(labels) -> list[dict]:
    """Devuelve la lista de spans alucinados con sus offsets de carácter.

    Cada span es un dict con al menos `start` y `end`; lista vacía si la
    respuesta está limpia.
    """
    raise NotImplementedError  # Fase 0


def span_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla por span con posición relativa, longitud, POS y tipo de entidad.

    Entrada: dataframe con `response` y `hallucination_labels`. Salida: una
    fila por span alucinado con las columnas del descriptivo de Nivel 2.
    """
    raise NotImplementedError  # Fase 0
