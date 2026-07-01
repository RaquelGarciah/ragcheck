"""Tests de carga y parseo de RAGTruth.

`has_hallucination` se prueba sin red; el esquema se prueba con un slice
pequeño y se salta si no hay conexión (para no romper CI sin red).
"""

import pytest

from ragcheck.data import COLUMNS, has_hallucination, load_ragtruth


def test_parseo_hallucination_labels_tres_formatos():
    # str "[]" -> limpio; str "[...]" -> alucina; list -> según longitud.
    assert has_hallucination("[]") == 0
    assert has_hallucination('[{"start": 0, "end": 5}]') == 1
    assert has_hallucination([]) == 0
    assert has_hallucination([{"start": 0}]) == 1


def test_parseo_cadena_no_evaluable():
    # Una cadena no parseable pero no vacía cuenta como alucinación.
    assert has_hallucination("texto suelto") == 1
    assert has_hallucination("") == 0


def test_esquema_columnas_y_dtypes():
    try:
        df = load_ragtruth("train[:50]")
    except Exception as e:  # sin red o Hub caído: no bloquear CI
        pytest.skip(f"no se pudo descargar el dataset: {e}")
    assert list(df.columns) == COLUMNS
    assert df["label"].dropna().isin([0, 1]).all()
    assert df[["source", "response"]].notna().all().all()
