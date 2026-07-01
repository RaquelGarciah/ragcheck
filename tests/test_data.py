"""Tests de carga y parseo de RAGTruth.

Verifican el esquema del dataframe y que `has_hallucination` maneja los tres
formatos del campo (lista, "[]" y "[...]"). Se completan en la Fase 0.
"""

import pytest

pytestmark = pytest.mark.skip(reason="pendiente de implementar en la Fase 0")


def test_esquema_columnas_y_dtypes():
    raise NotImplementedError


def test_parseo_hallucination_labels_tres_formatos():
    raise NotImplementedError
