"""Test de determinismo: dos entrenamientos con la misma semilla coinciden.

Sobre datos sintéticos (sin red): dos validaciones cruzadas idénticas producen
exactamente las mismas probabilidades fuera de muestra.
"""

import numpy as np
import pandas as pd

from ragcheck.models import build_logreg, build_xgboost
from ragcheck.training import cross_validate


def _datos():
    rng = np.random.RandomState(0)
    n = 300
    X = pd.DataFrame({"a": rng.rand(n), "b": rng.rand(n)})
    y = (X["a"] + rng.rand(n) * 0.3 > 0.6).astype(int).values
    groups = np.repeat(np.arange(n // 5), 5)  # 60 grupos de 5
    return X, y, groups


def test_logreg_deterministico():
    X, y, g = _datos()
    p1 = cross_validate(build_logreg(), X, y, g)["y_prob"]
    p2 = cross_validate(build_logreg(), X, y, g)["y_prob"]
    assert np.array_equal(p1, p2)


def test_xgboost_deterministico():
    X, y, g = _datos()
    p1 = cross_validate(build_xgboost(), X, y, g)["y_prob"]
    p2 = cross_validate(build_xgboost(), X, y, g)["y_prob"]
    assert np.array_equal(p1, p2)
