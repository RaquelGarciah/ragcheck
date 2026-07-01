"""Tests de entrenamiento: grid search y validación cruzada (datos sintéticos)."""

import numpy as np
import pandas as pd

from ragcheck.models import build_logreg
from ragcheck.training import cross_validate, grid_search, top_configs


def _datos():
    rng = np.random.RandomState(0)
    n = 300
    X = pd.DataFrame({"a": rng.rand(n), "b": rng.rand(n)})
    y = (X["a"] + rng.rand(n) * 0.3 > 0.6).astype(int).values
    groups = np.repeat(np.arange(n // 5), 5)
    return X, y, groups


def test_grid_search_encuentra_mejor_y_reajusta():
    X, y, g = _datos()
    gs = grid_search(build_logreg(), {"C": [0.01, 1.0, 100.0]}, X, y, g)
    assert gs.best_params_["C"] in (0.01, 1.0, 100.0)
    # El best_estimator_ queda reentrenado y predice.
    assert gs.best_estimator_.predict_proba(X).shape == (len(y), 2)


def test_top_configs_ordena_por_score():
    X, y, g = _datos()
    gs = grid_search(build_logreg(), {"C": [0.01, 1.0, 100.0]}, X, y, g)
    top = top_configs(gs, k=3)
    assert len(top) == 3
    assert top["mean_test_score"].is_monotonic_decreasing


def test_cross_validate_cubre_todas_las_filas():
    X, y, g = _datos()
    out = cross_validate(build_logreg(), X, y, g)
    assert out["y_prob"].shape == (len(y),)
    assert set(np.unique(out["fold"])) == set(range(5))
