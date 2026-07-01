"""Test crítico: la partición GroupKFold por `source` no filtra documentos.

En cada fold, ninguna `source` aparece a la vez en train y test. Su fallo
bloquea cualquier merge a main.
"""

import numpy as np
from sklearn.model_selection import GroupKFold

from ragcheck.config import N_SPLITS


def test_groupkfold_sin_solape_de_source():
    # Documentos repetidos: cada `source` aparece en varias filas.
    groups = np.repeat(np.arange(20), 5)  # 20 fuentes, 5 respuestas cada una
    X = np.zeros((groups.size, 1))
    y = np.random.RandomState(0).randint(0, 2, size=groups.size)

    gkf = GroupKFold(n_splits=N_SPLITS)
    for train_idx, test_idx in gkf.split(X, y, groups):
        train_sources = set(groups[train_idx])
        test_sources = set(groups[test_idx])
        assert train_sources.isdisjoint(test_sources)
