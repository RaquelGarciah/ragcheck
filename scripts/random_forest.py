"""Entrena, ajusta (grid search) y evalúa el bosque aleatorio sobre RAGTruth.

Búsqueda en rejilla con GroupKFold por `source` maximizando F1, evaluación por
secciones A–F del mejor modelo, y persistencia. Genera
outputs/reports/random_forest.md.
"""

import random

import numpy as np

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_random_forest  # noqa: E402
from ragcheck.report import model_report  # noqa: E402
from ragcheck.training import (  # noqa: E402
    cross_validate,
    fit_final,
    grid_search,
    save_model,
    top_configs,
)

NAME = "random_forest"
PARAM_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [None, 6, 12],
    "max_features": ["sqrt", 0.5],
    "min_samples_leaf": [1, 5],
}


def main() -> None:
    df = load_ragtruth("train")
    X = extract_features(df)
    y = df["label"].values
    groups = df["source"].values

    gs = grid_search(build_random_forest(), PARAM_GRID, X, y, groups)
    print(f"[{NAME}] mejor F1 (CV) = {gs.best_score_:.3f} con {gs.best_params_}")
    print(top_configs(gs).round(4).to_string(index=False))

    p = cross_validate(gs.best_estimator_, X, y, groups)["y_prob"]
    s = model_report(NAME, gs, y, p)
    print(f"[{NAME}] ajustado: AUC-ROC={s['auc_roc']:.3f} F1={s['f1']:.3f} "
          f"bal_acc={s['balanced_accuracy']:.3f} Brier={s['brier']:.3f}")
    save_model(fit_final(gs.best_estimator_, X, y), NAME)


if __name__ == "__main__":
    main()
