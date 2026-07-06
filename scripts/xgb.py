"""Entrena, ajusta (grid search) y evalúa XGBoost sobre RAGTruth.

Búsqueda en rejilla con GroupKFold por `context` maximizando F1, evaluación por
secciones A–F del mejor modelo, y persistencia. Genera outputs/reports/xgboost.md.

El fichero se llama `xgb.py` (no `xgboost.py`) para no ensombrecer al paquete
`xgboost` al ejecutar el script.
"""

import random

import numpy as np

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_xgboost  # noqa: E402
from ragcheck.report import model_report  # noqa: E402
from ragcheck.training import (  # noqa: E402
    cross_validate,
    fit_final,
    grid_search,
    save_model,
    top_configs,
)

NAME = "xgboost"
PARAM_GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [300, 600],
    "subsample": [0.8, 1.0],
    "reg_lambda": [1, 2],
}


def main() -> None:
    df = load_ragtruth("train")
    X = extract_features(df)
    y = df["label"].values
    groups = df["context"].values

    gs = grid_search(build_xgboost(), PARAM_GRID, X, y, groups)
    print(f"[{NAME}] mejor F1 (CV) = {gs.best_score_:.3f} con {gs.best_params_}")
    print(top_configs(gs).round(4).to_string(index=False))

    p = cross_validate(gs.best_estimator_, X, y, groups)["y_prob"]
    s = model_report(NAME, gs, y, p)
    print(f"[{NAME}] ajustado: AUC-ROC={s['auc_roc']:.3f} F1={s['f1']:.3f} "
          f"bal_acc={s['balanced_accuracy']:.3f} Brier={s['brier']:.3f}")
    save_model(fit_final(gs.best_estimator_, X, y), NAME)


if __name__ == "__main__":
    main()
