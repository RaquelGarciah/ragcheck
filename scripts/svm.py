"""Entrena, ajusta (grid search) y evalúa la SVM de núcleo RBF sobre RAGTruth.

Búsqueda en rejilla con GroupKFold por `source` maximizando F1, evaluación por
secciones A–F del mejor modelo, y persistencia. Genera outputs/reports/svm.md.

La rejilla se busca SIN `probability` (F1 usa `predict`, mucho más rápido); solo
la mejor configuración se reentrena con `probability=True` para obtener las
probabilidades que necesitan las curvas y la calibración. El estandarizado
previo es imprescindible (ver build_svm).
"""

import random

import numpy as np

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.report import model_report  # noqa: E402
from ragcheck.training import (  # noqa: E402
    cross_validate,
    fit_final,
    grid_search,
    save_model,
    top_configs,
)

NAME = "svm"
PARAM_GRID = {
    "svc__C": [0.5, 1.0, 10.0],
    "svc__gamma": ["scale", 0.1, 1.0],
}


def _svm(probability: bool, **params) -> object:
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=probability, random_state=SEED, **params),
    )


def main() -> None:
    df = load_ragtruth("train")
    X = extract_features(df)
    y = df["label"].values
    groups = df["source"].values

    gs = grid_search(_svm(probability=False), PARAM_GRID, X, y, groups)
    print(f"[{NAME}] mejor F1 (CV) = {gs.best_score_:.3f} con {gs.best_params_}")
    print(top_configs(gs).round(4).to_string(index=False))

    best = {k.split("__")[1]: v for k, v in gs.best_params_.items()}
    final = _svm(probability=True, **best)
    p = cross_validate(final, X, y, groups)["y_prob"]
    s = model_report(NAME, gs, y, p)
    print(f"[{NAME}] ajustado: AUC-ROC={s['auc_roc']:.3f} F1={s['f1']:.3f} "
          f"bal_acc={s['balanced_accuracy']:.3f} Brier={s['brier']:.3f}")
    save_model(fit_final(final, X, y), NAME)


if __name__ == "__main__":
    main()
