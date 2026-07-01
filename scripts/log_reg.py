"""Entrena y evalúa la regresión logística sobre RAGTruth.

Carga los datos, extrae features, valida con GroupKFold por `source`, calcula
las métricas A–F, guarda el modelo entrenado e imprime el resumen.
"""

import random

import numpy as np

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_logreg  # noqa: E402
from ragcheck.training import cross_validate, fit_final, save_model  # noqa: E402

NAME = "logreg"


def main() -> None:
    df = load_ragtruth("train")
    X = extract_features(df)
    y = df["label"].values
    p = cross_validate(build_logreg(), X, y, df["source"].values)["y_prob"]
    s = ev.summary(y, p)
    print(f"[{NAME}] AUC-ROC={s['auc_roc']:.3f} IC95%={tuple(round(v,3) for v in s['auc_ci'])} "
          f"AUC-PR={s['auc_pr']:.3f} F1={s['f1']:.3f} bal_acc={s['balanced_accuracy']:.3f} "
          f"Brier={s['brier']:.3f} ECE={s['ece']:.3f}")
    save_model(fit_final(build_logreg(), X, y), NAME)


if __name__ == "__main__":
    main()
