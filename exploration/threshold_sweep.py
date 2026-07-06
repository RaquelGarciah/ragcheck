"""Barrido de umbrales y comparación de estrategias de decisión.

Borrador (NO citable). Responde a una pregunta metodológica: ¿cómo cambian
precision/recall/F1/balanced-accuracy al mover el umbral, y qué criterio de
selección de umbral conviene? Se compara con lo que hace Sergio (umbral fijo
0.5) frente a lo nuestro (Youden sobre las OOF).

Usa OOF de GroupKFold por `context` (mismo protocolo honesto que el resto del
repo). Dos modelos representativos: logreg (lineal) y xgboost (fuerte).
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

from sklearn.metrics import roc_curve  # noqa: E402

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_logreg, build_xgboost  # noqa: E402
from ragcheck.training import cross_validate  # noqa: E402


def sweep(y, p, grid):
    """Métricas al variar el umbral sobre una malla fija."""
    rows = []
    for t in grid:
        m = ev.threshold_metrics(y, p, t)
        rows.append(
            {
                "umbral": round(t, 2),
                "precision": m["precision"],
                "recall": m["recall"],
                "F1": m["f1"],
                "specificity": m["specificity"],
                "bal_acc": ev.balanced_metrics(y, p, t)["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows).set_index("umbral")


def strategies(y, p):
    """Cuatro criterios distintos de fijar el umbral, comparados en la misma tabla.

    - fijo_0.5      : lo que hace Sergio.
    - youden        : maximiza sensibilidad+especificidad (lo actual del repo).
    - max_F1        : maximiza F1 directamente.
    - prevalencia   : umbral que iguala la tasa de positivos predicha a la real.
    """
    thr = {
        "fijo_0.5": 0.5,
        "youden": ev.best_threshold(y, p),
        "max_F1": _max_f1_threshold(y, p),
        "prevalencia": float(np.quantile(p, 1.0 - y.mean())),
    }
    rows = []
    for name, t in thr.items():
        m = ev.threshold_metrics(y, p, t)
        rows.append(
            {
                "estrategia": name,
                "umbral": round(t, 3),
                "precision": m["precision"],
                "recall": m["recall"],
                "F1": m["f1"],
                "bal_acc": ev.balanced_metrics(y, p, t)["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows).set_index("estrategia")


def _max_f1_threshold(y, p):
    """Umbral que maximiza F1 recorriendo los puntos de corte de la curva ROC."""
    _, _, thr = roc_curve(y, p)
    f1s = [ev.threshold_metrics(y, p, t)["f1"] for t in thr]
    return float(thr[int(np.argmax(f1s))])


def nested_threshold_auc(y, p, folds):
    """Honesto: umbral elegido por Youden EN train de cada fold, aplicado a test.

    Evita la trampa de elegir el umbral sobre las mismas OOF que se evalúan.
    Devuelve F1 y bal_acc agregados fuera de muestra.
    """
    y_pred = np.zeros(len(y), dtype=int)
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        t = ev.best_threshold(y[tr], p[tr])
        y_pred[te] = (p[te] >= t).astype(int)
    tp = int(((y == 1) & (y_pred == 1)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "F1": f1}


def main():
    df = load_ragtruth("train")
    X = extract_features(df)
    y = df["label"].values
    groups = df["context"].values
    grid = np.arange(0.1, 0.91, 0.1)

    for name, build in [("logreg", build_logreg), ("xgboost", build_xgboost)]:
        print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
        cv = cross_validate(build(), X, y, groups)
        p, fold = cv["y_prob"], cv["fold"]
        auc = ev.discrimination(y, p)["auc_roc"]
        print(f"AUC-ROC (indep. del umbral) = {auc:.4f}\n")

        print("— Barrido de umbrales —")
        print(sweep(y, p, grid).round(3).to_string())

        print("\n— Estrategias de selección de umbral (sobre las mismas OOF) —")
        print(strategies(y, p).round(3).to_string())

        print("\n— Youden honesto (umbral fijado en train de cada fold) —")
        print({k: round(v, 3) for k, v in nested_threshold_auc(y, p, fold).items()})


if __name__ == "__main__":
    main()
