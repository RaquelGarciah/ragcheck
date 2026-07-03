"""Selección de variables definitiva: RFE + método del codo sobre F1.

Calcula el ranking RFE (eliminación recursiva por importancia de xgboost), evalúa
las 6 métricas OOF (GroupKFold por `source`, umbral Youden) para cada tamaño de
subconjunto añadiendo en orden RFE, y elige el número de variables por el CODO de la
curva de F1 (menor k que alcanza el 99 % del mejor F1). Genera fsel_rfe_codo y deja
por stdout el conjunto definitivo.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt  # noqa: E402

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_xgboost  # noqa: E402
from ragcheck.plotting import savefig, set_style  # noqa: E402
from ragcheck.training import cross_validate  # noqa: E402

METRICS = ["acc", "f1", "recall", "precision", "auc", "bal_acc"]
LABELS = {"acc": "accuracy", "f1": "F1", "recall": "recall",
          "precision": "precisión", "auc": "AUC", "bal_acc": "bal-acc"}


def oof(cols, X, y, g):
    p = cross_validate(build_xgboost(), X[cols], y, g)["y_prob"]
    thr = ev.best_threshold(y, p)
    d, tm = ev.discrimination(y, p), ev.threshold_metrics(y, p, thr)
    return {"auc": d["auc_roc"], "f1": tm["f1"], "recall": tm["recall"],
            "precision": tm["precision"], "acc": tm["accuracy"],
            "bal_acc": ev.balanced_metrics(y, p, thr)["balanced_accuracy"]}


def main():
    df = load_ragtruth("train")
    X, y, g = extract_features(df), df["label"].values, df["source"].values

    # Ranking RFE: elimina recursivamente la menos importante (xgboost).
    cur, dropped = list(X.columns), []
    while len(cur) > 1:
        m = build_xgboost().fit(X[cur], y)
        drop = pd.Series(m.feature_importances_, index=cur).idxmin()
        dropped.append(drop)
        cur.remove(drop)
    ranking = cur + list(reversed(dropped))  # más → menos importante
    print("Ranking RFE (más→menos):", ranking)

    rows = [oof(ranking[:k], X, y, g) for k in range(1, len(ranking) + 1)]
    curve = pd.DataFrame(rows, index=range(1, len(ranking) + 1))

    # Codo sobre F1: menor k con F1 >= 99% del mejor F1.
    f1 = curve["f1"].values
    k_codo = int(np.argmax(f1 >= 0.99 * f1.max()) + 1)
    chosen = ranking[:k_codo]

    set_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for met in METRICS:
        ax.plot(curve.index, curve[met], marker="o", ms=3, label=LABELS[met])
    ax.axvline(k_codo, color="grey", ls="--", lw=1.2)
    ax.text(k_codo + 0.15, 0.55, f"codo F1: k={k_codo}", fontsize=9, color="grey")
    ax.set_xticks(curve.index)
    ax.set_xticklabels([f"{k}\n{ranking[k-1][:10]}" for k in curve.index],
                       rotation=90, fontsize=6)
    ax.set(xlabel="nº de variables (orden RFE)", ylabel="métrica (OOF, GroupKFold)",
           title="Selección RFE: métricas frente al nº de variables (codo por F1)")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    savefig(fig, "fsel_rfe_codo")

    print(f"\nCODO por F1 -> k = {k_codo}")
    print(f"CONJUNTO DEFINITIVO ({k_codo}): {chosen}")
    print("\nCurva de métricas por k:")
    print(curve.round(3).to_string())
    print(f"\nMejor F1 OOF={f1.max():.3f}; en el codo F1={curve.loc[k_codo,'f1']:.3f}, "
          f"AUC={curve.loc[k_codo,'auc']:.3f}, acc={curve.loc[k_codo,'acc']:.3f}")


if __name__ == "__main__":
    main()
