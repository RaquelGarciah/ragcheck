"""Evaluación final honesta en el split de test oficial de RAGTruth.

Carga los cinco modelos ya ajustados (grid search sobre train, GroupKFold por
`source`), los evalúa sobre el test oficial (que no se ha tocado al entrenar) y
consolida la tabla comparativa por secciones A–F. Es el número definitivo y
defendible de cada modelo. Genera outputs/reports/test_oficial.md y las figuras
roc_test / pr_test.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import REPORTS_DIR, SEED

random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.inference import load_model  # noqa: E402
from ragcheck.plotting import savefig, set_style  # noqa: E402

# Nombres de fichero de los modelos guardados (xgboost se llama así, no xgb).
MODELS = ["logreg", "knn", "svm", "random_forest", "xgboost"]


def _row(name, y, p):
    s = ev.summary(y, p)
    lo, hi = s["auc_ci"]
    return {
        "modelo": name,
        "AUC-ROC": s["auc_roc"],
        "IC95%": f"[{lo:.3f}, {hi:.3f}]",
        "AUC-PR": s["auc_pr"],
        "F1": s["f1"],
        "precision": s["precision"],
        "recall": s["recall"],
        "bal_acc": s["balanced_accuracy"],
        "Brier": s["brier"],
        "ECE": s["ece"],
    }


def main() -> None:
    df = load_ragtruth("test")
    X = extract_features(df)
    y = df["label"].values
    print(f"test oficial: n={len(df)}, fracción alucina={y.mean():.3f}")

    rows, curves = [], {}
    for name in MODELS:
        p = load_model(name).predict_proba(X)[:, 1]
        curves[name] = p
        rows.append(_row(name, y, p))

    table = pd.DataFrame(rows).set_index("modelo").sort_values("AUC-ROC", ascending=False)
    print("\n" + table.round(3).to_string())

    set_style()
    _curve_figure(y, curves, roc_curve, "roc_test", "Curvas ROC (test oficial)",
                  "Tasa de falsos positivos", "Tasa de verdaderos positivos", "auc_roc")
    _curve_figure(y, curves, precision_recall_curve, "pr_test",
                  "Curvas precisión-recall (test oficial)", "Recall", "Precision", "auc_pr")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORTS_DIR / "test_oficial.md"
    with open(report, "w") as f:
        f.write("# Evaluación en el test oficial de RAGTruth\n\n")
        f.write(f"n = {len(df)} respuestas; fracción que alucina = {y.mean():.3f}.\n\n")
        f.write("Modelos ajustados por grid search (GroupKFold por `source`) sobre train, ")
        f.write("evaluados sobre el test oficial. Umbral por Youden.\n\n")
        f.write(table.round(3).to_markdown())
        f.write("\n\nFiguras: `outputs/figures/roc_test.*`, `pr_test.*`.\n")
    print(f"\nInforme -> {report}")


def _curve_figure(y, curves, curve_fn, name, title, xlabel, ylabel, metric_key):
    fig, ax = plt.subplots(figsize=(5, 5))
    for model, p in curves.items():
        m = ev.discrimination(y, p)[metric_key]
        if curve_fn is roc_curve:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, label=f"{model} ({m:.3f})")
        else:
            prec, rec, _ = precision_recall_curve(y, p)
            ax.plot(rec, prec, label=f"{model} ({m:.3f})")
    if curve_fn is roc_curve:
        ax.plot([0, 1], [0, 1], "k--", lw=1)
    else:
        ax.axhline(y.mean(), color="k", ls="--", lw=1)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize=8, loc="lower left" if curve_fn is not roc_curve else "lower right")
    savefig(fig, name)


if __name__ == "__main__":
    main()
