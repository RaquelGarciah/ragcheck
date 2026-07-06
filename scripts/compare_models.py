"""Comparativa de los cinco modelos: reproduce la línea base del TFG.

Entrena y evalúa los cinco clasificadores con la misma partición GroupKFold por
`context`, consolida las métricas A–F en una tabla comparativa (con IC de
bootstrap para el AUC), superpone las curvas ROC y PR, y añade baselines
honestos (azar, clase mayoritaria, una sola feature). Escribe
outputs/reports/e0_baseline.md y las figuras en outputs/figures/.
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
from ragcheck.models import BUILDERS, build_logreg  # noqa: E402
from ragcheck.plotting import savefig, set_style  # noqa: E402
from ragcheck.training import cross_validate  # noqa: E402


def _row(name, y, p):
    """Fila de la tabla A–F para unas probabilidades fuera de muestra `p`."""
    thr = ev.best_threshold(y, p)
    a = ev.discrimination(y, p)
    b = ev.threshold_metrics(y, p, thr)
    c = ev.balanced_metrics(y, p, thr)
    d = ev.calibration(y, p)
    _, lo, hi = ev.bootstrap_ci(y, p)
    return {
        "modelo": name,
        "AUC-ROC": a["auc_roc"],
        "IC95%": f"[{lo:.3f}, {hi:.3f}]",
        "AUC-PR": a["auc_pr"],
        "F1": b["f1"],
        "precision": b["precision"],
        "recall": b["recall"],
        "bal_acc": c["balanced_accuracy"],
        "Brier": d["brier"],
        "ECE": d["ece"],
    }


def main() -> None:
    df = load_ragtruth("train")
    X = extract_features(df)
    y = df["label"].values
    groups = df["context"].values

    rows, curves = [], {}
    for name, build in BUILDERS.items():
        print(f"[{name}] validando...", flush=True)
        p = cross_validate(build(), X, y, groups)["y_prob"]
        curves[name] = p
        rows.append(_row(name, y, p))

    # Baseline de una sola feature (containment), con el mismo protocolo.
    p_cont = cross_validate(build_logreg(), X[["containment"]], y, groups)["y_prob"]
    rows.append(_row("baseline: solo containment", y, p_cont))
    # Baseline trivial de clase mayoritaria.
    rows.append(
        {
            "modelo": "baseline: clase mayoritaria",
            "AUC-ROC": 0.5,
            "IC95%": "--",
            "AUC-PR": float(y.mean()),
            "F1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "bal_acc": 0.5,
            "Brier": float(np.mean((y - y.mean()) ** 2)),
            "ECE": np.nan,
        }
    )

    table = pd.DataFrame(rows).set_index("modelo")
    print("\n" + table.round(3).to_string())

    set_style()
    _roc_figure(y, curves)
    _pr_figure(y, curves)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORTS_DIR / "e0_baseline.md"
    with open(report, "w") as f:
        f.write("# E0 — Línea base: comparativa de modelos\n\n")
        f.write(f"n = {len(df)} respuestas, {df['context'].nunique()} documentos.\n\n")
        f.write("Validación GroupKFold por `context`. Umbral por índice de Youden.\n\n")
        f.write(table.round(3).to_markdown())
        f.write("\n\nFiguras: `outputs/figures/roc_modelos.*`, `pr_modelos.*`.\n")
    print(f"\nInforme -> {report}")


def _roc_figure(y, curves):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={ev.discrimination(y, p)['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC por modelo")
    ax.legend(fontsize=8, loc="lower right")
    savefig(fig, "roc_modelos")


def _pr_figure(y, curves):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP={ev.discrimination(y, p)['auc_pr']:.3f})")
    ax.axhline(y.mean(), color="k", ls="--", lw=1, label=f"azar ({y.mean():.2f})")
    ax.set(xlabel="Recall", ylabel="Precision")
    ax.set_title("Curvas precisión-recall por modelo")
    ax.legend(fontsize=8, loc="lower left")
    savefig(fig, "pr_modelos")


if __name__ == "__main__":
    main()
