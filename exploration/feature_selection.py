"""Selección de variables forward stepwise (wrapper) por modelo.

Responde: ¿cuántas features hacen falta de verdad, o sobran? Para cada modelo se
hace forward selection greedy maximizando F1 en GroupKFold por `context`, y se
registran las seis métricas (auc, f1, recall, precision, acc, bal_acc) en cada
tamaño de subconjunto. La curva es OOF: si se aplana, las features extra son
redundantes; si baja, hay overfitting.

Borrador (NO citable). Genera outputs/reports/feature_selection.md y
outputs/figures/fsel_<modelo>.png. SVM sobre submuestra por coste.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt  # noqa: E402

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.config import REPORTS_DIR  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import BUILDERS  # noqa: E402
from ragcheck.plotting import savefig, set_style  # noqa: E402
from ragcheck.training import cross_validate  # noqa: E402

METRICS = ["auc", "f1", "recall", "precision", "acc", "bal_acc"]


def _oof_metrics(build, X, y, groups):
    """Seis métricas OOF de un subconjunto de features (umbral Youden)."""
    p = cross_validate(build(), X, y, groups)["y_prob"]
    thr = ev.best_threshold(y, p)
    d = ev.discrimination(y, p)
    tm = ev.threshold_metrics(y, p, thr)
    bm = ev.balanced_metrics(y, p, thr)
    return {
        "auc": d["auc_roc"], "f1": tm["f1"], "recall": tm["recall"],
        "precision": tm["precision"], "acc": tm["accuracy"],
        "bal_acc": bm["balanced_accuracy"],
    }


def forward_select(build, X, y, groups, select_on="f1"):
    """Forward stepwise: en cada paso añade la feature que más sube `select_on`."""
    remaining = list(X.columns)
    chosen, rows = [], []
    while remaining:
        best_feat, best_m = None, None
        for c in remaining:
            m = _oof_metrics(build, X[chosen + [c]], y, groups)
            if best_m is None or m[select_on] > best_m[select_on]:
                best_feat, best_m = c, m
        chosen.append(best_feat)
        remaining.remove(best_feat)
        rows.append({"n": len(chosen), "added": best_feat, **best_m})
    return pd.DataFrame(rows)


def _plot(df, name):
    """Curva de las seis métricas frente al número de features (orden de entrada)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for met in METRICS:
        ax.plot(df["n"], df[met], marker="o", ms=3, label=met)
    ax.set_xlabel("nº de features (orden forward stepwise)")
    ax.set_ylabel("métrica (OOF, GroupKFold)")
    ax.set_title(f"{name} — selección forward")
    ax.set_xticks(df["n"])
    ax.set_xticklabels(df["added"], rotation=90, fontsize=6)
    ax.legend(ncol=3, fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    savefig(fig, f"fsel_{name}")


def main():
    set_style()
    df = load_ragtruth("train")
    X = extract_features(df)
    y = df["label"].values
    groups = df["context"].values

    # SVM sobre submuestra por coste (RBF + probability es O(n^2) por fold).
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(y), size=5000, replace=False)

    lines = ["# Selección forward stepwise por modelo (greedy F1, GroupKFold)\n"]
    lines.append("Curva OOF; SVM sobre submuestra de 5000. Umbral Youden por paso.\n")
    for name, build in BUILDERS.items():
        if name == "svm":
            res = forward_select(build, X.iloc[sub], y[sub], groups[sub])
            note = " (submuestra 5000)"
        else:
            res = forward_select(build, X, y, groups)
            note = ""
        _plot(res, name)
        best = res.loc[res["f1"].idxmax()]
        # tamaño mínimo que llega al 99% del mejor F1 (rodilla práctica)
        knee = int(res[res["f1"] >= 0.99 * best["f1"]]["n"].min())
        lines.append(f"\n## {name}{note}\n")
        lines.append(f"- Mejor F1 = {best['f1']:.3f} con {int(best['n'])} features "
                     f"(AUC {best['auc']:.3f}, acc {best['acc']:.3f}).")
        lines.append(f"- **Rodilla (99% del mejor F1): {knee} features.** "
                     f"Orden de entrada: {', '.join(res['added'])}.\n")
        lines.append(res.round(3).to_markdown(index=False))
        print(f"[{name}] mejor F1={best['f1']:.3f} @ {int(best['n'])} feats; "
              f"rodilla {knee}")

    report = REPORTS_DIR / "feature_selection.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"Informe -> {report}")


if __name__ == "__main__":
    main()
