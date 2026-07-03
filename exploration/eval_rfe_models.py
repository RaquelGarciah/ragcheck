"""Evaluación de los 5 modelos sobre el conjunto RFE elegido (11 variables).

Para cada modelo: grid search (GroupKFold por `source`, scoring F1) sobre train
restringido a las 11 variables, refit del mejor, evaluación en el test oficial. Saca
la tabla comparativa (P/R/F1/AUC/acc), el ganador por promedio F1+AUC, la ROC por
modelo (roc_<modelo>) y las curvas conjuntas (roc_test_rfe, pr_test_rfe). Escribe
outputs/reports/test_oficial_rfe.md.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.config import REPORTS_DIR  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import (  # noqa: E402
    build_knn, build_logreg, build_random_forest, build_svm, build_xgboost,
)
from ragcheck.plotting import savefig, set_style  # noqa: E402
from ragcheck.training import cross_validate, grid_search  # noqa: E402

CHOSEN = ["containment", "task_QA", "task_Summary", "task_Data2txt", "num_context",
          "answer_len", "jaccard", "sent_cont_min", "novel_bigram", "num_overlap",
          "sent_sim_min"]

# Rejillas modestas por modelo (GroupKFold, F1). SVM se busca sobre submuestra.
GRIDS = {
    "logreg": (build_logreg, {"C": [0.3, 1.0, 3.0], "class_weight": [None, "balanced"]}),
    "knn": (build_knn, {"kneighborsclassifier__n_neighbors": [11, 15, 25],
                        "kneighborsclassifier__weights": ["uniform", "distance"]}),
    "svm": (build_svm, {"svc__C": [1.0, 10.0], "svc__gamma": ["scale"]}),
    "random_forest": (build_random_forest, {"max_depth": [None, 12, 20]}),
    "xgboost": (build_xgboost, {"max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1]}),
}


def main():
    tr, te = load_ragtruth("train"), load_ragtruth("test")
    Xtr = extract_features(tr)[CHOSEN]
    Xte = extract_features(te)[CHOSEN]
    ytr, yte, g = tr["label"].values, te["label"].values, tr["source"].values
    print(f"train n={len(ytr)}, test n={len(yte)}, features={len(CHOSEN)}")

    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(ytr), 5000, replace=False)

    rows, curves, best_params = [], {}, {}
    for name, (build, grid) in GRIDS.items():
        Xg, yg, gg = (Xtr.iloc[sub], ytr[sub], g[sub]) if name == "svm" else (Xtr, ytr, g)
        gs = grid_search(build(), grid, Xg, yg, gg, scoring="f1")
        best = gs.best_estimator_.__class__  # noqa
        model = build().set_params(**gs.best_params_).fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        curves[name] = p
        best_params[name] = gs.best_params_
        s = ev.summary(yte, p)
        rows.append({"modelo": name, "AUC": s["auc_roc"], "F1": s["f1"],
                     "precision": s["precision"], "recall": s["recall"],
                     "accuracy": s["accuracy"], "bal_acc": s["balanced_accuracy"],
                     "prom_F1_AUC": (s["f1"] + s["auc_roc"]) / 2})
        print(f"[{name}] best={gs.best_params_} F1={s['f1']:.3f} AUC={s['auc_roc']:.3f}")

    tab = pd.DataFrame(rows).set_index("modelo").sort_values("prom_F1_AUC", ascending=False)
    winner = tab.index[0]

    set_style()
    # ROC por modelo (para el patrón de Sergio uno-a-uno).
    for name, p in curves.items():
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {ev.discrimination(yte, p)['auc_roc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="RFP (1 - especificidad)", ylabel="RVP (sensibilidad)",
               title=f"Curva ROC — {name}")
        ax.legend(loc="lower right")
        savefig(fig, f"roc_{name}")

    # Curvas conjuntas.
    for curve_fn, tag, title, xl, yl, key, loc in [
        (roc_curve, "roc_test_rfe", "Curvas ROC (test oficial, 11 variables)",
         "RFP", "RVP", "auc_roc", "lower right"),
        (precision_recall_curve, "pr_test_rfe", "Curvas precisión-recall (test oficial)",
         "recall", "precisión", "auc_pr", "lower left")]:
        fig, ax = plt.subplots(figsize=(5, 5))
        for name, p in curves.items():
            m = ev.discrimination(yte, p)[key]
            if curve_fn is roc_curve:
                fpr, tpr, _ = roc_curve(yte, p)
                ax.plot(fpr, tpr, label=f"{name} ({m:.3f})")
            else:
                pr, rc, _ = precision_recall_curve(yte, p)
                ax.plot(rc, pr, label=f"{name} ({m:.3f})")
        if curve_fn is roc_curve:
            ax.plot([0, 1], [0, 1], "k--", lw=1)
        else:
            ax.axhline(yte.mean(), color="k", ls="--", lw=1)
        ax.set(xlabel=xl, ylabel=yl, title=title)
        ax.legend(fontsize=8, loc=loc)
        savefig(fig, tag)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rep = REPORTS_DIR / "test_oficial_rfe.md"
    with open(rep, "w") as f:
        f.write("# Evaluación en test oficial — conjunto RFE (11 variables)\n\n")
        f.write(f"Variables: {', '.join(CHOSEN)}.\n\n")
        f.write(f"n test = {len(yte)}, prevalencia = {yte.mean():.3f}. Grid search "
                "GroupKFold por `source` (F1); umbral por Youden.\n\n")
        f.write(tab.round(3).to_markdown())
        f.write(f"\n\n**Ganador (promedio F1+AUC): {winner}.**\n\n")
        f.write("Hiperparámetros óptimos por modelo:\n\n")
        for name, bp in best_params.items():
            f.write(f"- {name}: {bp}\n")
    print(f"\n{tab.round(3).to_string()}\nGanador: {winner}\nInforme -> {rep}")


if __name__ == "__main__":
    main()
