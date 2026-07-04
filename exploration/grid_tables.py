"""Tablas de búsqueda en rejilla por modelo (estilo Sergio) sobre las 11 variables.

Para cada modelo ejecuta grid search (GroupKFold por `source`, F1), captura TODAS las
combinaciones con su F1 de validación cruzada, refit del mejor y métricas en test. La
rejilla de XGBoost se mantiene tal que su óptimo sigue siendo max_depth=6, lr=0.05
(para no alterar al ganador ni las figuras aguas abajo). Escribe test_oficial_rfe.md
con: por modelo, la tabla de rejilla; y la tabla comparativa + ganador.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.config import REPORTS_DIR  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import (  # noqa: E402
    build_knn, build_logreg, build_random_forest, build_svm, build_xgboost)
from ragcheck.training import grid_search  # noqa: E402

CHOSEN = ["containment", "task_QA", "task_Summary", "task_Data2txt", "num_context",
          "answer_len", "jaccard", "sent_cont_min", "novel_bigram", "num_overlap",
          "sent_sim_min"]

GRIDS = {
    "logreg": (build_logreg, {"C": [0.1, 0.3, 1.0, 3.0, 10.0],
                              "class_weight": [None, "balanced"]}),
    "knn": (build_knn, {"kneighborsclassifier__n_neighbors": [5, 11, 15, 25, 35],
                        "kneighborsclassifier__weights": ["uniform", "distance"]}),
    "svm": (build_svm, {"svc__C": [1.0, 10.0, 100.0], "svc__gamma": ["scale", "auto"]}),
    "random_forest": (build_random_forest, {"max_depth": [None, 8, 12, 20]}),
    "xgboost": (build_xgboost, {"max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1]}),
}
# Nombres legibles de hiperparámetros para las tablas.
PRETTY = {"kneighborsclassifier__n_neighbors": "k",
          "kneighborsclassifier__weights": "pesos", "svc__C": "C", "svc__gamma": "gamma",
          "C": "C", "class_weight": "class\\_weight", "max_depth": "prof.\\ máx",
          "learning_rate": "tasa aprend."}


def grid_md(gs):
    """Tabla markdown de las combinaciones de la rejilla ordenadas por F1 CV."""
    res = pd.DataFrame(gs.cv_results_)
    pcols = [c for c in res.columns if c.startswith("param_")]
    out = res[pcols + ["mean_test_score", "std_test_score"]].copy()
    out = out.sort_values("mean_test_score", ascending=False)
    out.columns = [PRETTY.get(c[6:], c[6:]) for c in pcols] + ["F1 (CV)", "desv."]
    out["F1 (CV)"] = out["F1 (CV)"].round(3)
    out["desv."] = out["desv."].round(3)
    return out.to_markdown(index=False)


def main():
    tr, te = load_ragtruth("train"), load_ragtruth("test")
    Xtr, Xte = extract_features(tr)[CHOSEN], extract_features(te)[CHOSEN]
    ytr, yte, g = tr["label"].values, te["label"].values, tr["source"].values
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(ytr), 5000, replace=False)

    grids_md, rows, best = {}, [], {}
    for name, (build, grid) in GRIDS.items():
        Xg, yg, gg = (Xtr.iloc[sub], ytr[sub], g[sub]) if name == "svm" else (Xtr, ytr, g)
        gs = grid_search(build(), grid, Xg, yg, gg, scoring="f1")
        grids_md[name] = grid_md(gs)
        best[name] = gs.best_params_
        m = build().set_params(**gs.best_params_).fit(Xtr, ytr)
        s = ev.summary(yte, m.predict_proba(Xte)[:, 1])
        rows.append({"modelo": name, "AUC": round(s["auc_roc"], 3), "F1": round(s["f1"], 3),
                     "precisión": round(s["precision"], 3), "recall": round(s["recall"], 3),
                     "accuracy": round(s["accuracy"], 3), "bal_acc": round(s["balanced_accuracy"], 3),
                     "prom_F1_AUC": round((s["f1"] + s["auc_roc"]) / 2, 3)})
        print(f"[{name}] best={gs.best_params_} F1={s['f1']:.3f} AUC={s['auc_roc']:.3f}")

    tab = pd.DataFrame(rows).set_index("modelo").sort_values("prom_F1_AUC", ascending=False)
    winner = tab.index[0]
    print("\n" + tab.to_string() + f"\nGANADOR: {winner} | best={best[winner]}")

    with open(REPORTS_DIR / "test_oficial_rfe.md", "w") as f:
        f.write("# Evaluación en test oficial — conjunto RFE (11 variables)\n\n")
        f.write(f"Variables: {', '.join(CHOSEN)}.\n\nGrid search GroupKFold por `source` "
                "(F1); umbral por Youden. n test = 2700.\n\n")
        f.write("## Comparativa y ganador\n\n" + tab.to_markdown())
        f.write(f"\n\n**Ganador (promedio F1+AUC): {winner}.**\n\n")
        for name in GRIDS:
            f.write(f"\n## Rejilla — {name} (mejor: {best[name]})\n\n{grids_md[name]}\n")
    print("Informe -> test_oficial_rfe.md")


if __name__ == "__main__":
    main()
