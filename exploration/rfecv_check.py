"""RFECV: contraste del forward stepwise con otro wrapper (eliminación recursiva).

Borrador (NO citable). Segundo método wrapper para la selección de variables, como
contraste del forward stepwise. RFECV (Guyon et al., 2002) elimina recursivamente la
feature menos importante y elige por validación cruzada el número óptimo. Protocolo
honesto: GroupKFold por `context`, scoring F1. Solo aplica a modelos con importancias
(logreg, random_forest, xgboost); SVM-RBF y KNN no exponen `coef_`/`feature_importances_`.

Para cada modelo compara TODAS las métricas OOF (auc, f1, recall, precision, acc,
bal_acc, umbral Youden) de tres conjuntos: RFECV, forward-7 (recomendado) y las 18.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

from sklearn.feature_selection import RFECV  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.config import N_SPLITS  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_logreg, build_random_forest, build_xgboost  # noqa: E402
from ragcheck.training import cross_validate  # noqa: E402

METRICS = ["auc", "f1", "recall", "precision", "acc", "bal_acc"]
FORWARD7 = ["containment", "answer_len", "num_overlap", "jaccard",
            "neg_diff", "num_context", "sent_sim_min"]


def oof(build, X, y, groups) -> dict:
    """Seis métricas OOF de un subconjunto (umbral Youden), protocolo honesto."""
    p = cross_validate(build(), X, y, groups)["y_prob"]
    thr = ev.best_threshold(y, p)
    d, tm = ev.discrimination(y, p), ev.threshold_metrics(y, p, thr)
    bm = ev.balanced_metrics(y, p, thr)
    return {"auc": d["auc_roc"], "f1": tm["f1"], "recall": tm["recall"],
            "precision": tm["precision"], "acc": tm["accuracy"],
            "bal_acc": bm["balanced_accuracy"]}


def rfecv_features(estimator, X, y, groups) -> list:
    """Features elegidas por RFECV (GroupKFold, scoring F1). X ya estandarizada."""
    sel = RFECV(estimator, step=1, cv=GroupKFold(n_splits=N_SPLITS),
                scoring="f1", min_features_to_select=1, n_jobs=-1)
    sel.fit(X, y, groups=groups)
    return list(X.columns[sel.support_])


def main():
    df = load_ragtruth("train")
    X = extract_features(df)
    y = df["label"].values
    groups = df["context"].values
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)

    # Estimadores con importancias válidas para RFE (el ranking usa coef_/importances).
    ranked = {
        "logreg": LogisticRegression(max_iter=2000, random_state=SEED),
        "random_forest": build_random_forest(),
        "xgboost": build_xgboost(),
    }
    builders = {"logreg": build_logreg, "random_forest": build_random_forest,
                "xgboost": build_xgboost}

    for name, est in ranked.items():
        feats = rfecv_features(est, Xs, y, groups)
        configs = {
            f"RFECV ({len(feats)})": feats,
            "forward-7": FORWARD7,
            "18 (todas)": list(X.columns),
        }
        rows = []
        for cfg, cols in configs.items():
            m = oof(builders[name], X[cols], y, groups)
            rows.append({"conjunto": cfg, **{k: round(m[k], 3) for k in METRICS}})
        print(f"\n{'='*72}\n{name.upper()}\n{'='*72}")
        print(f"RFECV elige {len(feats)}: {feats}")
        no = [c for c in X.columns if c not in feats]
        print(f"RFECV descarta {len(no)}: {no}")
        print(pd.DataFrame(rows).set_index("conjunto").to_string())


if __name__ == "__main__":
    main()
