"""RFECV completo: k óptimo, features seleccionadas, ranking y curva CV.

Borrador (NO citable como código; produce figura citable). RFECV (Guyon et al.,
2002) elimina recursivamente la feature menos importante y elige por validación
cruzada cuántas conservar. Protocolo honesto: GroupKFold por `source`, scoring AUC
(métrica interna, independiente del umbral). Solo modelos con importancias
(logreg estandarizado, random_forest, xgboost). Genera fsel_rfecv.png.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.feature_selection import RFECV  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ragcheck.config import N_SPLITS  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_random_forest, build_xgboost  # noqa: E402
from ragcheck.plotting import savefig, set_style  # noqa: E402


def main():
    df = load_ragtruth("train")
    X = extract_features(df)
    y = df["label"].values
    groups = df["source"].values
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)

    models = {
        "logreg": LogisticRegression(max_iter=2000, random_state=SEED),
        "random_forest": build_random_forest(),
        "xgboost": build_xgboost(),
    }

    set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, est in models.items():
        sel = RFECV(est, step=1, cv=GroupKFold(n_splits=N_SPLITS), scoring="accuracy",
                    min_features_to_select=1, n_jobs=-1)
        sel.fit(Xs, y, groups=groups)
        scores = sel.cv_results_["mean_test_score"]
        ks = np.arange(1, len(scores) + 1)
        k_opt = sel.n_features_
        # Orden por ranking_ (1 = superviviente = más importante).
        order = sorted(X.columns, key=lambda c: sel.ranking_[list(X.columns).index(c)])
        chosen = list(X.columns[sel.support_])

        ax.plot(ks, scores, marker="o", ms=3, label=f"{name} (k*={k_opt}, AUC={scores.max():.3f})")
        ax.axvline(k_opt, color=ax.lines[-1].get_color(), ls=":", lw=1, alpha=0.6)

        print(f"\n{'='*74}\n{name.upper()}  —  k óptimo = {k_opt}\n{'='*74}")
        print(f"Seleccionadas ({len(chosen)}): {chosen}")
        print(f"Descartadas ({18 - len(chosen)}): {[c for c in X.columns if c not in chosen]}")
        print("Ranking completo (más → menos importante):")
        print("  " + " > ".join(order))
        print("Curva CV (AUC) por k:")
        print("  " + ", ".join(f"{k}:{s:.3f}" for k, s in zip(ks, scores)))

    ax.set(xlabel="nº de features conservadas", ylabel="accuracy (CV, GroupKFold)",
           title="RFECV con scoring = accuracy")
    ax.set_xticks(np.arange(1, 19))
    ax.legend(fontsize=8, loc="lower right")
    savefig(fig, "fsel_rfecv_acc")
    print("\nFigura -> outputs/figures/fsel_rfecv_acc.png")


if __name__ == "__main__":
    main()
