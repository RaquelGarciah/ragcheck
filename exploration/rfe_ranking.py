"""Ranking RFE completo de las 18 features (orden de importancia).

Borrador (NO citable). RFE con n_features_to_select=1 elimina de una en una la menos
importante y deja un orden completo 1..18. Se computa para xgboost (modelo de
referencia) y random_forest, y se contrasta con la importancia por permutación.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

from sklearn.feature_selection import RFE  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_random_forest, build_xgboost  # noqa: E402

BUILDERS = {"xgboost": build_xgboost, "random_forest": build_random_forest}


def main():
    df = load_ragtruth("train")
    X = extract_features(df)
    y = df["label"].values
    groups = df["source"].values

    rank = {}
    for name, build in BUILDERS.items():
        sel = RFE(build(), n_features_to_select=1, step=1).fit(X, y)
        # ranking_ = 1 para la superviviente (más importante) .. 18 la 1a eliminada.
        order = sorted(X.columns, key=lambda c: sel.ranking_[list(X.columns).index(c)])
        rank[name] = {f: i + 1 for i, f in enumerate(order)}

    # Permutación honesta: OOF sobre un split GroupKFold (train interno / valid).
    gkf = GroupKFold(n_splits=5)
    tr, va = next(gkf.split(X, y, groups))
    m = build_xgboost().fit(X.iloc[tr], y[tr])
    perm = permutation_importance(m, X.iloc[va], y[va], scoring="roc_auc",
                                  n_repeats=10, random_state=SEED)
    perm_drop = dict(zip(X.columns, perm.importances_mean))

    order_xgb = sorted(X.columns, key=lambda c: rank["xgboost"][c])
    rows = []
    for f in order_xgb:
        rows.append({"feature": f,
                     "RFE xgboost": rank["xgboost"][f],
                     "RFE rf": rank["random_forest"][f],
                     "perm ΔAUC": round(perm_drop[f], 4)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
