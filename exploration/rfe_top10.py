"""Replica exacta del RFE por importancia de la otra sesión + F1 del top-10.

Reproduce el algoritmo de feature_selection_wrappers.py (drop recursivo de la
feature con menor feature_importances_ de xgboost) y da el top-10, para comprobar
si coincide entre sesiones (mismo código + semilla 42 = mismo output). Reporta el
F1 (OOF GroupKFold, Youden) y en test del top-10 y del top-7.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_xgboost  # noqa: E402
from ragcheck.training import cross_validate  # noqa: E402

tr = load_ragtruth("train")
Xtr, ytr, g = extract_features(tr), tr["label"].values, tr["context"].values
te = load_ragtruth("test")
Xte, yte = extract_features(te), te["label"].values
cols_all = list(Xtr.columns)

# --- RFE por importancia (idéntico a la otra sesión) ---
cur, rfe_order = list(cols_all), []
while len(cur) > 1:
    m = build_xgboost().fit(Xtr[cur], ytr)
    imp = pd.Series(m.feature_importances_, index=cur)
    drop = imp.idxmin()
    rfe_order.append(drop)
    cur.remove(drop)

their_rank = list(reversed(rfe_order)) + cur      # expresión de su script
correct_rank = cur + list(reversed(rfe_order))    # superviviente (más imp.) primero

print("Superviviente (nunca eliminado):", cur)
print("Orden de eliminación (1º = menos importante):", rfe_order)
print("\n[SU expresión  reversed(rfe_order)+cur] top-10:", their_rank[:10])
print("[CORRECTO  cur+reversed(rfe_order)]      top-10:", correct_rank[:10])


def f1s(cols):
    p_oof = cross_validate(build_xgboost(), Xtr[cols], ytr, g)["y_prob"]
    thr = ev.best_threshold(ytr, p_oof)
    f1_oof = ev.threshold_metrics(ytr, p_oof, thr)["f1"]
    p_te = build_xgboost().fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:, 1]
    f1_te = ev.threshold_metrics(yte, p_te, thr)["f1"]
    return f1_oof, f1_te


print("\n=== F1 (OOF / test) por corte del ranking CORRECTO ===")
for k in (7, 10, 18):
    o, t = f1s(correct_rank[:k])
    print(f"  top-{k:<2} {correct_rank[:k]}\n        F1_OOF={o:.3f}  F1_test={t:.3f}")
