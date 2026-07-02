"""¿Cuántas features hacen falta? Podado por importancia, validado en test."""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED
random.seed(SEED); np.random.seed(SEED)
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.models import BUILDERS, build_xgboost
from ragcheck.plotting import savefig, set_style
from ragcheck.training import cross_validate
set_style()

tr = load_ragtruth("train"); Xtr = extract_features(tr); ytr = tr["label"].values
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values
groups = tr["source"].values

# Ranking de features por importancia de permutación (xgb, en train).
mdl = build_xgboost().fit(Xtr, ytr)
r = permutation_importance(mdl, Xtr, ytr, scoring="roc_auc", n_repeats=10,
                           random_state=SEED, n_jobs=-1)
rank = list(pd.Series(r.importances_mean, index=Xtr.columns)
            .sort_values(ascending=False).index)
print("Ranking (importancia perm., train):", rank)

KS = [1, 2, 3, 5, 7, 10, 14, 18]
MODELS = ["logreg", "random_forest", "xgboost"]
rows = []
for name in MODELS:
    for k in KS:
        cols = rank[:k]
        # umbral Youden fijado en OOF de train (honesto)
        p_oof = cross_validate(BUILDERS[name](), Xtr[cols], ytr, groups)["y_prob"]
        thr = ev.best_threshold(ytr, p_oof)
        m = BUILDERS[name]().fit(Xtr[cols], ytr)
        p = m.predict_proba(Xte[cols])[:, 1]
        tm = ev.threshold_metrics(yte, p, thr)
        rows.append({"modelo": name, "k": k, "AUC": ev.discrimination(yte, p)["auc_roc"],
                     "F1": tm["f1"], "acc": tm["accuracy"], "recall": tm["recall"],
                     "precision": tm["precision"]})
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))

# Figura: F1/AUC/acc vs k, un panel por modelo.
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, name in zip(axes, MODELS):
    d = res[res["modelo"] == name]
    for met in ["AUC", "F1", "acc"]:
        ax.plot(d["k"], d[met], marker="o", ms=4, label=met)
    ax.set_title(name); ax.set_xlabel("nº features (top-k por importancia)")
    ax.grid(True, alpha=0.3); ax.axvline(7, color="gray", ls=":", lw=0.8)
axes[0].set_ylabel("métrica (test oficial)"); axes[0].legend(fontsize=8)
fig.suptitle("Podado por importancia: rendimiento en test vs nº de features")
savefig(fig, "feat_podado_test")
print("Figura: feat_podado_test")

print(res.round(3).to_string(index=False))
