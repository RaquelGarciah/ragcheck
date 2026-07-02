"""Gráficas de apoyo: redundancia (correlación) e importancia por permutación."""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED
random.seed(SEED); np.random.seed(SEED)
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.inspection import permutation_importance
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.inference import load_model
from ragcheck.plotting import savefig, set_style
set_style()

tr = load_ragtruth("train"); Xtr = extract_features(tr); ytr = tr["label"].values
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values

# --- 1. Heatmap de correlación (Spearman) entre features ---
corr = Xtr.corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
            cbar_kws={"shrink": 0.7}, ax=ax, xticklabels=True, yticklabels=True)
ax.set_title("Correlación de Spearman entre features (redundancia)")
plt.setp(ax.get_xticklabels(), fontsize=6); plt.setp(ax.get_yticklabels(), fontsize=6)
savefig(fig, "feat_correlacion")

# bloques de features muy correlacionadas (|rho|>0.8)
high = [(a, b, round(corr.loc[a, b], 2)) for i, a in enumerate(corr.columns)
        for b in corr.columns[i+1:] if abs(corr.loc[a, b]) > 0.8]
print("Pares con |Spearman| > 0.8 (redundantes):")
for a, b, r in sorted(high, key=lambda t: -abs(t[2])): print(f"  {r:>5}  {a} ~ {b}")

# --- 2. Importancia por permutación (barra, test) ---
mdl = load_model("xgboost")
r = permutation_importance(mdl, Xte, yte, scoring="roc_auc",
                           n_repeats=20, random_state=SEED, n_jobs=-1)
imp = pd.Series(r.importances_mean, index=Xtr.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
imp.plot.barh(ax=ax, xerr=pd.Series(r.importances_std, index=Xtr.columns)[imp.index],
              color="steelblue")
ax.set_xlabel("caída de AUC al permutar (test)")
ax.set_title("Importancia por permutación (xgboost)")
savefig(fig, "feat_importancia_perm")
print("\nFiguras: feat_correlacion, feat_importancia_perm")
