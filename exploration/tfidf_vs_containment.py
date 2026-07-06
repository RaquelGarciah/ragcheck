"""tfidf_cos vs containment: por qué containment gana, y modelo tfidf + decorreladas."""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED
random.seed(SEED); np.random.seed(SEED)
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.models import build_xgboost
from ragcheck.training import cross_validate
pd.set_option("display.width", 200)

df = load_ragtruth("train")
X = extract_features(df); y = df["label"].values; groups = df["context"].values
tasks = df["task_type"].values


def auc_f1(cols, tag):
    p = cross_validate(build_xgboost(), X[cols], y, groups)["y_prob"]
    thr = ev.best_threshold(y, p)
    row = (f"{tag:<34} k={len(cols):<2} AUC={ev.discrimination(y,p)['auc_roc']:.3f} "
           f"F1={ev.threshold_metrics(y,p,thr)['f1']:.3f}")
    for tk in ["Data2txt", "QA", "Summary"]:
        m = tasks == tk
        row += f" {tk[:4]}={ev.discrimination(y[m],p[m])['auc_roc']:.3f}"
    print(row)


# 1. Correlación de todo con tfidf_cos
corr = X.corr(method="spearman")["tfidf_cos"].drop("tfidf_cos").sort_values(key=abs)
print("=== Correlación (Spearman) de cada feature con tfidf_cos ===")
print(corr.round(3).to_string())

# 2. AUC en solitario: containment vs tfidf_cos
print("\n=== Una sola feature ===")
auc_f1(["containment"], "solo containment")
auc_f1(["tfidf_cos"], "solo tfidf_cos")

# 3. Sets decorrelados respecto a tfidf_cos
uncorr_05 = ["tfidf_cos"] + list(corr[abs(corr) < 0.5].index)
uncorr_03 = ["tfidf_cos"] + list(corr[abs(corr) < 0.3].index)
print(f"\n|rho|<0.5 con tfidf → {len(uncorr_05)} feats: {uncorr_05}")
print(f"|rho|<0.3 con tfidf → {len(uncorr_03)} feats: {uncorr_03}")

print("\n=== Modelos (OOF, xgboost) ===")
auc_f1(["tfidf_cos"], "tfidf solo")
auc_f1(uncorr_03, "tfidf + decorreladas (<0.3)")
auc_f1(uncorr_05, "tfidf + decorreladas (<0.5)")
auc_f1(list(X.columns), "TODAS (18, referencia)")
# comparación: mismas pero ancladas en containment en vez de tfidf
c_un05 = ["containment"] + list(X.corr("spearman")["containment"].drop("containment")
                               [lambda s: abs(s) < 0.5].index)
auc_f1(c_un05, "containment + decorreladas (<0.5)")
