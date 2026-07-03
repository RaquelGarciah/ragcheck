"""F1 a umbral global vs MEJOR F1 posible por generador (óptimo sobre su propio test)."""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED
random.seed(SEED); np.random.seed(SEED)
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.models import build_xgboost
from ragcheck.training import cross_validate
pd.set_option("display.width", 200)

tr = load_ragtruth("train"); Xtr = extract_features(tr); ytr = tr["label"].values; g = tr["source"].values
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values; gen = te["model"].values
TOP7 = ["containment","task_QA","num_context","task_Summary","answer_len","jaccard","sent_cont_min"]
grid = np.linspace(0.02,0.98,193)

p_oof = cross_validate(build_xgboost(), Xtr[TOP7], ytr, g)["y_prob"]
thr_g = ev.best_threshold(ytr, p_oof)
p_te = build_xgboost().fit(Xtr[TOP7], ytr).predict_proba(Xte[TOP7])[:,1]

rows=[]
for gm in sorted(set(gen)):
    m = gen==gm; yy, pp = yte[m], p_te[m]
    f1_global = ev.threshold_metrics(yy, pp, thr_g)["f1"]
    # mejor F1 posible: optimizando el umbral SOBRE su propio test (cota superior, hace trampa)
    best = max((ev.threshold_metrics(yy, pp, t)["f1"], t) for t in grid)
    rows.append({"generador":gm,"prev":round(float(yy.mean()),3),
                 "AUC":round(ev.discrimination(yy,pp)["auc_roc"],3),
                 "AUC_PR":round(ev.discrimination(yy,pp)["auc_pr"],3),
                 "F1@global":round(f1_global,3),
                 "F1_max_posible":round(best[0],3),"thr_optimo":round(best[1],3)})
print(f"umbral global (Youden) = {thr_g:.3f}\n")
print(pd.DataFrame(rows).sort_values("prev").to_string(index=False))
