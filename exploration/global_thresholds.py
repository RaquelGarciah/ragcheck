"""Modelo global top7/top10 con 3 umbrales (Youden/maxF1/break-even), todas las métricas."""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED
random.seed(SEED); np.random.seed(SEED)
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.models import build_xgboost
from ragcheck.training import cross_validate

tr = load_ragtruth("train"); Xtr = extract_features(tr); ytr = tr["label"].values; g = tr["source"].values
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values
TOP7  = ["containment","task_QA","num_context","task_Summary","answer_len","jaccard","sent_cont_min"]
TOP10 = TOP7 + ["novel_bigram","num_overlap","sent_sim_min"]
grid = np.linspace(0.05,0.95,181)

def thr(y,p,kind):
    if kind=="Youden": return ev.best_threshold(y,p)
    if kind=="max-F1": return grid[int(np.argmax([ev.threshold_metrics(y,p,t)["f1"] for t in grid]))]
    return grid[int(np.argmin([abs(ev.threshold_metrics(y,p,t)["precision"]-ev.threshold_metrics(y,p,t)["recall"]) for t in grid]))]

for name, cols in [("top7",TOP7),("top10",TOP10)]:
    p_oof = cross_validate(build_xgboost(), Xtr[cols], ytr, g)["y_prob"]
    p_te  = build_xgboost().fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:,1]
    auc=ev.discrimination(yte,p_te)["auc_roc"]; aucpr=ev.discrimination(yte,p_te)["auc_pr"]
    print(f"\n=== {name} GLOBAL — AUC {auc:.3f}, AUC-PR {aucpr:.3f} (test oficial) ===")
    rows=[]
    for rule in ["Youden","max-F1","break-even"]:
        t=thr(ytr,p_oof,rule); m=ev.threshold_metrics(yte,p_te,t); b=ev.balanced_metrics(yte,p_te,t)
        rows.append({"umbral":rule,"thr":round(t,3),"F1":round(m["f1"],3),"acc":round(m["accuracy"],3),
                     "recall":round(m["recall"],3),"precision":round(m["precision"],3),
                     "bal_acc":round(b["balanced_accuracy"],3)})
    print(pd.DataFrame(rows).to_string(index=False))
