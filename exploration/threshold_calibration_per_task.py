"""Calibración de umbral POR TAREA con 3 reglas (Youden/maxF1/break-even)."""
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
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values
tk_tr = tr["task_type"].values; tk_te = te["task_type"].values
TASKS = ["Data2txt","QA","Summary"]
TOP7  = ["containment","task_QA","num_context","task_Summary","answer_len","jaccard","sent_cont_min"]
TOP10 = TOP7 + ["novel_bigram","num_overlap","sent_sim_min"]
grid = np.linspace(0.05,0.95,181)

def thr(y,p,kind):
    if kind=="youden": return ev.best_threshold(y,p)
    if kind=="maxf1":  return grid[int(np.argmax([ev.threshold_metrics(y,p,t)["f1"] for t in grid]))]
    # break-even: |precision-recall| mínimo
    return grid[int(np.argmin([abs(ev.threshold_metrics(y,p,t)["precision"]-ev.threshold_metrics(y,p,t)["recall"]) for t in grid]))]

for name, cols in [("top7",TOP7),("top10",TOP10)]:
    p_oof = cross_validate(build_xgboost(), Xtr[cols], ytr, g)["y_prob"]
    p_te  = build_xgboost().fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:,1]
    print(f"\n{'='*64}\n{name}  (AUC global {ev.discrimination(yte,p_te)['auc_roc']:.3f}) — umbral por tarea (test)\n{'='*64}")
    rows=[]
    for tk in TASKS:
        itr=tk_tr==tk; ite=tk_te==tk
        auct=ev.discrimination(yte[ite],p_te[ite])["auc_roc"]
        for rule in ["youden","maxf1","breakeven"]:
            t=thr(ytr[itr],p_oof[itr],rule); m=ev.threshold_metrics(yte[ite],p_te[ite],t)
            rows.append({"tarea":tk,"AUC":round(auct,3),"regla":rule,"thr":round(t,3),
                         "F1":round(m["f1"],3),"acc":round(m["accuracy"],3),
                         "recall":round(m["recall"],3),"prec":round(m["precision"],3)})
    # global agregado con umbral por tarea, por regla
    for rule in ["youden","maxf1","breakeven"]:
        yp=np.zeros(len(yte),int)
        for tk in TASKS:
            itr=tk_tr==tk; ite=tk_te==tk; t=thr(ytr[itr],p_oof[itr],rule); yp[ite]=(p_te[ite]>=t).astype(int)
        tp=((yte==1)&(yp==1)).sum();fp=((yte==0)&(yp==1)).sum();fn=((yte==1)&(yp==0)).sum();tn=((yte==0)&(yp==0)).sum()
        f1=2*tp/(2*tp+fp+fn); acc=(tp+tn)/len(yte); rec=tp/(tp+fn); prec=tp/(tp+fp)
        rows.append({"tarea":"GLOBAL","AUC":round(ev.discrimination(yte,p_te)["auc_roc"],3),"regla":rule,"thr":np.nan,
                     "F1":round(f1,3),"acc":round(acc,3),"recall":round(rec,3),"prec":round(prec,3)})
    print(pd.DataFrame(rows).to_string(index=False))
