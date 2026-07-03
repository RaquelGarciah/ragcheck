"""A) Umbrales (acc + maxF1acc) top7/top10. B) Modelos por tarea, 5 clasificadores."""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED, N_SPLITS
random.seed(SEED); np.random.seed(SEED)
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.models import BUILDERS
from ragcheck.training import cross_validate
pd.set_option("display.width", 230)

tr = load_ragtruth("train"); Xtr = extract_features(tr); ytr = tr["label"].values; gtr = tr["source"].values
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values
tk_tr = tr["task_type"].values; tk_te = te["task_type"].values
TASKS = ["Data2txt","QA","Summary"]

TOP7  = ["containment","task_QA","num_context","task_Summary","answer_len","jaccard","sent_cont_min"]
TOP10 = TOP7 + ["novel_bigram","num_overlap","sent_sim_min"]
NOTASK = lambda s: [c for c in s if not c.startswith("task_")]
grid = np.linspace(0.05, 0.95, 181)

def m_at(y,p,t):
    d=ev.threshold_metrics(y,p,t); return d["f1"],d["accuracy"],d["precision"],d["recall"]
def thr(y,p,kind):
    if kind=="youden": return ev.best_threshold(y,p)
    f=lambda t: {"maxf1":ev.threshold_metrics(y,p,t)["f1"],
                 "maxacc":ev.threshold_metrics(y,p,t)["accuracy"],
                 "maxf1acc":0.5*(ev.threshold_metrics(y,p,t)["f1"]+ev.threshold_metrics(y,p,t)["accuracy"])}[kind]
    return grid[int(np.argmax([f(t) for t in grid]))]

# ---------- PARTE A: umbrales (xgboost global) ----------
print("="*70,"\nA) UMBRALES — modelo global xgboost (test oficial)\n",'='*70)
for name, cols in [("top7",TOP7),("top10",TOP10)]:
    p_oof = cross_validate(BUILDERS["xgboost"](), Xtr[cols], ytr, gtr)["y_prob"]
    p_te  = BUILDERS["xgboost"]().fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:,1]
    print(f"\n{name}  (AUC test {ev.discrimination(yte,p_te)['auc_roc']:.3f})")
    for k in ["youden","maxf1","maxacc","maxf1acc"]:
        t=thr(ytr,p_oof,k); f,a,pr,rc=m_at(yte,p_te,t)
        print(f"  {k:<9} thr={t:.3f}  F1={f:.3f}  acc={a:.3f}  prec={pr:.3f}  recall={rc:.3f}")

# ---------- PARTE B: modelos por tarea ----------
print("\n"+"="*70,"\nB) MODELOS POR TAREA (5 clasificadores, umbral maxF1acc por tarea)\n",'='*70)
for name, cols in [("top7",TOP7),("top10",TOP10)]:
    cc = NOTASK(cols)
    print(f"\n### {name}  (por tarea usa {len(cc)} features sin one-hot: {cc})")
    rows=[]
    for mdl in ["logreg","knn","svm","random_forest","xgboost"]:
        for tk in TASKS:
            itr = tk_tr==tk; ite = tk_te==tk
            p_oof = cross_validate(BUILDERS[mdl](), Xtr[cc][itr], ytr[itr], gtr[itr])["y_prob"]
            t = thr(ytr[itr], p_oof, "maxf1acc")
            p_te = BUILDERS[mdl]().fit(Xtr[cc][itr], ytr[itr]).predict_proba(Xte[cc][ite])[:,1]
            f,a,pr,rc = m_at(yte[ite], p_te, t)
            rows.append({"modelo":mdl,"tarea":tk,"AUC":round(ev.discrimination(yte[ite],p_te)["auc_roc"],3),
                         "F1":round(f,3),"acc":round(a,3),"prec":round(pr,3),"recall":round(rc,3)})
    print(pd.DataFrame(rows).pivot_table(index="modelo",columns="tarea",values=["AUC","F1","acc"]).round(3).to_string())
