"""Curvas PR + puntos de operación (Youden/maxF1/breakeven) + global vs por-tarea."""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED
random.seed(SEED); np.random.seed(SEED)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.models import build_xgboost
from ragcheck.plotting import savefig, set_style
from ragcheck.training import cross_validate
set_style()

tr = load_ragtruth("train"); Xtr = extract_features(tr); ytr = tr["label"].values; g = tr["source"].values
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values; tk_te = te["task_type"].values

SETS = {
 "top7":  ["containment","task_QA","num_context","task_Summary","answer_len","jaccard","sent_cont_min"],
 "top10": ["containment","task_QA","num_context","task_Summary","answer_len","jaccard","sent_cont_min","novel_bigram","num_overlap","sent_sim_min"],
}
grid = np.linspace(0.05, 0.95, 181)

def thr_youden(y,p): return ev.best_threshold(y,p)
def thr_maxf1(y,p):  return grid[int(np.argmax([ev.threshold_metrics(y,p,t)["f1"] for t in grid]))]
def thr_breakeven(y,p):
    return grid[int(np.argmin([abs(ev.threshold_metrics(y,p,t)["precision"]-ev.threshold_metrics(y,p,t)["recall"]) for t in grid]))]

def f1p(y,p,t):
    m=ev.threshold_metrics(y,p,t); return m["f1"],m["precision"],m["recall"]

for name, cols in SETS.items():
    p_oof = cross_validate(build_xgboost(), Xtr[cols], ytr, g)["y_prob"]
    p_te  = build_xgboost().fit(Xtr[cols], ytr).predict_proba(Xte[cols])[:,1]
    ty, tf, tb = thr_youden(ytr,p_oof), thr_maxf1(ytr,p_oof), thr_breakeven(ytr,p_oof)
    print(f"\n=== {name}  (AUC test {ev.discrimination(yte,p_te)['auc_roc']:.3f}, AUC-PR {ev.discrimination(yte,p_te)['auc_pr']:.3f}) ===")
    for tag,t in [("Youden",ty),("max-F1",tf),("break-even",tb)]:
        f,pr,rc=f1p(yte,p_te,t); print(f"  {tag:<11} thr={t:.3f}  F1={f:.3f} prec={pr:.3f} recall={rc:.3f}")
    # por tarea (max-F1 por tarea en train, aplicado a test)
    yp=np.zeros(len(yte),int)
    for t_ in np.unique(tk_te):
        mtr=tr["task_type"].values==t_; th=thr_maxf1(ytr[mtr],p_oof[mtr]); mte=tk_te==t_
        yp[mte]=(p_te[mte]>=th).astype(int)
    tp=((yte==1)&(yp==1)).sum();fp=((yte==0)&(yp==1)).sum();fn=((yte==1)&(yp==0)).sum()
    print(f"  por-tarea   F1={2*tp/(2*tp+fp+fn):.3f}  (vs global max-F1 {f1p(yte,p_te,tf)[0]:.3f})")
    # figura PR
    pr,rc,_=precision_recall_curve(yte,p_te)
    fig,ax=plt.subplots(figsize=(5,4.5)); ax.plot(rc,pr,lw=1.5,color="steelblue")
    for tag,t,c in [("Youden",ty,"tab:red"),("max-F1",tf,"tab:green"),("break-even",tb,"tab:orange")]:
        f,P,R=f1p(yte,p_te,t); ax.plot(R,P,"o",color=c,ms=7,label=f"{tag} (F1={f:.2f})")
    ax.plot([0,1],[0,1],"--",color="gray",lw=0.6)  # línea P=R
    ax.set_xlabel("recall");ax.set_ylabel("precision");ax.set_title(f"Curva PR — {name} (test)")
    ax.legend(fontsize=8,loc="lower left");ax.set_xlim(0,1);ax.set_ylim(0,1)
    savefig(fig,f"pr_{name}")
print("\nFiguras: pr_top7, pr_top10")
