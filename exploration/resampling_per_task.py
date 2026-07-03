"""Balancear a 50/50 POR TAREA: ¿levanta la curva por tarea o desliza el punto?"""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED, XGBOOST_PARAMS
random.seed(SEED); np.random.seed(SEED)
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
pd.set_option("display.width", 210)

tr = load_ragtruth("train"); Xtr = extract_features(tr); ytr = tr["label"].values
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values
tk_tr = tr["task_type"].values; tk_te = te["task_type"].values; src = tr["source"].values
# por tarea el one-hot es constante -> features léxicas
COLS = ["containment","num_context","answer_len","jaccard","sent_cont_min"]
grid = np.linspace(0.02,0.98,193)

def oof_thr(X,y,g):
    p=np.zeros(len(y)); gkf=GroupKFold(n_splits=5)
    for a,b in gkf.split(X,y,g):
        m=XGBClassifier(**XGBOOST_PARAMS).fit(X.iloc[a],y[a]); p[b]=m.predict_proba(X.iloc[b])[:,1]
    return ev.best_threshold(y,p)

rows=[]
for tk in ["Data2txt","QA","Summary"]:
    itr=tk_tr==tk; ite=tk_te==tk
    Xa,ya,ga=Xtr[COLS][itr],ytr[itr],src[itr]; Xt,yt=Xte[COLS][ite],yte[ite]
    # base (natural)
    thr=oof_thr(Xa,ya,ga); pb=clone(XGBClassifier(**XGBOOST_PARAMS)).fit(Xa,ya).predict_proba(Xt)[:,1]
    d=ev.discrimination(yt,pb); m=ev.threshold_metrics(yt,pb,thr)
    rows.append({"tarea":tk,"config":"base (natural)","prev_tr":round(ya.mean(),2),"AUC":round(d["auc_roc"],3),
                 "AUC_PR":round(d["auc_pr"],3),"thr":round(thr,3),"F1":round(m["f1"],3),"recall":round(m["recall"],3),"prec":round(m["precision"],3)})
    # undersample 50/50
    i0=np.where(ya==0)[0]; i1=np.where(ya==1)[0]; rng=np.random.default_rng(SEED)
    n=min(len(i0),len(i1)); keep=np.concatenate([rng.choice(i0,n,False),rng.choice(i1,n,False)])
    pu=clone(XGBClassifier(**XGBOOST_PARAMS)).fit(Xa.iloc[keep],ya[keep]).predict_proba(Xt)[:,1]
    d=ev.discrimination(yt,pu); m=ev.threshold_metrics(yt,pu,0.5)
    rows.append({"tarea":tk,"config":"balanceada 50/50 @0.5","prev_tr":0.5,"AUC":round(d["auc_roc"],3),
                 "AUC_PR":round(d["auc_pr"],3),"thr":0.5,"F1":round(m["f1"],3),"recall":round(m["recall"],3),"prec":round(m["precision"],3)})
    # base con umbral que iguala recall del balanceado (equivalencia)
    tgt=m["recall"]; tm=grid[int(np.argmin([abs(ev.threshold_metrics(yt,pb,t)["recall"]-tgt) for t in grid]))]
    mm=ev.threshold_metrics(yt,pb,tm)
    rows.append({"tarea":tk,"config":f"base @umbral {tm:.2f} (=recall bal.)","prev_tr":round(ya.mean(),2),"AUC":round(ev.discrimination(yt,pb)["auc_roc"],3),
                 "AUC_PR":round(ev.discrimination(yt,pb)["auc_pr"],3),"thr":round(tm,3),"F1":round(mm["f1"],3),"recall":round(mm["recall"],3),"prec":round(mm["precision"],3)})
print(pd.DataFrame(rows).to_string(index=False))
