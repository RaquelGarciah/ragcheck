"""Balancear a 50/50 vs mover umbral: ¿levanta la curva o solo desliza el punto?"""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED
random.seed(SEED); np.random.seed(SEED)
from sklearn.base import clone
from xgboost import XGBClassifier
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.config import XGBOOST_PARAMS
from ragcheck.training import cross_validate
pd.set_option("display.width", 200)

tr = load_ragtruth("train"); Xtr = extract_features(tr); ytr = tr["label"].values; g = tr["source"].values
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values
TOP7 = ["containment","task_QA","num_context","task_Summary","answer_len","jaccard","sent_cont_min"]
X7, Xt7 = Xtr[TOP7], Xte[TOP7]
neg, pos = int((ytr==0).sum()), int((ytr==1).sum()); spw = neg/pos

def base():  return XGBClassifier(**XGBOOST_PARAMS)
def weighted(): return XGBClassifier(**{**XGBOOST_PARAMS, "scale_pos_weight": spw})

def line(tag, model, X, y, thr=None):
    p_oof = cross_validate(model, X, y, g if len(y)==len(ytr) else np.arange(len(y)))["y_prob"] if thr is None else None
    m = clone(model).fit(X, y); p = m.predict_proba(Xt7)[:,1]
    t = thr if thr is not None else ev.best_threshold(y, p_oof)
    d = ev.discrimination(yte, p); tm = ev.threshold_metrics(yte, p, t)
    return {"config":tag,"AUC":round(d["auc_roc"],3),"AUC_PR":round(d["auc_pr"],3),"thr":round(t,3),
            "F1":round(tm["f1"],3),"recall":round(tm["recall"],3),"prec":round(tm["precision"],3)}, p

rows=[]
r,_ = line("base (natural, Youden)", base(), X7, ytr); rows.append(r)
r,pw = line("scale_pos_weight (~50/50) @0.5", weighted(), X7, ytr, thr=0.5); rows.append(r)
r,_ = line("scale_pos_weight, Youden", weighted(), X7, ytr); rows.append(r)
# undersample a 50/50
idx0=np.where(ytr==0)[0]; idx1=np.where(ytr==1)[0]
rng=np.random.default_rng(SEED); keep0=rng.choice(idx0,size=len(idx1),replace=False)
sub=np.concatenate([keep0,idx1]); Xs,ys=X7.iloc[sub],ytr[sub]
mu=clone(base()).fit(Xs,ys); pu=mu.predict_proba(Xt7)[:,1]
du=ev.discrimination(yte,pu); tu=ev.threshold_metrics(yte,pu,0.5)
rows.append({"config":"undersample 50/50 @0.5","AUC":round(du["auc_roc"],3),"AUC_PR":round(du["auc_pr"],3),
             "thr":0.5,"F1":round(tu["f1"],3),"recall":round(tu["recall"],3),"prec":round(tu["precision"],3)})

# DEMOSTRACIÓN: base con umbral que iguala el recall del balanceado
pb = clone(base()).fit(X7,ytr).predict_proba(Xt7)[:,1]
target_rec = rows[1]["recall"]  # recall del scale_pos_weight@0.5
grid=np.linspace(0.02,0.98,193)
tmatch=grid[int(np.argmin([abs(ev.threshold_metrics(yte,pb,t)["recall"]-target_rec) for t in grid]))]
mm=ev.threshold_metrics(yte,pb,tmatch)
rows.append({"config":f"base con umbral={tmatch:.2f} (=recall del balanceado)","AUC":round(ev.discrimination(yte,pb)["auc_roc"],3),
             "AUC_PR":round(ev.discrimination(yte,pb)["auc_pr"],3),"thr":round(tmatch,3),
             "F1":round(mm["f1"],3),"recall":round(mm["recall"],3),"prec":round(mm["precision"],3)})
print(f"train: {pos} pos / {neg} neg  (prev {pos/(pos+neg):.3f});  scale_pos_weight={spw:.2f}\n")
print(pd.DataFrame(rows).to_string(index=False))
