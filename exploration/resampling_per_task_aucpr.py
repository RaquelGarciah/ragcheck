"""¿Balancear por tarea LEVANTA la curva (AUC-PR)? base vs scale_pos_weight vs undersample(5 semillas)."""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED, XGBOOST_PARAMS
random.seed(SEED); np.random.seed(SEED)
from xgboost import XGBClassifier
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
pd.set_option("display.width", 200)

tr = load_ragtruth("train"); Xtr = extract_features(tr); ytr = tr["label"].values
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values
tk_tr = tr["task_type"].values; tk_te = te["task_type"].values
COLS = ["containment","num_context","answer_len","jaccard","sent_cont_min"]

def aucpr(y,p): return ev.discrimination(y,p)["auc_pr"]
rows=[]
for tk in ["Data2txt","QA","Summary"]:
    itr=tk_tr==tk; ite=tk_te==tk
    Xa,ya=Xtr[COLS][itr],ytr[itr]; Xt,yt=Xte[COLS][ite],yte[ite]
    # base
    pb=XGBClassifier(**XGBOOST_PARAMS).fit(Xa,ya).predict_proba(Xt)[:,1]
    # scale_pos_weight (sin tirar datos, determinista)
    spw=(ya==0).sum()/(ya==1).sum()
    pw=XGBClassifier(**{**XGBOOST_PARAMS,"scale_pos_weight":spw}).fit(Xa,ya).predict_proba(Xt)[:,1]
    # undersample 50/50, 5 semillas -> AUC-PR media±std
    i0,i1=np.where(ya==0)[0],np.where(ya==1)[0]; n=min(len(i0),len(i1)); us=[]
    for s in range(5):
        rng=np.random.default_rng(s); keep=np.concatenate([rng.choice(i0,n,False),rng.choice(i1,n,False)])
        us.append(aucpr(yt,XGBClassifier(**XGBOOST_PARAMS).fit(Xa.iloc[keep],ya[keep]).predict_proba(Xt)[:,1]))
    rows.append({"tarea":tk,"prev":round(ya.mean(),2),"AUC_PR base":round(aucpr(yt,pb),3),
                 "AUC_PR scale_pos_w":round(aucpr(yt,pw),3),
                 "AUC_PR undersample":f"{np.mean(us):.3f}±{np.std(us):.3f}"})
print(pd.DataFrame(rows).to_string(index=False))
