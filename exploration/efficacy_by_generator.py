"""Eficacia del detector (top-7, Youden global) por LLM generador de la respuesta."""
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
gen = te["model"].values
TOP7 = ["containment","task_QA","num_context","task_Summary","answer_len","jaccard","sent_cont_min"]

p_oof = cross_validate(build_xgboost(), Xtr[TOP7], ytr, g)["y_prob"]
thr = ev.best_threshold(ytr, p_oof)
p_te = build_xgboost().fit(Xtr[TOP7], ytr).predict_proba(Xte[TOP7])[:,1]
print(f"umbral Youden global = {thr:.3f}\n")

rows=[]
for gm in sorted(set(gen)):
    m = gen==gm; yy, pp = yte[m], p_te[m]
    d = ev.discrimination(yy, pp); tm = ev.threshold_metrics(yy, pp, thr)
    rows.append({"generador":gm,"n":int(m.sum()),"prev":round(float(yy.mean()),3),
                 "AUC":round(d["auc_roc"],3),"AUC_PR":round(d["auc_pr"],3),
                 "F1":round(tm["f1"],3),"acc":round(tm["accuracy"],3),
                 "recall":round(tm["recall"],3),"prec":round(tm["precision"],3)})
rows.append({"generador":"GLOBAL","n":len(yte),"prev":round(float(yte.mean()),3),
             "AUC":round(ev.discrimination(yte,p_te)["auc_roc"],3),"AUC_PR":round(ev.discrimination(yte,p_te)["auc_pr"],3),
             "F1":round(ev.threshold_metrics(yte,p_te,thr)["f1"],3),"acc":round(ev.threshold_metrics(yte,p_te,thr)["accuracy"],3),
             "recall":round(ev.threshold_metrics(yte,p_te,thr)["recall"],3),"prec":round(ev.threshold_metrics(yte,p_te,thr)["precision"],3)})
print(pd.DataFrame(rows).sort_values("AUC",ascending=False).to_string(index=False))
