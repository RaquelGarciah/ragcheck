"""Curva ROC del detector definitivo (top-7) en test: global y por tarea."""
import random, numpy as np
from ragcheck.config import SEED
random.seed(SEED); np.random.seed(SEED)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.models import build_xgboost
from ragcheck.plotting import savefig, set_style
set_style()

tr = load_ragtruth("train"); Xtr = extract_features(tr); ytr = tr["label"].values
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values
tk = te["task_type"].values
TOP7 = ["containment","task_QA","num_context","task_Summary","answer_len","jaccard","sent_cont_min"]
p = build_xgboost().fit(Xtr[TOP7], ytr).predict_proba(Xte[TOP7])[:,1]

fig, ax = plt.subplots(figsize=(5.5,5))
fpr,tpr,_ = roc_curve(yte,p); ax.plot(fpr,tpr,lw=2,color="black",
    label=f"GLOBAL (AUC={ev.discrimination(yte,p)['auc_roc']:.3f})")
for t,c in [("Data2txt","tab:blue"),("QA","tab:orange"),("Summary","tab:red")]:
    m=tk==t; f,tp,_=roc_curve(yte[m],p[m])
    ax.plot(f,tp,lw=1.3,color=c,label=f"{t} (AUC={ev.discrimination(yte[m],p[m])['auc_roc']:.3f})")
ax.plot([0,1],[0,1],"--",color="gray",lw=0.7)
ax.set_xlabel("FPR (1 − especificidad)"); ax.set_ylabel("TPR (sensibilidad)")
ax.set_title("Curva ROC — detector top-7 (test oficial)"); ax.legend(fontsize=8,loc="lower right")
ax.set_xlim(0,1); ax.set_ylim(0,1)
savefig(fig,"roc_top7")
print("Figura: roc_top7")
