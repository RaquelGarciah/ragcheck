"""3 figuras por tarea (top-7, test): PR, matriz de confusión, recall vs umbral."""
import random, numpy as np
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
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values; tk = te["task_type"].values
TOP7 = ["containment","task_QA","num_context","task_Summary","answer_len","jaccard","sent_cont_min"]
thr = ev.best_threshold(ytr, cross_validate(build_xgboost(),Xtr[TOP7],ytr,g)["y_prob"])
p = build_xgboost().fit(Xtr[TOP7],ytr).predict_proba(Xte[TOP7])[:,1]
TASKS=[("Data2txt","tab:blue"),("QA","tab:orange"),("Summary","tab:red")]

# --- 1. PR por tarea ---
fig,ax=plt.subplots(figsize=(5.5,5))
for t,c in TASKS:
    m=tk==t; pr,rc,_=precision_recall_curve(yte[m],p[m])
    ax.plot(rc,pr,color=c,lw=1.5,label=f"{t} (AUC-PR={ev.discrimination(yte[m],p[m])['auc_pr']:.3f}, prev={yte[m].mean():.2f})")
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("Curva precisión-recall por tarea (top-7, test)")
ax.legend(fontsize=8,loc="upper right"); ax.set_xlim(0,1); ax.set_ylim(0,1)
savefig(fig,"pr_por_tarea")

# --- 2. Matriz de confusión por tarea (umbral Youden global) ---
pred=(p>=thr).astype(int)
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,(t,_) in zip(axes,TASKS):
    m=tk==t; y_,yp_=yte[m],pred[m]
    cm=np.array([[int(((y_==0)&(yp_==0)).sum()),int(((y_==0)&(yp_==1)).sum())],
                 [int(((y_==1)&(yp_==0)).sum()),int(((y_==1)&(yp_==1)).sum())]])
    im=ax.imshow(cm,cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=13,
                    color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set_xticks([0,1]); ax.set_xticklabels(["pred limpio","pred alucina"],fontsize=8)
    ax.set_yticks([0,1]); ax.set_yticklabels(["real limpio","real alucina"],fontsize=8)
    ax.set_title(f"{t}  (recall={cm[1,1]/(cm[1,0]+cm[1,1]):.2f})")
fig.suptitle(f"Matriz de confusión por tarea (top-7, umbral Youden {thr:.2f})")
savefig(fig,"confusion_por_tarea")

# --- 3. Recall vs umbral por tarea ---
grid=np.linspace(0,1,101)
fig,ax=plt.subplots(figsize=(6.5,4.5))
for t,c in TASKS:
    m=tk==t; rec=[ev.threshold_metrics(yte[m],p[m],u)["recall"] for u in grid]
    ax.plot(grid,rec,color=c,lw=1.5,label=t)
ax.axvline(thr,color="k",ls=":",lw=0.9,label=f"Youden global ({thr:.2f})")
ax.set_xlabel("umbral"); ax.set_ylabel("recall"); ax.set_title("Recall en función del umbral, por tarea (top-7, test)")
ax.legend(fontsize=8); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=0.3)
savefig(fig,"recall_vs_umbral_por_tarea")
print("Figuras: pr_por_tarea, confusion_por_tarea, recall_vs_umbral_por_tarea")
