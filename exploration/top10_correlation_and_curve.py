"""Top-7 / top-10 (RFE), matriz de correlación del top-10, y métricas vs tamaño de subset."""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED
random.seed(SEED); np.random.seed(SEED)
import matplotlib.pyplot as plt, seaborn as sns
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.models import build_xgboost
from ragcheck.plotting import savefig, set_style
from ragcheck.training import cross_validate
set_style()

tr = load_ragtruth("train"); Xtr = extract_features(tr); ytr = tr["label"].values; g = tr["source"].values
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values
cols = list(Xtr.columns)

# --- RFE ranking (recursivo por importancia) ---
cur=list(cols); order=[]
while len(cur)>1:
    m=build_xgboost().fit(Xtr[cur],ytr)
    drop=pd.Series(m.feature_importances_,index=cur).idxmin(); order.append(drop); cur.remove(drop)
rank=list(reversed(order))+cur
top7, top10 = rank[:7], rank[:10]
print("TOP-7 :", top7)
print("TOP-10:", top10)

# --- Matriz de correlación (Spearman) del top-10 ---
corr = Xtr[top10].corr(method="spearman")
fig,ax=plt.subplots(figsize=(7.5,6.5))
sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,vmin=-1,vmax=1,square=True,
            annot_kws={"size":7},cbar_kws={"shrink":0.8},ax=ax)
ax.set_title("Correlación de Spearman — top-10 (RFE)")
plt.setp(ax.get_xticklabels(),fontsize=7,rotation=45,ha="right"); plt.setp(ax.get_yticklabels(),fontsize=7)
savefig(fig,"corr_top10")

# --- Métricas vs tamaño de subset (orden RFE, test, umbral Youden) ---
rows=[]
for k in range(1,len(rank)+1):
    c=rank[:k]
    p_oof=cross_validate(build_xgboost(),Xtr[c],ytr,g)["y_prob"]; thr=ev.best_threshold(ytr,p_oof)
    p=build_xgboost().fit(Xtr[c],ytr).predict_proba(Xte[c])[:,1]; m=ev.threshold_metrics(yte,p,thr)
    rows.append({"k":k,"AUC":ev.discrimination(yte,p)["auc_roc"],"acc":m["accuracy"],
                 "F1":m["f1"],"recall":m["recall"],"precision":m["precision"]})
d=pd.DataFrame(rows)
fig,ax=plt.subplots(figsize=(8,4.5))
for met,c in [("acc","tab:blue"),("F1","tab:orange"),("recall","tab:green"),("precision","tab:red"),("AUC","gray")]:
    ax.plot(d["k"],d[met],marker="o",ms=3,label=met,color=c,ls="--" if met=="AUC" else "-")
ax.axvline(7,color="k",ls=":",lw=0.8); ax.axvline(10,color="k",ls=":",lw=0.8)
ax.text(7,ax.get_ylim()[0],"7",ha="center",fontsize=8); ax.text(10,ax.get_ylim()[0],"10",ha="center",fontsize=8)
ax.set_xlabel("nº de features (orden RFE)"); ax.set_ylabel("métrica (test oficial)")
ax.set_title("Métricas vs tamaño del subset"); ax.set_xticks(range(1,19)); ax.legend(ncol=5,fontsize=8); ax.grid(alpha=0.3)
savefig(fig,"metrics_vs_k")
print("\n", d.round(3).to_string(index=False))
print("Figuras: corr_top10, metrics_vs_k")
