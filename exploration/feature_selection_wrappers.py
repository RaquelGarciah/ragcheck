"""Varios métodos de selección + test directo del one-hot de tarea."""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED, N_SPLITS
random.seed(SEED); np.random.seed(SEED)
from sklearn.base import clone
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.models import build_xgboost
from ragcheck.training import cross_validate
pd.set_option("display.width", 220)

tr = load_ragtruth("train"); Xtr = extract_features(tr); ytr = tr["label"].values; g = tr["source"].values
te = load_ragtruth("test");  Xte = extract_features(te); yte = te["label"].values
cols_all = list(Xtr.columns)

def oof_auc(cols):
    p = cross_validate(build_xgboost(), Xtr[cols], ytr, g)["y_prob"]
    return ev.discrimination(ytr, p)["auc_roc"]

def test_row(tag, cols):
    m = build_xgboost().fit(Xtr[cols], ytr); p = m.predict_proba(Xte[cols])[:,1]
    p_oof = cross_validate(build_xgboost(), Xtr[cols], ytr, g)["y_prob"]
    thr = ev.best_threshold(ytr, p_oof); tm = ev.threshold_metrics(yte, p, thr)
    print(f"{tag:<26} k={len(cols):<2} AUC={ev.discrimination(yte,p)['auc_roc']:.3f} F1={tm['f1']:.3f} acc={tm['accuracy']:.3f} rec={tm['recall']:.3f}")

# --- 1. RFE por importancia (backward, barato) ---
cur = list(cols_all); rfe_order = []
while len(cur) > 1:
    m = build_xgboost().fit(Xtr[cur], ytr)
    imp = pd.Series(m.feature_importances_, index=cur)
    drop = imp.idxmin(); rfe_order.append(drop); cur.remove(drop)
rfe_rank = list(reversed(rfe_order)) + cur  # de más a menos importante
print("RFE (importancia) ranking:", rfe_rank)

# --- 2. Backward elimination por AUC (wrapper) ---
cur = list(cols_all); back = []
while len(cur) > 1:
    best_auc, best_drop = -1, None
    for c in cur:
        a = oof_auc([x for x in cur if x != c])
        if a > best_auc: best_auc, best_drop = a, c
    back.append((len(cur)-1, best_drop, round(best_auc,3))); cur.remove(best_drop)
print("\nBackward elimination (AUC) — (quedan, se quita, AUC tras quitar):")
for n, d, a in back: print(f"  {n:>2}  quita {d:<16} AUC={a}")
back_survivors_7 = cur + [d for _,d,_ in back[::-1]][:6]  # los 7 últimos en caer

# --- 3. Test directo: ¿el one-hot suma? ---
seven = ["containment","answer_len","num_overlap","jaccard","neg_diff","num_context","sent_sim_min"]
task3 = ["task_QA","task_Summary","task_Data2txt"]
print("\n=== TEST OFICIAL — comparación de sets ===")
test_row("7 (perm top)", seven)
test_row("7 + one-hot tarea", seven + task3)
test_row("RFE top-7", rfe_rank[:7])
test_row("RFE top-10", rfe_rank[:10])
test_row("18 (todas)", cols_all)
