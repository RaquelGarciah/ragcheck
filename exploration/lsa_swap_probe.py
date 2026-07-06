"""¿Sustituir tfidf_cos por LSA mejora? Tres variantes, OOF GroupKFold, xgboost."""
import random, numpy as np, pandas as pd
from ragcheck.config import SEED, N_SPLITS
random.seed(SEED); np.random.seed(SEED)
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.models import build_xgboost

df = load_ragtruth("train")
X = extract_features(df); y = df["label"].values; groups = df["context"].values
tasks = df["task_type"].values
resp = list(df["output"]); src = list(df["context"])


def lsa_cos_perfold(tr, te):
    """Coseno LSA (TF-IDF+SVD ajustado SOLO con train) para todas las filas."""
    tfidf = TfidfVectorizer(max_features=20000)
    M = tfidf.fit_transform([src[i] for i in tr] + [resp[i] for i in tr])
    svd = TruncatedSVD(n_components=min(100, M.shape[1]-1), random_state=SEED).fit(M)
    def cos(idx):
        A = svd.transform(tfidf.transform([resp[i] for i in idx]))
        B = svd.transform(tfidf.transform([src[i] for i in idx]))
        num = (A*B).sum(1); den = np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)
        return np.divide(num, den, out=np.zeros_like(num), where=den>0)
    return cos


def oof(make_X):
    p = np.zeros(len(y)); gkf = GroupKFold(n_splits=N_SPLITS)
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = make_X(tr, te)
        m = clone(build_xgboost()).fit(Xtr, y[tr])
        p[te] = m.predict_proba(Xte)[:,1]
    return p


def rep(tag, p):
    row = f"{tag:<22}GLOBAL_AUC={ev.discrimination(y,p)['auc_roc']:.3f}"
    thr = ev.best_threshold(y,p); row += f"  F1={ev.threshold_metrics(y,p,thr)['f1']:.3f}"
    for tk in ["Data2txt","QA","Summary"]:
        mm = tasks==tk; row += f"  {tk}={ev.discrimination(y[mm],p[mm])['auc_roc']:.3f}"
    print(row)


# A: baseline (18, con tfidf_cos)
rep("A) 18 con tfidf_cos", oof(lambda tr,te: (X.iloc[tr], X.iloc[te])))
# C: quitar tfidf_cos (17)
Xnt = X.drop(columns=["tfidf_cos"])
rep("C) 17 sin tfidf_cos", oof(lambda tr,te: (Xnt.iloc[tr], Xnt.iloc[te])))
# B: sustituir tfidf_cos por lsa_cos (por fold)
def makeB(tr, te):
    cos = lsa_cos_perfold(tr, te)
    Xtr = Xnt.iloc[tr].assign(lsa_cos=cos(tr)); Xte = Xnt.iloc[te].assign(lsa_cos=cos(te))
    return Xtr, Xte
rep("B) 17 + lsa (swap)", oof(makeB))
