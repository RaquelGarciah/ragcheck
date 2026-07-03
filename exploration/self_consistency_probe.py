"""¿Aporta el 'máximo desacuerdo entre frases' (auto-consistencia)?"""
import random, re, numpy as np
from ragcheck.config import SEED
random.seed(SEED); np.random.seed(SEED)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features, _sentences
from ragcheck.models import build_xgboost
from ragcheck.training import cross_validate

def disagreement(response):
    """1 - mínima similitud TF-IDF entre pares de frases de la respuesta.
    Alto = hay una frase que se desmarca del resto (posible inserción)."""
    ss = _sentences(response)
    if len(ss) < 2: return 0.0
    try:
        V = TfidfVectorizer().fit_transform(ss)
    except ValueError:
        return 0.0
    sim = cosine_similarity(V); np.fill_diagonal(sim, 1.0)
    return float(1.0 - sim.min())

df = load_ragtruth("train")
X = extract_features(df); y = df["label"].values; groups = df["source"].values
tasks = df["task_type"].values
dis = np.array([disagreement(r) for r in df["response"]])

def rep(tag, Xin):
    p = cross_validate(build_xgboost(), Xin, y, groups)["y_prob"]
    thr = ev.best_threshold(y, p)
    row = f"{tag:<24} AUC={ev.discrimination(y,p)['auc_roc']:.3f} F1={ev.threshold_metrics(y,p,thr)['f1']:.3f}"
    for tk in ["Data2txt","QA","Summary"]:
        m = tasks==tk; row += f" {tk[:4]}={ev.discrimination(y[m],p[m])['auc_roc']:.3f}"
    print(row)

import pandas as pd
print("correlación disagreement con label:", round(float(np.corrcoef(dis, y)[0,1]), 3))
rep("solo disagreement", pd.DataFrame({"dis": dis}))
rep("18 base", X)
rep("18 + disagreement", X.assign(sent_disagree=dis))
