"""Opción 1: supervisión a nivel frase con los spans de RAGTruth (weak supervision).

Etiqueta cada frase por solape con un span alucinado, entrena un clasificador de
frases (GroupKFold por source) y agrega a respuesta por MAX. Compara con el
modelo response-level. Sin fuga: en inferencia las features de frase no usan spans.
"""
import random, re, numpy as np, pandas as pd
from ragcheck.config import SEED, N_SPLITS
random.seed(SEED); np.random.seed(SEED)
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import (extract_features, containment, jaccard, tfidf_cos,
                               num_overlap, num_context, novel_bigram, neg_diff, answer_len)
from ragcheck.models import build_xgboost
from ragcheck.spans import parse_spans
from ragcheck.training import cross_validate
pd.set_option("display.width", 200)

_SENT_RE = re.compile(r"[^.!?\n]+")
_TOK = re.compile(r"[a-z0-9]+")
FEATS = {"containment": containment, "jaccard": jaccard, "tfidf_cos": tfidf_cos,
         "num_overlap": num_overlap, "num_context": num_context,
         "novel_bigram": novel_bigram, "neg_diff": neg_diff, "answer_len": answer_len}
TASKS = ["Data2txt", "QA", "Summary"]


def _sent_offsets(text):
    return [(m.group(), m.start(), m.end()) for m in _SENT_RE.finditer(text)
            if _TOK.search(m.group().lower())]


def _overlaps(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


df = load_ragtruth("train", keep_spans=True)
y_resp = df["label"].values
tasks_resp = df["task_type"].values

# --- construir dataset de frases ---
rows = []
for i, (resp, src, labels, task) in enumerate(
        zip(df["response"], df["source"], df["hallucination_labels"], df["task_type"])):
    spans = [(sp["start"], sp["end"]) for sp in parse_spans(labels)]
    for stext, s0, s1 in _sent_offsets(resp):
        ys = int(any(_overlaps(s0, s1, h0, h1) for h0, h1 in spans))
        feats = {k: fn(stext, src) for k, fn in FEATS.items()}
        for t in TASKS:
            feats[f"task_{t}"] = float(task == t)
        rows.append({**feats, "y": ys, "resp_id": i, "source": df["source"].iloc[i], "task": task})
S = pd.DataFrame(rows)
print(f"frases: {len(S)}  ({S['y'].mean():.3f} alucinadas)  de {len(df)} respuestas")

featcols = list(FEATS) + [f"task_{t}" for t in TASKS]
Xs = S[featcols]; ys = S["y"].values; gs = S["source"].values

# --- OOF a nivel frase (GroupKFold por source) ---
p_sent = np.zeros(len(S)); gkf = GroupKFold(n_splits=N_SPLITS)
for tr, te in gkf.split(Xs, ys, gs):
    m = clone(build_xgboost()).fit(Xs.iloc[tr], ys[tr])
    p_sent[te] = m.predict_proba(Xs.iloc[te])[:, 1]
S["p"] = p_sent

# --- agregar a respuesta por MAX ---
agg = S.groupby("resp_id")["p"].max()
p_resp_ws = np.zeros(len(df))
p_resp_ws[agg.index.values] = agg.values

# --- baseline response-level (18 features, mismos folds) ---
Xr = extract_features(df)
p_resp_base = cross_validate(build_xgboost(), Xr, y_resp, df["source"].values)["y_prob"]


def report(tag, p):
    thr = ev.best_threshold(y_resp, p)
    tm = ev.threshold_metrics(y_resp, p, thr)
    row = (f"{tag:<22} AUC={ev.discrimination(y_resp,p)['auc_roc']:.3f} "
           f"F1={tm['f1']:.3f} recall={tm['recall']:.3f} prec={tm['precision']:.3f}")
    for tk in TASKS:
        mm = tasks_resp == tk
        rt = ev.threshold_metrics(y_resp[mm], p[mm], thr)
        row += f"  {tk[:4]}(AUC={ev.discrimination(y_resp[mm],p[mm])['auc_roc']:.3f},rec={rt['recall']:.2f})"
    print(row)


print("\n=== Respuesta-level (mismos folds, umbral Youden global) ===")
report("base 18 (resp-level)", p_resp_base)
report("frase->max (weak sup)", p_resp_ws)

# importancia frase
mi = clone(build_xgboost()).fit(Xs, ys)
imp = pd.Series(mi.feature_importances_, index=featcols).sort_values(ascending=False)
print("\nImportancia (clasificador de frases):")
print(imp.round(3).to_string())
