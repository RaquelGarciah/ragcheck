"""Híbrido: modelo response-level 18 + estadísticos de la señal por frase (stacking OOF)."""
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
_SENT_RE = re.compile(r"[^.!?\n]+"); _TOK = re.compile(r"[a-z0-9]+")
FEATS = {"containment": containment, "jaccard": jaccard, "tfidf_cos": tfidf_cos,
         "num_overlap": num_overlap, "num_context": num_context,
         "novel_bigram": novel_bigram, "neg_diff": neg_diff, "answer_len": answer_len}
TASKS = ["Data2txt", "QA", "Summary"]


def _sent_off(t):
    return [(m.group(), m.start(), m.end()) for m in _SENT_RE.finditer(t) if _TOK.search(m.group().lower())]


df = load_ragtruth("train", keep_spans=True)
y = df["label"].values; tasks = df["task_type"].values
groups = df["source"].values

rows = []
for i, (resp, src, labels, task) in enumerate(
        zip(df["response"], df["source"], df["hallucination_labels"], df["task_type"])):
    spans = [(sp["start"], sp["end"]) for sp in parse_spans(labels)]
    for st, s0, s1 in _sent_off(resp):
        ys = int(any(s0 < h1 and h0 < s1 for h0, h1 in spans))
        f = {k: fn(st, src) for k, fn in FEATS.items()}
        for t in TASKS: f[f"task_{t}"] = float(task == t)
        rows.append({**f, "y": ys, "resp_id": i, "source": src})
S = pd.DataFrame(rows)
fc = list(FEATS) + [f"task_{t}" for t in TASKS]

# OOF por frase
p_sent = np.zeros(len(S)); gkf = GroupKFold(n_splits=N_SPLITS)
for tr, te in gkf.split(S[fc], S["y"].values, S["source"].values):
    m = clone(build_xgboost()).fit(S[fc].iloc[tr], S["y"].values[tr])
    p_sent[te] = m.predict_proba(S[fc].iloc[te])[:, 1]
S["p"] = p_sent

# estadísticos por respuesta (features nuevas)
g = S.groupby("resp_id")["p"]
stat = pd.DataFrame({"resp_id": g.max().index})
agg = pd.DataFrame({
    "sent_max": g.max(), "sent_mean": g.mean(), "sent_std": g.std().fillna(0),
    "sent_top2": g.apply(lambda s: np.mean(np.sort(s)[-2:])),
    "frac_high": g.apply(lambda s: float((s > 0.5).mean())),
}).reindex(range(len(df))).fillna(0)

Xr = extract_features(df)
Xh = pd.concat([Xr.reset_index(drop=True), agg.reset_index(drop=True)], axis=1)


def report(tag, p):
    thr = ev.best_threshold(y, p); tm = ev.threshold_metrics(y, p, thr)
    row = (f"{tag:<24} AUC={ev.discrimination(y,p)['auc_roc']:.3f} F1={tm['f1']:.3f} "
           f"recall={tm['recall']:.3f} prec={tm['precision']:.3f}")
    for tk in TASKS:
        mm = tasks == tk; rt = ev.threshold_metrics(y[mm], p[mm], thr)
        row += f"  {tk[:4]}(AUC={ev.discrimination(y[mm],p[mm])['auc_roc']:.3f},rec={rt['recall']:.2f})"
    print(row)


report("base 18", cross_validate(build_xgboost(), Xr, y, groups)["y_prob"])
report("hibrido 18+frase(5)", cross_validate(build_xgboost(), Xh, y, groups)["y_prob"])
m = clone(build_xgboost()).fit(Xh, y)
imp = pd.Series(m.feature_importances_, index=Xh.columns).sort_values(ascending=False)
print("\nTop-8 importancia (híbrido):"); print(imp.head(8).round(3).to_string())
