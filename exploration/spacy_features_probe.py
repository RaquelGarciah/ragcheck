"""Prototipo: ¿mueven las features de spaCy (dependencias/SVO/binding) a Summary?

Mide el aporte marginal por tarea (OOF GroupKFold, xgboost) de 3 features
estructurales sobre las 18 actuales. Solo si mueve Summary se integra.
"""
import random, time, numpy as np, pandas as pd
from ragcheck.config import SEED
random.seed(SEED); np.random.seed(SEED)
import spacy
from ragcheck import evaluate as ev
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.models import build_xgboost
from ragcheck.training import cross_validate

nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])


def _repr(doc):
    arcs = {(t.text.lower(), t.dep_, t.head.text.lower())
            for t in doc if not t.is_punct and not t.is_space}
    svo = set()
    for tok in doc:
        if tok.pos_ == "VERB":
            subs = [c.text.lower() for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
            objs = [c.text.lower() for c in tok.children if c.dep_ in ("dobj", "obj", "attr", "dative", "pobj")]
            for s in subs:
                for o in objs:
                    svo.add((s, tok.text.lower(), o))
    ents = list(doc.ents)
    entnum = set()
    for n in doc:
        if n.like_num:
            if ents:
                e = min(ents, key=lambda e: min(abs(n.i - e.start), abs(n.i - e.end)))
                entnum.add((e.text.lower(), n.text.lower()))
            else:
                entnum.add((n.head.text.lower(), n.text.lower()))
    return arcs, svo, entnum


def _feats(rr, sr):
    ra, rs, rn = rr; sa, ss, sn = sr
    dep = len(ra - sa) / len(ra) if ra else 0.0
    svo = len(rs - ss) / len(rs) if rs else 0.0
    binding = len(rn & sn) / len(rn) if rn else 1.0
    return dep, svo, binding


df = load_ragtruth("train")
y = df["label"].values; groups = df["source"].values; tasks = df["task_type"].values
X = extract_features(df)

t = time.time()
uniq = list(set(df["response"]) | set(df["source"]))
reprs = {}
for txt, doc in zip(uniq, nlp.pipe(uniq, batch_size=64)):
    reprs[txt] = _repr(doc)
print(f"parseados {len(uniq)} textos únicos en {time.time()-t:.0f}s")

dep, svo, binding = [], [], []
for r, s in zip(df["response"], df["source"]):
    d, v, b = _feats(reprs[r], reprs[s])
    dep.append(d); svo.append(v); binding.append(b)
Xp = X.assign(dep_novelty=dep, svo_novelty=svo, ent_num_binding=binding)


def per_task_auc(Xin, tag):
    p = cross_validate(build_xgboost(), Xin, y, groups)["y_prob"]
    row = f"{tag:<16}GLOBAL={ev.discrimination(y,p)['auc_roc']:.3f}"
    for tk in ["Data2txt", "QA", "Summary"]:
        m = tasks == tk
        row += f"  {tk}={ev.discrimination(y[m],p[m])['auc_roc']:.3f}"
    print(row)


print("AUC-ROC por tarea:")
per_task_auc(X, "18 features")
per_task_auc(Xp, "+ spaCy (3)")

# importancia de las nuevas
m = build_xgboost().fit(Xp, y)
imp = pd.Series(m.feature_importances_, index=Xp.columns).sort_values(ascending=False)
print("\nRanking de las nuevas:")
for f in ["dep_novelty", "svo_novelty", "ent_num_binding"]:
    print(f"  {f}: importancia {imp[f]:.4f}, puesto {list(imp.index).index(f)+1}/{len(imp)}")
