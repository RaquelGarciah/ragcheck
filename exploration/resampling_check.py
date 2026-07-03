"""¿Remuestrear para igualar alucinaciones por tarea mueve la frontera?

Borrador (NO citable). Verifica empíricamente si reequilibrar las clases (por
tarea, a 50/50) cambia la SEPARACIÓN del modelo (AUC-PR por tarea, invariante al
umbral) o solo desliza el punto de operación. Predicción de researcher: la
frontera no se mueve —remuestrear ≈ mover umbral / `scale_pos_weight`—.

Dos remuestreos sin dependencias externas, ajustados SOLO en train, por tarea:
  · oversample : replica positivos de cada tarea hasta igualar a los negativos.
  · undersample: submuestrea negativos de cada tarea hasta igualar a los positivos.
Evaluación en el test oficial intacto. Frontera = AUC-PR (Data2txt/QA/Summary).
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

from sklearn.metrics import average_precision_score  # noqa: E402

from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_xgboost  # noqa: E402

TASKS = ("Data2txt", "QA", "Summary")


def _resample_idx(y: np.ndarray, task: np.ndarray, mode: str, rng) -> np.ndarray:
    """Índices de train reequilibrados a 50/50 dentro de cada tarea."""
    keep = []
    for k in TASKS:
        pos = np.where((task == k) & (y == 1))[0]
        neg = np.where((task == k) & (y == 0))[0]
        if mode == "oversample":
            n = max(len(pos), len(neg))
            keep += [rng.choice(pos, n, replace=True), rng.choice(neg, n, replace=True)]
        else:  # undersample
            n = min(len(pos), len(neg))
            keep += [rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]
    return np.concatenate(keep)


def _pr_row(name, yte, task_te, pte) -> dict:
    row = {"estrategia": name, "GLOBAL": round(average_precision_score(yte, pte), 3)}
    for k in TASKS:
        m = task_te == k
        row[k] = round(average_precision_score(yte[m], pte[m]), 3)
    return row


def main() -> None:
    tr, te = load_ragtruth("train"), load_ragtruth("test")
    Xtr, Xte = extract_features(tr), extract_features(te)
    ytr, yte = tr["label"].values, te["label"].values
    task_te = te["task_type"].values
    task_tr = tr["task_type"].values
    rng = np.random.RandomState(SEED)

    rows = []
    # Base: sin remuestrear.
    pte = build_xgboost().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    rows.append(_pr_row("base (sin remuestreo)", yte, task_te, pte))

    for mode in ("oversample", "undersample"):
        idx = _resample_idx(ytr, task_tr, mode, rng)
        pte = build_xgboost().fit(Xtr.iloc[idx], ytr[idx]).predict_proba(Xte)[:, 1]
        rows.append(_pr_row(f"{mode} 50/50 por tarea", yte, task_te, pte))

    print("AUC-PR por tarea (frontera; invariante al umbral) — test oficial\n")
    print(pd.DataFrame(rows).set_index("estrategia").to_string())


if __name__ == "__main__":
    main()
