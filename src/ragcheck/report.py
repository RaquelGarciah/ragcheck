"""Informe por modelo: rejilla de hiperparámetros y métricas A–F.

Lo comparten los scripts atómicos de cada modelo para volcar un markdown
homogéneo en outputs/reports/<name>.md.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV

from ragcheck import evaluate as ev
from ragcheck.config import REPORTS_DIR
from ragcheck.training import top_configs


def model_report(name: str, gs: GridSearchCV, y: np.ndarray, y_prob: np.ndarray) -> dict:
    """Escribe outputs/reports/<name>.md y devuelve el resumen A–F del modelo.

    `gs` es el GridSearchCV ajustado; `y_prob` son las probabilidades fuera de
    muestra del modelo ganador (mismo protocolo GroupKFold).
    """
    s = ev.summary(y, y_prob)
    lo, hi = s["auc_ci"]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.md"
    lines = [
        f"# {name} — grid search y evaluación\n",
        f"- Mejor F1 (CV GroupKFold por `source`): **{gs.best_score_:.3f}**",
        f"- Mejores hiperparámetros: `{gs.best_params_}`\n",
        "## Rejilla de hiperparámetros (top-5 por F1)\n",
        top_configs(gs).round(4).to_markdown(index=False),
        "\n\n## Métricas del modelo ajustado (secciones A–F)\n",
        f"- **A** AUC-ROC {s['auc_roc']:.3f} (IC95% [{lo:.3f}, {hi:.3f}]) · AUC-PR {s['auc_pr']:.3f}",
        f"- **B** F1 {s['f1']:.3f} · precision {s['precision']:.3f} · recall {s['recall']:.3f} "
        f"· accuracy {s['accuracy']:.3f} · specificity {s['specificity']:.3f}",
        f"- **C** balanced accuracy {s['balanced_accuracy']:.3f}",
        f"- **D** Brier {s['brier']:.3f} · ECE {s['ece']:.3f}",
        f"- Umbral (Youden) {s['threshold']:.3f}",
    ]
    path.write_text("\n".join(lines))
    return s
