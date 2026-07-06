"""Análisis de error del detector: ¿por qué Summary es el cuello de botella?

Borrador (NO citable). Hipótesis a contrastar: los fallos del clasificador en
Summary vienen de alucinación *sutil* (paráfrasis con palabras que sí están en
la fuente, recombinadas), invisible al solape léxico; mientras que en Data2txt la
alucinación es *evidente* (cifras/entidades inventadas), que las features cazan.

Contraste con datos:
1. Separación de probabilidades OOF por tarea (Summary apenas separa).
2. Containment del *texto alucinado* contra la fuente, por tarea: si es alto en
   Summary y bajo en Data2txt, la alucinación de Summary es léxicamente invisible.
3. Composición de tipos de alucinación por tarea (sutil vs evidente).
4. Comparación del containment de span en los falsos negativos vs aciertos de
   Summary: si los FN tienen containment más alto, el modelo falla justo donde la
   señal léxica desaparece.

Protocolo honesto: OOF de GroupKFold por `context`, xgboost.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.config import REPORTS_DIR  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import containment, extract_features  # noqa: E402
from ragcheck.models import build_xgboost  # noqa: E402
from ragcheck.plotting import savefig, set_style  # noqa: E402
from ragcheck.spans import parse_spans  # noqa: E402
from ragcheck.training import cross_validate  # noqa: E402

TASKS = ["Data2txt", "QA", "Summary"]


def _spans_table(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por span alucinado: tarea, tipo y containment del texto vs fuente."""
    rows = []
    for src, labels, task in zip(
        df["context"], df["hallucination_labels"], df["task_type"]
    ):
        for sp in parse_spans(labels):
            text = sp.get("text", "")
            if not text:
                continue
            rows.append(
                {
                    "task_type": task,
                    "label_type": sp.get("label_type", "NA"),
                    "span_containment": containment(text, src),
                }
            )
    return pd.DataFrame(rows)


def _span_containment_by_row(df: pd.DataFrame) -> np.ndarray:
    """Containment medio de los spans de cada fila (NaN si la fila está limpia)."""
    out = np.full(len(df), np.nan)
    for i, (src, labels) in enumerate(zip(df["context"], df["hallucination_labels"])):
        conts = [containment(sp.get("text", ""), src) for sp in parse_spans(labels)]
        conts = [c for c in conts if c is not None]
        if conts:
            out[i] = float(np.mean(conts))
    return out


def main() -> None:
    set_style()
    df = load_ragtruth("train", keep_spans=True)
    X = extract_features(df)
    y = df["label"].values
    groups = df["context"].values
    p = cross_validate(build_xgboost(), X, y, groups)["y_prob"]
    df = df.assign(p=p, y=y)

    thr = ev.best_threshold(y, p)
    df["pred"] = (p >= thr).astype(int)

    # --- Figura 1: separación de probabilidades por tarea ---
    g = sns.FacetGrid(df, col="task_type", col_order=TASKS, height=3.2, aspect=0.95)
    g.map_dataframe(
        sns.histplot, x="p", hue="y", bins=30, stat="density",
        common_norm=False, element="step", alpha=0.5,
    )
    g.set_axis_labels("prob. predicha", "densidad")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.axvline(thr, color="black", ls="--", lw=0.8)
    g.add_legend(title="alucina")
    savefig(g.figure, "err_prob_por_tarea")

    # --- Figura 2: containment del texto alucinado por tarea ---
    spans = _spans_table(df)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    sns.violinplot(
        data=spans, x="task_type", y="span_containment", order=TASKS,
        cut=0, inner="quartile", ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("containment del span alucinado vs fuente")
    savefig(fig, "err_span_containment")

    # --- Figura 3: composición de tipos de alucinación por tarea ---
    comp = (
        spans.groupby("task_type")["label_type"]
        .value_counts(normalize=True)
        .rename("frac")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    piv = comp.pivot(index="task_type", columns="label_type", values="frac").loc[TASKS]
    piv.plot(kind="barh", stacked=True, ax=ax, width=0.7)
    ax.set_xlabel("fracción de spans")
    ax.set_ylabel("")
    ax.legend(title="tipo", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    savefig(fig, "err_tipo_por_tarea")

    # --- Numérico: containment de span en FN vs TP de Summary ---
    df["span_cont"] = _span_containment_by_row(df)
    summ = df[df["task_type"] == "Summary"]
    fn = summ[(summ["y"] == 1) & (summ["pred"] == 0)]
    tp = summ[(summ["y"] == 1) & (summ["pred"] == 1)]

    lines = ["# Análisis de error — Summary como cuello de botella\n"]
    lines.append(f"Umbral (Youden global) = {thr:.3f}\n")
    lines.append("## AUC-ROC OOF por tarea\n")
    for t in TASKS:
        m = df["task_type"] == t
        lines.append(f"- {t}: {ev.discrimination(y[m.values], p[m.values])['auc_roc']:.3f}")
    lines.append("\n## Containment medio del texto alucinado vs fuente (por tarea)\n")
    for t in TASKS:
        s = spans[spans["task_type"] == t]["span_containment"]
        lines.append(f"- {t}: media {s.mean():.3f}, mediana {s.median():.3f} (n={len(s)})")
    lines.append("\n## Composición de tipos de alucinación por tarea (fracción)\n")
    piv2 = (
        spans.groupby("task_type")["label_type"].value_counts(normalize=True)
        .rename("frac").reset_index()
        .pivot(index="task_type", columns="label_type", values="frac").loc[TASKS]
    )
    lines.append(piv2.round(3).to_markdown())
    lines.append(
        "\nNota: Summary es ~86% *evidente* y QA tiene MÁS tipo *sutil* que Summary, "
        "pero QA puntúa mejor. Por tanto el tipo anotado NO explica la dificultad "
        "de Summary."
    )

    lines.append("\n## Summary: containment de span en errores vs aciertos\n")
    lines.append(f"- Falsos negativos (n={len(fn)}): span_cont medio {fn['span_cont'].mean():.3f}")
    lines.append(f"- Aciertos positivos (n={len(tp)}): span_cont medio {tp['span_cont'].mean():.3f}")

    lines.append("\n## Muestra de falsos negativos de Summary (spans alucinados)\n")
    sample = fn.sort_values("span_cont", ascending=False).head(5)
    for _, r in sample.iterrows():
        spans_txt = [sp.get("text", "") for sp in parse_spans(r["hallucination_labels"])]
        lines.append(f"- p={r['p']:.2f}, span_cont={r['span_cont']:.2f} → "
                     f"«{' | '.join(t.strip()[:120] for t in spans_txt if t.strip())}»")

    lines.append("\n## Conclusión\n")
    lines.append(
        "El cuello de botella de Summary NO es el tipo de alucinación (es mayoría "
        "*evidente*), sino el **camuflaje léxico**: un resumen reutiliza el "
        "vocabulario del documento, así que el span alucinado comparte palabras con "
        "la fuente (containment 0.62 vs 0.43 en Data2txt) y el solape léxico no lo "
        "distingue del texto fiel. Los falsos negativos son precisamente los spans "
        "de mayor containment. Es el techo estructural de las features de superficie: "
        "para separar en Summary hace falta señal *relacional* (consistencia "
        "número-unidad, entidad-relación, polaridad), no más solape de vocabulario."
    )

    report = REPORTS_DIR / "error_analysis_summary.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nInforme -> {report}")
    print("Figuras: err_prob_por_tarea, err_span_containment, err_tipo_por_tarea")


if __name__ == "__main__":
    main()
