"""Análisis descriptivo de los spans alucinados (Nivel 2).

Parsea los spans anotados y genera las figuras que caracterizan el fenómeno:
posición relativa en la respuesta, longitud del span, categoría gramatical
(POS) y tipo de entidad (NER), número de spans por respuesta, y comparación por
tarea y por tipo de alucinación. spacy se usa solo como herramienta de análisis.
Todas las figuras van a outputs/figures/.
"""

import random

import numpy as np

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.plotting import coma, eje_coma, savefig, set_style  # noqa: E402
from ragcheck.spans import parse_spans, span_distributions  # noqa: E402

TASKS = ("Data2txt", "QA", "Summary")

# Los cuatro tipos de RAGTruth, en español (dos ejes: evidente/sutil, sin base/conflicto).
TYPE_ES = {
    "Evident Baseless Info": "Evidente · sin base",
    "Evident Conflict": "Evidente · conflicto",
    "Subtle Baseless Info": "Sutil · sin base",
    "Subtle Conflict": "Sutil · conflicto",
}
TYPE_ORDER = list(TYPE_ES.values())

# Entidad NER de spaCy -> categoría de lo que inventa la alucinación.
ENT_CAT = {
    **{e: "Numérico" for e in ("CARDINAL", "MONEY", "QUANTITY", "ORDINAL", "PERCENT")},
    **{e: "Temporal" for e in ("DATE", "TIME")},
    **{e: "Nombres propios" for e in ("PERSON", "ORG", "GPE", "NORP", "FAC", "LOC",
                                      "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE")},
    "NONE": "Sin entidad",
}
CAT_ORDER = ["Numérico", "Temporal", "Nombres propios", "Sin entidad"]


def fig_tipos_por_tarea(spans: "pd.DataFrame") -> None:
    """Composición de los cuatro tipos de alucinación por tarea (barras apiladas)."""
    d = spans.copy()
    d["tipo"] = d["label_type"].map(TYPE_ES)
    frac = (d.groupby("task_type")["tipo"].value_counts(normalize=True)
            .unstack().reindex(index=list(TASKS), columns=TYPE_ORDER).fillna(0.0))
    cmap = sns.color_palette("colorblind", len(TYPE_ORDER))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    left = np.zeros(len(frac))
    for j, col in enumerate(TYPE_ORDER):
        ax.barh(frac.index, frac[col], left=left, color=cmap[j], label=col)
        for i, v in enumerate(frac[col].values):
            if v > 0.06:
                ax.text(left[i] + v / 2, i, coma(v * 100, 0) + "%",
                        va="center", ha="center", fontsize=8, color="white")
        left += frac[col].values
    ax.set(xlim=(0, 1), xlabel="fracción de tramos", ylabel="",
           title="Tipo de alucinación por tarea")
    eje_coma(ax, "x", 1)
    ax.legend(ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.06),
              frameon=False)
    savefig(fig, "desc_tipos_por_tarea")


def fig_anatomia_span(spans: "pd.DataFrame") -> None:
    """Anatomía del tramo alucinado: qué inventa (categoría) y su longitud en tokens."""
    d = spans.copy()
    d["cat"] = d["entity"].map(ENT_CAT).fillna("Sin entidad")
    frac = d["cat"].value_counts(normalize=True).reindex(CAT_ORDER).fillna(0.0)
    cb = sns.color_palette("colorblind")
    colors = {"Numérico": cb[0], "Temporal": cb[2], "Nombres propios": cb[1],
              "Sin entidad": cb[7]}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    orden = CAT_ORDER[::-1]
    axes[0].barh(orden, frac[orden].values, color=[colors[c] for c in orden])
    for i, v in enumerate(frac[orden].values):
        axes[0].text(v + 0.006, i, coma(v * 100, 1) + "%", va="center", fontsize=8)
    axes[0].set(xlim=(0, float(frac.max()) * 1.22), xlabel="fracción de tramos",
                title="(a) Qué inventa la alucinación")
    eje_coma(axes[0], "x", 1)
    sns.histplot(spans, x="length_tokens", bins=40, ax=axes[1], color=cb[0])
    axes[1].set_xlim(0, spans["length_tokens"].quantile(0.98))
    axes[1].set(xlabel="tokens", ylabel="tramos", title="(b) Longitud del tramo")
    fig.tight_layout()
    savefig(fig, "desc_anatomia_span")


def main() -> None:
    set_style()
    df = load_ragtruth("train", keep_spans=True)
    df["n_spans"] = df["hallucination_labels"].map(lambda x: len(parse_spans(x)))
    spans = span_distributions(df)
    print(f"{len(spans)} spans alucinados en {int((df['n_spans'] > 0).sum())} respuestas.")

    # 1. Posición relativa del span en la respuesta.
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(spans, x="position", bins=20, ax=ax)
    ax.set(title="¿Dónde ocurre la alucinación?", xlabel="posición relativa en la respuesta", ylabel="spans")
    savefig(fig, "spans_posicion")

    # 2. Longitud del span (tokens), esperable bimodal.
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(spans, x="length_tokens", bins=40, ax=ax)
    ax.set(title="Longitud del span alucinado", xlabel="tokens", ylabel="spans")
    ax.set_xlim(0, spans["length_tokens"].quantile(0.98))
    savefig(fig, "spans_longitud")

    # 3. Categoría gramatical dominante del span.
    fig, ax = plt.subplots(figsize=(6, 4))
    order = spans["pos"].value_counts().head(10).index
    sns.countplot(spans[spans["pos"].isin(order)], y="pos", order=order, ax=ax)
    ax.set(title="Categoría gramatical del span", xlabel="spans", ylabel="")
    savefig(fig, "spans_pos")

    # 4. Tipo de entidad (NER).
    ents = spans[spans["entity"] != "NONE"]
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ents["entity"].value_counts().head(10).index
    sns.countplot(ents[ents["entity"].isin(order)], y="entity", order=order, ax=ax)
    ax.set(title="Tipo de entidad inventada", xlabel="spans", ylabel="")
    savefig(fig, "spans_entidad")

    # 5. Número de spans por respuesta alucinada.
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(df[df["n_spans"] > 0], x="n_spans", bins=range(1, 12), ax=ax)
    ax.set(title="Alucinaciones por respuesta", xlabel="nº de spans", ylabel="respuestas")
    savefig(fig, "spans_por_respuesta")

    # 6. Longitud del span por tarea (¿Data2txt más gruesas?).
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(spans, x="task_type", y="length_tokens", ax=ax)
    ax.set(title="Tamaño del span por tarea", xlabel="", ylabel="tokens")
    ax.set_ylim(0, spans["length_tokens"].quantile(0.98))
    savefig(fig, "spans_longitud_por_tarea")

    # 7. Distribución del tipo de alucinación anotado.
    fig, ax = plt.subplots(figsize=(6, 4))
    order = spans["label_type"].value_counts().index
    sns.countplot(spans, y="label_type", order=order, ax=ax)
    ax.set(title="Tipo de alucinación", xlabel="spans", ylabel="")
    savefig(fig, "spans_tipo")

    # Figuras nuevas del descriptivo (§1.1.1): tipos por tarea y anatomía del tramo.
    fig_tipos_por_tarea(spans)
    fig_anatomia_span(spans)

    print("resumen POS:", spans["pos"].value_counts().head(5).to_dict())
    print("resumen entidad:", ents["entity"].value_counts().head(5).to_dict())
    print("resumen tipo:", spans["label_type"].value_counts().head(5).to_dict())

    # Cifras para la prosa de §1.1.1.
    cat = spans["entity"].map(ENT_CAT).fillna("Sin entidad")
    print("\n=== §1.1.1 categoría de lo inventado (fracción de tramos) ===")
    print((cat.value_counts(normalize=True).reindex(CAT_ORDER) * 100).round(1).to_dict())
    print("=== §1.1.1 tipos por tarea (fracción dentro de la tarea) ===")
    d = spans.copy(); d["tipo"] = d["label_type"].map(TYPE_ES)
    print((d.groupby("task_type")["tipo"].value_counts(normalize=True)
           .unstack().reindex(index=list(TASKS), columns=TYPE_ORDER) * 100).round(1))


if __name__ == "__main__":
    main()
