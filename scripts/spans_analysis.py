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
from ragcheck.plotting import savefig, set_style  # noqa: E402
from ragcheck.spans import parse_spans, span_distributions  # noqa: E402


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

    print("resumen POS:", spans["pos"].value_counts().head(5).to_dict())
    print("resumen entidad:", ents["entity"].value_counts().head(5).to_dict())
    print("resumen tipo:", spans["label_type"].value_counts().head(5).to_dict())


if __name__ == "__main__":
    main()
