# ragcheck — detección de alucinaciones en RAG con ML clásico

Detector de alucinaciones para sistemas RAG basado en **Machine Learning clásico**
y features léxicas interpretables. Dada una respuesta de un LLM y la fuente
recuperada, estima la probabilidad de que la respuesta contenga información **no
respaldada** por la fuente, con técnicas explicables y deterministas, a un coste
mil veces menor que un evaluador tipo "LLM-as-a-judge".

> Trabajo Fin de Grado en Matemáticas y Ciencia de Datos (UCM). Autora: Raquel
> García Hernández. Tutor: Daniel Vélez Serrano.

## Hipótesis

Un clasificador clásico sobre features léxicas (solapamiento, longitudes,
números, TF-IDF, LSA) alcanza **AUC-ROC ≥ 0.80** sobre RAGTruth con evaluación
honesta: `GroupKFold` por documento, test de permutación e intervalos de
confianza.

## Uso rápido

```python
from ragcheck import score

score(response, source)   # -> probabilidad de alucinación en [0, 1]
```

```bash
ragcheck score --response "..." --source "..."
ragcheck evaluate --model logreg
```

## Instalación

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m spacy download en_core_web_sm   # descriptivo de spans (POS/NER)
```

## Estructura

- `src/ragcheck/` — la librería: datos, features, spans, modelos, entrenamiento,
  evaluación (métricas A–F), figuras e inferencia.
- `scripts/` — scripts atómicos: descriptivos (`visualization`, `spans_analysis`),
  un fichero por modelo (`log_reg`, `knn`, `svm`, `random_forest`, `xgb`) y la
  comparativa (`compare_models`).
- `tests/`, `outputs/` (informes, figuras, modelos), `data/` (cache RAGTruth).

Detalle completo y estándares en [CLAUDE.md](CLAUDE.md); alcance técnico en
[GUIA_TFG_ALUCINACIONES.md](GUIA_TFG_ALUCINACIONES.md).

## Resultados

Se completa al cerrar la Fase 0 (tabla comparativa de los cinco modelos con las
métricas A–F). Partida de referencia (prueba de viabilidad): ~0.78 AUC-ROC con
cinco features simples, hasta ~0.84 añadiendo TF-IDF del output.

## Reproducibilidad

Semilla 42 en todo; `GroupKFold` por documento; TF-IDF/SVD ajustados solo con
train; sin embeddings neuronales ni APIs de pago. Python 3.12.

## Licencia

MIT.
