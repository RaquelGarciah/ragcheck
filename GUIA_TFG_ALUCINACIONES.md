# Guía de arranque — TFG: Detección de alucinaciones en RAG con ML clásico

> Documento para empezar el proyecto **desde cero**. Resume el problema, los
> datos, las features, los modelos, la evaluación y el plan de trabajo, con los
> resultados ya obtenidos en la prueba de viabilidad como punto de partida.

---

## 1. Idea en una frase

Dada una **respuesta** generada por un LLM y la **fuente** (contexto recuperado
en un sistema RAG), predecir con un **clasificador clásico** si la respuesta
contiene una **alucinación** (información que NO está respaldada por la fuente).

**Pregunta de investigación**: ¿hasta qué punto features léxicas/estadísticas
clásicas, interpretables y deterministas, bastan para detectar alucinaciones en
salidas RAG reales, y cuál es su techo de rendimiento?

---

## 2. Restricciones (del tutor) — no negociables

- **ML clásico** como modelo de inferencia: regresión logística, árboles,
  ensembles (XGBoost), SVM, KNN. **NO** LLMs como clasificador.
- Un LLM **sí** puede usarse como *fuente de datos ya etiquetados* (es el caso:
  RAGTruth son salidas de LLM anotadas).
- **Sin** `torch`, `transformers` ni `sentence-transformers` (sin embeddings
  neuronales).
- **Determinismo total**: semilla 42 en numpy, scikit-learn y xgboost; mismos
  resultados en cada ejecución.
- Código en inglés; comentarios y docstrings en español.

---

## 3. Entorno

- **Python 3.12** (no usar 3.14: faltan wheels estables de xgboost/spacy).
- Crear venv e instalar dependencias:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm   # solo si se usan features de NER
```

- `requirements.txt` mínimo: `scikit-learn`, `xgboost`, `datasets`, `pandas`,
  `numpy`, `scipy`, `spacy`, `nltk`. (Para el notebook: `jupyter`, `nbconvert`,
  `ipykernel`.)

---

## 4. Datos

### Dataset
`wandb/RAGTruth-processed` (HuggingFace Hub) — salidas RAG **reales** de LLMs con
alucinaciones **anotadas a mano**. https://huggingface.co/datasets/wandb/RAGTruth-processed

- Tamaño: **15.090 train + 2.700 test**.
- Descarga (cachea sola, sin token):

```python
from datasets import load_dataset
ds = load_dataset("wandb/RAGTruth-processed", split="train")   # y split="test"
```

### Columnas relevantes
| Columna | Tipo | Contenido |
|---|---|---|
| `context` | str | la **fuente** (pasaje recuperado) |
| `output` | str | la **respuesta** del LLM |
| `hallucination_labels` | **str** | lista JSON de spans alucinados; `"[]"` = limpio |
| `task_type` | str | `QA`, `Summary`, `Data2txt` |
| `query`, `model`, `temperature`, `id` | — | metadatos |

### Balance de clases
- Global: **44,5 % alucinación** / 55,5 % limpio.
- Por tarea: `Data2txt` 69,4 % · `QA` 31,1 % · `Summary` 31,1 %.

### Preparación obligatoria (5 pasos)
1. **Etiqueta** — `hallucination_labels` es un *string*, hay que parsearlo:
   ```python
   import ast
   def has_hallu(x):
       if isinstance(x, str):
           try: x = ast.literal_eval(x)
           except Exception: return 1 if x.strip() not in ("", "[]") else 0
       return 1 if (x is not None and len(x) > 0) else 0
   df["label"] = df["hallucination_labels"].map(has_hallu)   # 1 = alucina, 0 = limpio
   ```
2. **Renombrar** para el pipeline: `output` → respuesta, `context` → fuente.
3. **Balancear** con semilla (o usar `class_weight`/`scale_pos_weight` + AUC-PR):
   ```python
   pos = df[df.label==1].sample(N, random_state=42)
   neg = df[df.label==0].sample(N, random_state=42)
   data = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
   ```
4. **Features** sobre `(output, context)` (ver §5). Solo requieren tokenizar.
5. **Particionar SIN fuga de documento**: la misma `context` aparece repetida
   (varias respuestas por pasaje). Usar `GroupKFold` agrupando por `context`, o
   el split oficial train/test. *Una partición aleatoria infla el resultado.*

No hace falta: limpiar HTML, stopwords, stemming ni traducir (todo es inglés).

---

## 5. Features (versión base: 5 simples y explicables)

Notación: $R$ = palabras únicas de la respuesta, $F$ = de la fuente.

| Feature | Fórmula | Qué mide |
|---|---|---|
| `containment` | $\frac{\lvert R\cap F\rvert}{\lvert R\rvert}$ | precisión léxica: % de la respuesta respaldado por la fuente (señal dominante) |
| `jaccard` | $\frac{\lvert R\cap F\rvert}{\lvert R\cup F\rvert}$ | solape global simétrico |
| `tfidf_cos` | $\frac{\langle u,v\rangle}{\lVert u\rVert\lVert v\rVert}$ | parecido ponderando palabras raras (resultó **redundante**) |
| `num_overlap` | $\frac{\lvert N_R\cap N_F\rvert}{\lvert N_R\rvert}$ | % de números de la respuesta presentes en la fuente (caza fechas/cifras inventadas) |
| `answer_len` | $\lvert\text{tokens}\rvert$ | longitud de la respuesta (más larga → más alucina) |

Cuatro son operaciones de conjuntos (triviales de explicar). `tfidf_cos` es la
única con álgebra lineal y **no aporta** sobre las demás → se puede quitar y
quedarse en **4 features**.

Implementación de referencia: `src/features.py` y el notebook (§9).

---

## 6. Modelos

Exactamente dos, con semilla 42:
- `LogisticRegression(max_iter=2000, random_state=42)` — interpretable
  (coeficientes = dirección y peso de cada feature).
- `XGBClassifier(random_state=42, eval_metric="logloss")` — defaults para la
  línea base; versión regularizada para mejorar
  (`n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
  colsample_bytree=0.8, reg_lambda=2`).

---

## 7. Evaluación

### Métricas (secciones A–F; contrato de `src/ragcheck/evaluate.py`)
- **A — Discriminación**: AUC-ROC (principal), AUC-PR (positiva = alucinación).
- **B — Al umbral óptimo**: precision, recall, F1, accuracy, specificity (umbral
  elegido en validación, nunca sobre test).
- **C — Robustas al desbalance**: balanced accuracy.
- **D — Calibración**: Brier, ECE y reliability diagram.
- **E — Matriz de confusión.**
- **F — Significancia estadística**: intervalos por bootstrap y test de DeLong
  para comparar AUCs entre modelos.

### Verificación de que el resultado es REAL (incluir en la memoria)
1. **GroupKFold por `context`** vs KFold aleatorio: si no baja → no hay fuga de
   documento. *(En viabilidad: 0.804 vs 0.805 → sin fuga.)*
2. **Test de permutación**: barajar etiquetas; el AUC debe caer a ~0.5.
   *(0.805 → 0.502, p≈0.03.)*
3. **Split de test oficial**: entrenar en train, evaluar en test. *(0.835.)*
4. Líneas base: azar (0.5), clase mayoritaria, una sola feature (`containment`).
5. Intervalos de confianza (bootstrap o test de DeLong para comparar modelos).

---

## 8. Resultados de partida (prueba de viabilidad)

Sobre RAGTruth, muestra balanceada, ML clásico:

| Configuración | AUC-ROC |
|---|---|
| 5 features simples (n=3000), CV 5-fold | ~0.78 |
| 5 features, GroupKFold (sin fuga) | ~0.78 |
| 5 features, test oficial | ~0.82 |
| + features de novedad + más datos (n=5000) | ~0.80 |
| + TF-IDF del output (XGBoost, n=5000) | ~0.84 |

Feature dominante: `containment` (AUC en solitario ~0.76). `tfidf_cos`
redundante. Para contexto: detectores especializados de consistencia factual
rondan 0.75–0.85 en estos benchmarks → **hay margen real de mejora**, que es lo
que justifica un TFG completo.

---

## 9. Estructura del repositorio (código)

El proyecto es la librería instalable `ragcheck` (`src-layout`). La estructura
completa y el contrato de trabajo están en `CLAUDE.md` §3. En resumen:

- `src/ragcheck/` — el motor: `data`, `spans`, `features`, `models`, `training`,
  `evaluate`, `plotting`, `inference`, `cli`.
- `scripts/` — scripts atómicos citables: los dos descriptivos
  (`visualization.py`, `spans_analysis.py`), un modelo por fichero (`log_reg`,
  `knn`, `svm`, `random_forest`, `xgb`) y `compare_models.py`.
- `exploration/` — borrador de pruebas y notebooks (no citable).
- `tests/`, `outputs/`, `data/`.

La prueba de viabilidad (§8) se hizo fuera de este repo; sus resultados son el
punto de partida a reproducir en la Fase 0, no código ya presente aquí.

---

## 10. Plan de trabajo para el TFG completo

**Fase 0 — Reproducir la base.** Pipeline limpio: descarga → preparación →
5 features → 2 modelos → métricas + verificación. Documentar la línea base ~0.78.

**Fase 1 — Análisis de datos.** Distribución por tarea (QA/Summary/Data2txt),
tipos de alucinación (conflicto evidente vs información sin base), longitudes,
ejemplos cualitativos de aciertos y errores.

**Fase 2 — Mejorar el AUC sin perder determinismo** (todo ML clásico):
1. **LSA / TruncatedSVD** sobre TF-IDF → features densas "semánticas" sin
   transformers; álgebra lineal pura ($A=U\Sigma V^\top$), determinista con
   `random_state=42`. *(Palanca recomendada nº1; ajustar SOLO en train.)*
2. Features de grounding más finas: alineación frase-respuesta ↔ mejor
   frase-fuente; chequeo tipado de números/fechas/entidades; n-gramas de carácter.
3. Más datos (hay 15k) + XGBoost regularizado.
4. Calibración (Platt/isotónica) y selección de umbral según coste de error.

**Fase 3 — Evaluación rigurosa.** GroupKFold, permutación, test oficial,
intervalos de confianza, ablaciones, calibración. Análisis por tipo de tarea.

**Fase 4 — Memoria.** Marco teórico (RAG, alucinaciones, TF-IDF, logística,
gradient boosting, ROC/AUC), metodología, resultados, discusión del techo y
limitaciones (qué señal NO está en el texto superficial), conclusiones.

---

## 11. Checklist de reproducibilidad

- [ ] Semilla 42 fijada en numpy, sklearn y xgboost.
- [ ] TF-IDF / SVD ajustados **solo con train** (nunca con test).
- [ ] Partición por documento (GroupKFold o split oficial).
- [ ] Resultados reportados con media ± desviación (CV), no de un único split.
- [ ] `requirements.txt` con versiones; entorno Python 3.12.
- [ ] Sin `torch`/`transformers`; sin LLM como inferencia.
```
