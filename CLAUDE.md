# TFG — Detección de alucinaciones en RAG con ML clásico

Este fichero es la referencia única para trabajar en el repositorio. Léelo
entero antes de tocar nada. La especificación del problema (dataset, features,
modelos, evaluación, resultados de la prueba de viabilidad) está en
`GUIA_TFG_ALUCINACIONES.md`; este `CLAUDE.md` es el contrato de trabajo:
estructura del código, estándares, flujo de git, cómo colaborar con Claude Code.

---

## 1. Contexto del proyecto

Trabajo Fin de Grado en Matemáticas y Ciencia de Datos, Universidad Complutense
de Madrid. **Autora: Raquel García Hernández.** Tutor: **Daniel Vélez Serrano** (perfil
de series temporales y ML aplicado).

**Objetivo del TFG.** Construir un detector de alucinaciones en sistemas RAG
usando Machine Learning clásico (regresión logística, XGBoost, ensembles,etc), features léxicas y estadísticas puramente clásicas, y evaluación rigurosa.
Reproducir con técnicas explicables y baratas lo que hoy hacen los evaluadores tipo "LLM-as-a-judge" (GPT-4 como juez), demostrando que un clasificador clásico alcanza concordancia razonable a coste mil veces menor.

**Hipótesis falsable.** Un clasificador clásico sobre features léxicas
(solapamiento, longitudes, números, TF-IDF, LSA) alcanza AUC-ROC ≥ 0.80 sobre
RAGTruth con evaluación honesta (`GroupKFold` por documento, test de
permutación, intervalos de confianza).

**Modelo pedagógico del documento final.** Molde de los TFG ejemplares de
Sergio Gil Gavela (`../tfg-tesis/normativa_tfg/ejemplos_tfg_ejemplares/`):
capítulos claros, matemática rigurosa, ventajas/desventajas explícitas por
modelo, un capítulo de manual del entregable, conclusiones objetivo por objetivo.
La memoria se redacta en el repo hermano `../tfg-tesis/` (código primero,
redacción después).

Ver `GUIA_TFG_ALUCINACIONES.md` para el resto del alcance técnico.

---

## 2. Restricciones no negociables

1. **ML clásico como modelo de inferencia**: regresión logística, árboles,
   ensembles (XGBoost, Random Forest), SVM, KNN. **Nunca** un LLM ni redes
   neuronales profundas como clasificador.
2. **Los LLMs solo aparecen como fuente de datos**: RAGTruth son salidas de
   LLM ya anotadas por humanos. En este repo no se llama a ninguna API de LLM.
3. **Sin `torch`, `transformers` ni `sentence-transformers`**. Nada de
   embeddings neuronales. La representación semántica se hace con TF-IDF y
   LSA (TruncatedSVD sobre TF-IDF).
4. **Determinismo total**. Semilla 42 en `numpy`, `sklearn`, `xgboost`,
   `random`. Mismo input, mismo output, siempre.
5. **Python 3.12**. No usar 3.13 ni 3.14 (faltan wheels estables de xgboost y
   spacy).
6. **Código en inglés, comentarios y docstrings en español.**
7. **Sin APIs de pago.** Sin cuentas OpenAI/Anthropic/Google. Todo abierto y
   reproducible en local.

---

## 3. Estructura del repositorio

Dos repositorios hermanos, en la misma carpeta padre:
- **`ALUCINACIONES_RAG/`** (este, PÚBLICO) — el código: la librería `ragcheck`.
- **`../tfg-tesis/`** (PRIVADO) — la redacción de la memoria (kit + LaTeX en
  `tesis/`). Es la fuente que consume las cifras y figuras de `outputs/`.

El entregable es una **librería instalable** (`pip install -e .`) con `src-layout`:
un motor reutilizable y testeado (`src/ragcheck/`) y scripts finales atómicos
citables en la memoria (`scripts/`), separados del borrador de pruebas
(`exploration/`).

```
ALUCINACIONES_RAG/
├── pyproject.toml              paquete `ragcheck` (entry point CLI `ragcheck`)
├── src/ragcheck/               LA LIBRERÍA (motor reutilizable y testeado)
│   ├── config.py               SEED=42, rutas, hiperparámetros de los 5 modelos
│   ├── data.py                 load_ragtruth + parseo de hallucination_labels
│   ├── spans.py                parseo de spans + distribuciones descriptivas
│   ├── features.py             containment, jaccard, tfidf_cos, num_overlap, answer_len
│   ├── models.py               build_logreg/xgboost/random_forest/svm/knn
│   ├── training.py             CV GroupKFold por `source`, fit, persistencia
│   ├── evaluate.py             métricas por secciones A–F (números, sin plotting)
│   ├── plotting.py             tema seaborn común + helpers de figuras
│   ├── inference.py            score(response, source) -> prob; carga el modelo
│   └── cli.py                  `ragcheck train|evaluate|score`
├── scripts/                    SCRIPTS ATÓMICOS CITABLES (finos, importan ragcheck)
│   ├── visualization.py        descriptivo del dataset (Nivel 1, seaborn)
│   ├── spans_analysis.py       descriptivo de spans alucinados (Nivel 2, seaborn)
│   ├── log_reg.py knn.py svm.py random_forest.py xgb.py   un modelo por fichero
│   └── compare_models.py       tabla comparativa de los 5 (línea base)
├── exploration/                borrador: pruebas y notebooks (NO citable)
├── tests/                      test_features, test_data, test_spans,
│                               test_no_leakage (bloqueante en CI), test_determinism
├── outputs/{reports,figures,models}   informes, figuras y modelos entrenados
└── data/                       cache de RAGTruth (gitignored)
```

**Principio.** Los scripts son finos (importan de `ragcheck`, hacen una cosa,
guardan informe+figura). La lógica reutilizable y testeada vive en la librería:
los números salen de la librería, las figuras seaborn se pintan en los scripts.
Cada script es atómico y citable, sin duplicar código. `scripts/xgb.py` se llama
así (no `xgboost.py`) para no ensombrecer al paquete `xgboost`.

---

## 4. Estándares de código

Estos son los mismos que aplicarías en cualquier proyecto matemático limpio.

- **Python 3.12+**. Type hints en toda función pública.
- **Nombres cortos matemáticos cuando corresponda**: `X_train`, `y_test`,
  `auc`, `f1`, no `training_features_matrix_for_the_model`.
- **Comentarios inline solo si el código no se explica solo.** Comenta el
  *porqué*, no el *qué*.
- **Docstrings en español** en toda función pública, breves (una o dos frases),
  con cita bibliográfica cuando el método tiene una referencia académica
  concreta.Ejemplo:
  `"""Descomposición de valores singulares truncada (Halko et al., 2011)."""`
En todas las funciones debe estar claro lo que entra y lo que devuelve.
- **Sin manejo defensivo interno**. Valida solo en las boundaries (carga de
  datos, entrada de la CLI). Dentro asume contratos razonables.
- **Sin logging excesivo**. Un `print` o `logger.info` por hito importante,
  no en cada función.
- **Semilla en TODO**. Cada script arranca con:
  ```python
  import random, numpy as np
  from config import SEED
  random.seed(SEED); np.random.seed(SEED)
  ```
- **Sin código muerto ni imports sin usar.**
- **Longitud de líneas**: 88 caracteres (compatible con `black`).

### Cómo NO debe verse el código

- Docstrings idénticas con bloques `Args:`/`Returns:`/`Raises:` en cada
  función.
- `try/except` alrededor de operaciones que no pueden fallar.
- Sobre-modularización: un fichero con cinco líneas y una sola función.
- Nombres tipo `calculate_containment_score_between_response_and_source`
  cuando `containment(r, s)` basta.

---

## 5. Flujo de git y ramas

- `main` está protegida: no se hace `push` directo.
- Cada tarea nueva se desarrolla en una rama con prefijo semántico:
  - `feat/<descripcion>` — funcionalidad nueva.
  - `fix/<descripcion>` — corrección.
  - `exp/<nombre-experimento>` — experimento nuevo.
  - `docs/<descripcion>` — documentación.
  - `refactor/<descripcion>` — reorganización sin cambio funcional.
- **Commits en español** con prefijos cortos: `feat:`, `fix:`, `docs:`,
  `test:`, `refactor:`, `exp:`, `bitacora:`.
- Cada rama se abre a PR sobre `main` cuando el trabajo está listo. Descripción
  clara del PR: qué entra, qué tests se añaden, referencia a BITACORA si
  aplica.
- Auto-merge con squash cuando CI verde (`gh pr merge --auto --squash
  --delete-branch`).
- Commits pequeños y descriptivos. Nunca mezclar refactor y feature en el
  mismo commit.

---

## 6. Tests

Cobertura mínima pero no negociable:

1. **`tests/test_features.py`** — para cada feature, un caso feliz con entrada
   realista, un caso límite (respuesta vacía, fuente vacía, sin números), y un
   test de determinismo (mismo input → mismo output).

2. **`tests/test_data.py`** — la función de carga devuelve el esquema esperado
   (columnas correctas, dtypes correctos), y la función de parseo de
   `hallucination_labels` maneja los tres formatos posibles
   (`str "[]"`, `str "[...]"`, `list`).

3. **`tests/test_spans.py`** — el parseo de spans alucinados devuelve los offsets
   correctos y una lista vacía cuando la respuesta está limpia.

4. **`tests/test_no_leakage.py`** — test crítico no opcional. Verifica que en
   cada fold de `GroupKFold` por `source`, ninguna `source` aparece
   simultáneamente en train y test. Este test corre en CI y su fallo bloquea
   cualquier merge a main.

5. **`tests/test_determinism.py`** — entrenar dos veces con la misma semilla
   sobre un subset pequeño y verificar que las probabilidades predichas
   coinciden hasta el último decimal.

Comando estándar: `pytest tests/ -v`. Todos los tests verdes antes de mergear
a `main`. Salvo `test_no_leakage`, arrancan como esqueletos `skip` hasta que la
Fase 0 implemente la lógica que prueban.

---

## 7. Cómo trabajar con Claude Code

Este proyecto se desarrolla en colaboración con Claude Code. Reglas del juego:

**Antes de arrancar cualquier tarea.**
- Leer `CLAUDE.md` (este fichero) entero.
- Leer `GUIA_TFG_ALUCINACIONES.md` para saber qué se está haciendo.
- Leer `BITACORA.md` para no repetir errores ya documentados.
- Comprobar el estado del repo (`git status`, `git log --oneline -20`).

**Durante el desarrollo.**
- Trabajar en una rama nueva (nunca directo en `main`).
- Commits atómicos con mensaje descriptivo en español.
- Cada script nuevo tiene su semilla fijada y un `if __name__ == "__main__"`
  al final que hace visible el resultado principal.

**Al cerrar una tarea.**
- `pytest tests/ -v` en verde.
- Si es un experimento, generar un fichero `outputs/reports/<nombre>.md` con
  el objetivo, el método, la tabla de resultados y una conclusión de tres
  líneas.
- Actualizar `BITACORA.md` si el experimento produjo una decisión metodológica
  relevante (no si solo produjo un número).
- Abrir PR con `gh pr create` y activar auto-merge.

**Si te bloqueas más de una hora.**
- Documentar el bloqueo en `BITACORA.md` (sección "Bloqueos abiertos") y
  parar. No improvisar soluciones que comprometan la arquitectura.

**Si encuentras un error en algo ya hecho.**
- No lo arregles silenciosamente. Documenta el error en `BITACORA.md`
  (contexto, causa, solución) y arréglalo con un commit `fix:` separado.
  La trazabilidad de errores es parte de la defensa del TFG.

---

## 8. BITACORA.md

Cuaderno de campo del proyecto, versionado. **Documenta el proceso, no solo el
resultado.** Es material defendible ante el tribunal.

**Cuándo actualizar.**
- Al cerrar una fase del plan de trabajo.
- Al tomar una decisión metodológica relevante (umbral fijado, feature
  descartada por redundante, cambio de partición, etc.).
- Al resolver un error que generó tiempo perdido.
- Al descubrir algo interesante sobre el fenómeno estudiado (por ejemplo,
  "el AUC en Data2txt es mucho más alto que en QA porque las alucinaciones son
  más gruesas").

**Cuándo NO actualizar.**
- Después de cada commit pequeño.
- Para anotar progreso trivial.
- Con mensajes tipo "he completado X y voy a empezar Y".

**Formato de cada entrada.**

```markdown
## [YYYY-MM-DD] [Milestone | Decisión | Error | Hallazgo] - Título

**Contexto.** Una o dos frases sobre qué se estaba haciendo.

**Detalle.** Lo ocurrido. Suficiente para entenderlo dentro de tres semanas
sin reconstruir el contexto.

**Implicaciones para la memoria.** Si aplica. Qué sección o frase de la
memoria del TFG habrá que revisar a la luz de esto.

**Referencias.** Commits, ficheros, papers.
```

---

## 9. Siguientes pasos concretos

La estructura ya está montada (paquete `ragcheck` con módulos como stubs,
scripts, tests, CI). Estamos en **Fase 0 — Reproducir la base** (`§10` de
`GUIA_TFG_ALUCINACIONES.md`): rellenar la lógica de los stubs, en orden.

1. **Setup del entorno**. `.venv` con Python 3.12, `pip install -e ".[dev]"`,
   `python -m spacy download en_core_web_sm`. Verificar que `datasets` descarga
   RAGTruth-processed y que `import ragcheck` funciona.

2. **`src/ragcheck/data.py`**. `load_ragtruth(split) -> pd.DataFrame` que
   descarga, parsea `hallucination_labels` a `label` binaria (snippet de la guía
   §4.5) con `has_hallucination`, renombra `output → response`,
   `context → source`, y devuelve `[id, source, response, label, task_type, model]`.

3. **`src/ragcheck/features.py`**. Implementar las cinco funciones puras
   (`containment`, `jaccard`, `tfidf_cos`, `num_overlap`, `answer_len`) y
   `extract_features(df)`. (`config.py` y `models.py` ya están hechos.)

4. **`src/ragcheck/spans.py`**. `parse_spans` y `span_distributions` para el
   descriptivo de Nivel 2.

5. **`src/ragcheck/training.py` y `evaluate.py`**. CV con `GroupKFold` por
   `source`, persistencia, y las métricas por secciones A–F.

6. **`scripts/`**. Los dos descriptivos (`visualization.py`, `spans_analysis.py`),
   un modelo por fichero (`log_reg`, `knn`, `svm`, `random_forest`, `xgb`) y
   `compare_models.py`, que reproduce la línea base ~0.78 y escribe
   `outputs/reports/e0_baseline.md`.

7. **Tests**. Completar `test_features`, `test_data`, `test_spans`,
   `test_determinism` (quitar el `skip`). `test_no_leakage` ya está activo.

8. **PR de la Fase 0 a `main`**. Con la tabla de resultados y entrada en BITACORA.

---

## 10. Lo que NO se hace

- **No usar KFold aleatorio** sobre RAGTruth. Siempre `GroupKFold` por
  `source` o el split oficial train/test. Un KFold aleatorio filtra información
  entre folds y el resultado se infla.
- **No entrenar TF-IDF o SVD sobre el conjunto entero antes de partir.** Se
  ajusta solo con `X_train` y se aplica a `X_test`. Cualquier otra cosa es
  fuga.
- **No calibrar hiperparámetros sobre el test.** Se usa un split de validación
  dentro del train si es necesario.
- **No usar embeddings neuronales**. Ni ahora ni después. La única
  representación semántica permitida es LSA (TruncatedSVD sobre TF-IDF).
- **No mergear a `main` sin tests verdes y CI en verde.**
- **No actualizar BITACORA con ruido.** Solo entradas significativas.
- **No refactorizar código que funciona y pasa tests "por estilo".**

---

## 11. Referencias mínimas para el marco teórico

Para la memoria del TFG, referencias que necesitarán aparecer:

- **RAG**: Lewis et al. 2020 (paper original de RAG).
- **Alucinaciones en LLMs**: Ji et al. 2023 (survey de alucinaciones).
- **RAGTruth**: Niu et al. 2024 (paper del dataset).
- **TF-IDF**: Salton & Buckley 1988.
- **LSA / TruncatedSVD**: Deerwester et al. 1990; Halko et al. 2011 (SVD
  aleatorizado).
- **Regresión logística**: Cox 1958; Hastie/Tibshirani/Friedman ESL cap. 4.
- **Gradient Boosting**: Friedman 2001.
- **XGBoost**: Chen & Guestrin 2016.
- **ROC / AUC**: Fawcett 2006.
- **Bootstrap para intervalos**: Efron 1979; DeLong et al. 1988 para comparar
  AUCs.
- **Calibración de probabilidades**: Platt 1999 (Platt scaling); Zadrozny &
  Elkan 2002 (isotónica).

---

## 12. Resumen ejecutivo

Prioridades por orden:

1. Determinismo y reproducibilidad (semillas, cache, entorno fijo).
2. Evaluación honesta (GroupKFold, permutación, intervalos).
3. Código limpio que mapea 1-a-1 a la memoria del TFG.
4. BITACORA actualizada con cada decisión relevante.
5. Tests verdes antes de cerrar cualquier PR.
6. Nada que parezca producido por IA (código sin patrones IA típicos).

Empieza por leer `GUIA_TFG_ALUCINACIONES.md`, después ejecuta la Fase 0 en el
orden de §9 de este fichero.


# 13. RESUMEN DEL PROYECTO
Título: "Detección de alucinaciones en sistemas RAG mediante técnicas de Machine Learning"


