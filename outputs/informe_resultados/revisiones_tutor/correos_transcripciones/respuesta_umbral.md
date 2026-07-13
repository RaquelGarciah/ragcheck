# Respuesta a la pregunta del tutor sobre el umbral de decisión

**Pregunta (Daniel, p. 26 de `resultados_revisado.tex`, líneas 1235-1236):**
> «Me pregunto si el umbral por tarea lo habías fijado con el índice de Youden o si lo habías hecho
> sin distinguir por tarea.»

## Respuesta corta

Son **dos criterios distintos**:

- **Umbral global**: índice de **Youden**, un único corte para toda la base.
- **Umbral por tarea**: **un corte por tarea que maximiza el F1 de esa tarea** (no Youden). Se exploró
  también una variante que maximiza F2 (pesa más el recall).

Ambos se fijan **sobre el conjunto de entrenamiento** (predicciones out-of-fold de la validación
cruzada) y se aplican **sin reajuste al test**.

## Detalle verificable (código)

1. **Umbral global — Youden.** `src/ragcheck/evaluate.py:110-113`:
   ```python
   def best_threshold(y_true, y_prob):
       fpr, tpr, thr = roc_curve(y_true, y_prob)
       return float(thr[np.argmax(tpr - fpr)])   # J = sensibilidad + especificidad - 1
   ```
   Se aplica sobre las OOF de train (`exploration/grid_tables.py:99-102`,
   `exploration/imbalance_thresholds.py:114` → `t = ev.best_threshold(ytr, ptr)`).

2. **Umbral por tarea — max-F1 (variante F2).** `exploration/imbalance_thresholds.py:10-11,65-70,121`
   y `exploration/figuras_11feat.py:74-75`:
   ```python
   def best_fbeta(y, p, beta):            # corte que maximiza F_beta sobre la curva P-R
       prec, rec, thr = precision_recall_curve(y, p)
       fb = (1+beta**2)*prec*rec / (beta**2*prec + rec + 1e-12)
       return float(thr[int(np.argmax(fb[:-1]))])
   t_task = {k: best_fbeta(ytr[task==k], ptr[task==k], 1) for k in TASKS}   # por tarea, F1
   t_f2   = {k: best_fbeta(ytr[task==k], ptr[task==k], 2) for k in TASKS}   # por tarea, F2
   ```
   Es decir, para cada tarea (Data2txt, QA, Summary) se elige el corte que maximiza su F1 sobre las OOF
   de train. Las columnas con `*` de la tabla por tarea usan este umbral por tarea (F1).

## Por qué se distingue por tarea

El AUC es **independiente del umbral**; F1, precisión, recall y accuracy **dependen** de él. La
prevalencia de alucinación cambia mucho entre tareas (Data2txt ≈ 0,64; QA ≈ 0,18; Summary ≈ 0,23), así
que un único corte es subóptimo: el de Youden global es demasiado alto para QA y Summary y deja pasar
alucinaciones. Un corte por tarea recupera esos falsos negativos.

**Efecto (test, n = 2700, XGBoost):** los falsos negativos totales bajan de **285 → 125** (Summary
146→46, QA 76→27), a costa de la precisión y la accuracy agregadas (F1 global 0,685 → 0,640; accuracy
77,6 % → 65,9 %). Es un cambio de punto de operación, no de modelo: el mismo clasificador, otro corte.

## Qué corregir en la redacción (para que quede claro)

En el capítulo práctico, al presentar el umbral por tarea, decir explícitamente que **el global usa
Youden y el por tarea maximiza F1 (con F2 como variante)**, y que ambos se fijan en entrenamiento y se
aplican al test. Hoy el texto (p. 26) no lo especifica, y de ahí la duda del tutor. Las métricas
(Youden, F1, F2, precisión, recall, balanced accuracy) se definen en el capítulo teórico (3.2.5
Valoración), como pide el propio tutor (comentario p. 16).
