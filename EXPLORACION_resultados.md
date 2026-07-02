# Exploración de resultados — detector clásico sobre RAGTruth

Cuaderno de exploración de la sesión: diagnóstico, features, análisis de error,
métricas completas e importancia. Recoge todas las tablas producidas y las
conclusiones. **No es la fuente citable** (esa es `outputs/reports/`); es el hilo
de razonamiento con los datos, para entender los resultados de un vistazo.

Todas las cifras: protocolo honesto (`GroupKFold` por `source` en train para OOF;
evaluación final en el **split oficial test** de RAGTruth, n=2700, intacto).
Semilla 42. Modelo por defecto: XGBoost (mejor de la comparativa).

---

## 1. Punto de partida y diagnóstico por tarea

Con las 5 features iniciales, el detector daba AUC ~0,80 y un F1 que parecía
flojo. El primer paso fue **no** creerse el agregado y mirar por tarea, porque el
gap con el estado del arte no era uniforme.

**AUC-ROC por tarea (OOF, xgboost), evolución a lo largo de la sesión:**

| Tarea | 5 features | + nivel frase + task | + relacionales |
|---|---|---|---|
| Data2txt | 0,763 | 0,793 | **0,813** |
| QA | 0,800 | 0,809 | **0,812** |
| **Summary** | **0,676** | **0,702** | **0,701** |
| **GLOBAL** | **0,804** | **0,819** | **0,824** |

**Conclusión.** El cuello de botella está concentrado en **Summary**. Data2txt ya
rozaba el SOTA y QA estaba sano; Summary (AUC ~0,70) hunde el agregado. No es
problema de umbral (el AUC es independiente del umbral) → es problema de **señal**.

---

## 2. Estado del arte verificado (para posicionar el resultado)

Cifras **verificadas contra fuentes primarias** (ver `literatura_RAGTRUTH.md`
§11). La métrica del SOTA es **F1 a nivel respuesta** y **accuracy**, *no* AUC.

**F1 nivel respuesta sobre RAGTruth (split oficial):**

| Método | QA | Data2txt | Summ | **Global** |
|---|---|---|---|---|
| GPT-4-turbo (prompt) | 45,6 | 78,3 | 47,6 | 63,4 |
| Luna (encoder ligero) | 51,3 | 75,9 | 52,5 | 65,4 |
| LettuceDetect-base (150M) | 65,5 | 87,9 | 50,5 | 76,1 |
| Fine-tuned Llama-2-13B | 68,2 | 88,1 | 59,1 | 78,7 |
| LettuceDetect-large (396M) | 70,2 | 88,5 | 59,7 | 79,2 |
| **RAG-HAT (mejor)** | 74,8 | 91,6 | 67,6 | **83,9** |

**Accuracy sobre el subset RAGTruth (Lynx, Tabla 3):** GPT-3.5-turbo 50,7 ·
Claude-3-Sonnet 79,1 · Lynx-8B 80,0 · Lynx-70B 80,2 · **GPT-4o 84,3**.

**Patrón universal:** *todos* los sistemas sacan ~88–92% en Data2txt y solo
~48–68% en Summary. Nadie ha resuelto Summary — chocamos contra el mismo muro que
modelos de miles de millones de parámetros, solo un poco antes.

**AUC en la literatura (§11.9):** apenas se reporta. Solo cifras dispersas y en
regímenes distintos (ECS-Span 0,742 a nivel span; PARALLAX, por activaciones,
0,43–0,57 ≈ azar). **No hay AUC publicado de Lynx/HHEM/GPT-4o**, así que la
comparación externa se hace en F1/accuracy, no en AUC. PARALLAX además afirma que
en RAGTruth "la etiqueta no está correlacionada con diferencias de texto de
superficie" → que un clásico saque 0,82–0,84 de AUC es no trivial.

---

## 3. Análisis de error: por qué Summary es el techo

Hipótesis intuitiva ("Summary tiene más alucinación *sutil*") → **refutada**.

**Composición de tipos de alucinación por tarea (fracción de spans):**

| task_type | Evident Baseless | Evident Conflict | Subtle Baseless | Subtle Conflict |
|---|---|---|---|---|
| Data2txt | 0,383 | 0,447 | 0,163 | 0,007 |
| QA | 0,522 | 0,146 | 0,314 | 0,018 |
| Summary | 0,514 | 0,343 | 0,100 | 0,043 |

Summary es **~86% evidente**, y **QA tiene más tipo sutil (0,33) que Summary
(0,14) pero puntúa mejor**. El tipo anotado NO explica la dificultad.

**Lo que sí la explica: camuflaje léxico.** Containment del *texto alucinado*
contra la fuente, por tarea:

| Tarea | containment del span (media) | mediana |
|---|---|---|
| Data2txt | 0,427 | 0,429 |
| QA | 0,482 | 0,500 |
| **Summary** | **0,620** | **0,667** |

Un resumen reutiliza el vocabulario del documento, así que el span alucinado
comparte palabras con la fuente aunque el error sea evidente. Confirmación
directa: **dentro de Summary, los falsos negativos tienen más containment (0,641)
que los aciertos (0,555)** — el modelo falla justo en los spans camuflados.

**Muestra de falsos negativos de Summary (span_containment = 1,00):**

- `«age of 55 | NSYNC and Chris Kirkpatrick,»`
- `«majority of black members»`
- `«completed the Boston Marathon in 26.2 hours»` ← son 26,2 *millas*: error
  evidente, léxicamente invisible.
- `«two men»`
- `«six children.»`

**Conclusión.** El techo de Summary es estructural: resumir *es* recombinar, así
que la frase fiel y la alucinada tienen la misma firma léxica. Para separarlas
haría falta representación semántica de verdad — la frontera del ML clásico.

Figuras: `outputs/figures/err_prob_por_tarea.png`, `err_span_containment.png`,
`err_tipo_por_tarea.png`. Informe: `outputs/reports/error_analysis_summary.md`.

---

## 4. Features usadas (18 en total)

| Grupo | Features |
|---|---|
| **Escalares por par (9)** | `containment`, `jaccard`, `tfidf_cos`, `num_overlap`, `novel_bigram`, `novel_trigram`, `num_context`, `neg_diff`, `answer_len` |
| **Nivel frase (6)** | `sent_cont_min`, `sent_cont_mean`, `sent_cont_std`, `sent_frac_low`, `sent_sim_min`, `sent_sim_mean` |
| **One-hot tarea (3)** | `task_QA`, `task_Summary`, `task_Data2txt` |

**Ideas clave de diseño:**
- **Nivel frase**: una alucinación suele ser *una* frase sin respaldo que el
  promedio respuesta-entera diluye; se agrega por frase (min/std/…), y el mínimo
  delata la peor sostenida.
- **Relacionales** (`novel_bigram/trigram`, `num_context`, `neg_diff`): cazan la
  *recombinación* que el solape de unigramas no ve. Ej.: "26.2 hours" tiene
  unigram-containment 1,0 pero el bigrama "26.2 hours" no está en la fuente.

**Features probadas y DESCARTADAS por nulas (criterio "sin código muerto"):**
- **LSA** (TF-IDF + TruncatedSVD, ajustado por fold sin fuga): AUC +0,001 global,
  Summary plana. Redundante con las de similitud a nivel frase.
- **`sent_novel`** (novedad de n-gramas a nivel frase): +0,001 global. En Summary
  no separa porque la frase fiel de un resumen recombina igual que la alucinada.

---

## 5. Métricas completas

### 5.1 Comparativa de modelos (test oficial, n=2700, prev=0,349, umbral Youden)

| modelo | AUC | AUC-PR | F1 | prec | recall | acc | bal_acc | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| **random_forest** | **0,836** | 0,745 | 0,669 | 0,691 | 0,649 | 0,776 | 0,747 | 0,158 | 0,069 |
| xgboost | 0,835 | 0,748 | 0,681 | 0,678 | 0,684 | 0,776 | 0,755 | 0,157 | **0,063** |
| svm | 0,819 | 0,732 | **0,683** | 0,646 | 0,724 | 0,765 | **0,756** | 0,160 | 0,068 |
| knn | 0,826 | 0,735 | 0,672 | 0,656 | 0,688 | 0,765 | 0,747 | 0,160 | 0,065 |
| logreg | 0,816 | 0,711 | 0,658 | 0,557 | 0,803 | 0,708 | 0,730 | 0,177 | 0,111 |

### 5.2 XGBoost por tarea (test oficial, umbral 0,487)

| tarea | n | prev | AUC | AUC-PR | F1 | prec | recall | acc | bal_acc |
|---|---|---|---|---|---|---|---|---|---|
| GLOBAL | 2700 | 0,349 | 0,835 | 0,748 | 0,681 | 0,678 | 0,684 | 0,776 | 0,755 |
| Data2txt | 900 | 0,643 | 0,782 | 0,853 | **0,819** | 0,767 | 0,879 | 0,750 | 0,698 |
| QA | 900 | 0,178 | **0,822** | 0,535 | **0,519** | 0,481 | 0,562 | 0,814 | 0,716 |
| Summary | 900 | 0,227 | 0,689 | 0,389 | **0,303** | 0,460 | 0,225 | 0,764 | 0,574 |

**El hallazgo que explica el F1:** QA tiene MÁS AUC que Data2txt (0,822 vs 0,782)
pero MENOS de la mitad de F1 (0,52 vs 0,82). Causa: **prevalencia**. QA en test
solo tiene 17,8% de positivos; Data2txt 64,3%. Con pocos positivos la precisión se
desploma aunque el ranking (AUC) sea excelente. **El F1 mide separación mezclada
con prevalencia; el AUC mide separación pura.** Por eso el esfuerzo se ve en el AUC
y no en el F1.

### 5.3 Matriz de confusión (xgboost, test)

| tarea | TN | FP | FN | TP | lectura |
|---|---|---|---|---|---|
| GLOBAL | 1451 | 306 | 298 | 645 | — |
| Data2txt | 166 | 155 | 70 | 509 | caza casi todo (recall 88%) |
| QA | 643 | 97 | 70 | 90 | se calla (recall 56%) |
| **Summary** | 642 | 54 | **158** | **46** | **casi ciego: 46 de 204** |

### 5.4 XGBoost por tarea (CV OOF en train, n=15090, prev=0,445)

| tarea | n | prev | AUC | AUC-PR | F1 | prec | recall | bal_acc |
|---|---|---|---|---|---|---|---|---|
| GLOBAL | 15090 | 0,445 | 0,824 | 0,798 | 0,700 | 0,750 | 0,657 | 0,741 |
| Data2txt | 5298 | 0,694 | 0,813 | 0,896 | 0,846 | 0,810 | 0,885 | 0,708 |
| QA | 5034 | 0,311 | 0,812 | 0,660 | 0,583 | 0,649 | 0,529 | 0,700 |
| Summary | 4758 | 0,311 | 0,701 | 0,480 | 0,320 | 0,562 | 0,224 | 0,572 |

---

## 6. Umbral: por qué el "max-F1 gratis" es un espejismo

Se creyó que fijar el umbral por max-F1 (en vez de Youden) daba ~+3 puntos de F1.
Medido **honestamente** (umbral fijado en train, aplicado a test):

| Umbral (fijado en train) | F1 en test |
|---|---|
| Youden | **0,681** |
| max-F1 global | 0,674 ⬇ |
| max-F1 por tarea | 0,660 ⬇ |

**El truco empeora el F1 en test.** Causa: la prevalencia de train (0,445) ≠ test
(0,349); el umbral bajo que maximiza F1 en train es demasiado agresivo para el
test. En muestra (OOF de train) sí subía (0,724 vs 0,695), pero era un espejismo.
**Decisión: mantener Youden y no vender el umbral como palanca de F1.**

---

## 7. Qué features explican más

### 7.1 Importancia por permutación (xgboost, test) — caída de AUC

| feature | caída de AUC | | feature | caída de AUC |
|---|---|---|---|---|
| **containment** | **0,161** | | sent_sim_min | 0,008 |
| answer_len | 0,020 | | sent_cont_min | 0,006 |
| jaccard | 0,019 | | sent_sim_mean | 0,005 |
| num_context | 0,012 | | neg_diff | 0,004 |
| task_QA | 0,010 | | tfidf_cos | 0,004 |
| num_overlap | 0,010 | | resto | < 0,004 |

**`containment` explica ~8× más que la siguiente.** Rompiéndola, el AUC cae 16
puntos; cualquier otra, <2. Una única feature léxica trivial (fracción de palabras
de la respuesta presentes en la fuente) carga con casi toda la señal → argumento
"clásico e interpretable" hecho dato.

### 7.2 Permutación POR TAREA (top-3, caída de AUC)

| Tarea | #1 | #2 | #3 |
|---|---|---|---|
| **Data2txt** | **`num_context` (0,068)** | containment (0,053) | jaccard (0,026) |
| **QA** | containment (0,146) | answer_len (0,043) | neg_diff (0,012) |
| **Summary** | containment (0,108) | jaccard (0,034) | num_overlap (0,017) |

- En **Data2txt**, la relacional `num_context` es la #1 (por encima de
  containment): la consistencia número-unidad es *lo que* caza las cifras
  fabricadas. Justifica el trabajo con datos.
- **QA y Summary** son "problemas de containment": todo depende del solape léxico;
  en QA funciona (AUC 0,82), en Summary la misma feature rinde peor (AUC 0,69) por
  el camuflaje.

### 7.3 Coeficientes logreg (features estandarizadas)

`coef>0` empuja hacia ALUCINA; `coef<0` hacia LIMPIO (top por magnitud):

| feature | coef | | feature | coef |
|---|---|---|---|---|
| **containment** | **−1,842** | | sent_frac_low | −0,216 |
| sent_cont_mean | 0,617 | | tfidf_cos | 0,172 |
| novel_bigram | −0,449 | | jaccard | −0,171 |
| sent_cont_min | −0,383 | | num_context | −0,166 |
| task_QA | −0,279 | | task_Data2txt | 0,162 |
| sent_cont_std | 0,239 | | novel_trigram | 0,148 |
| sent_sim_min | 0,236 | | task_Summary | 0,117 |

`containment` = −1,84 (3× el siguiente), signo correcto (más solape → menos
alucina).

**Matiz honesto:** las features de solape (`containment`, `jaccard`, `tfidf_cos`,
`sent_cont_*`) están muy **correlacionadas**, así que los coeficientes logreg más
allá de `containment` **no son fiables** (efectos de supresión: p. ej.
`novel_bigram` −0,45 aparenta contraintuitivo — es colinealidad, no un hallazgo).
**La permutación es el ranking de fiar; la logreg, solo para leer `containment`.**

---

## 8. Síntesis: qué capa el rendimiento

1. **Summary** — señal insuficiente (camuflaje léxico). Techo estructural del ML
   clásico. AUC ~0,70 y recall que colapsa (46 de 204).
2. **Prevalencia baja en QA/Summary en el test** (18% / 23%) → un umbral global
   calla los positivos → recall se hunde → F1 baja aunque el AUC sea bueno. QA en
   realidad discrimina bien (AUC 0,82); su F1 (0,52) es artefacto de prevalencia.
3. **Desajuste train→test**: QA pasa de 31% de positivos en train a 18% en test;
   el umbral aprendido queda alto.

**Lectura para la memoria.** El **AUC (0,84) refleja lo que el modelo aprende**;
el **F1 (0,68) está mecánicamente penalizado** por prevalencia y umbral. Reportar
AUC como métrica interna y F1/accuracy para comparar con el SOTA, siempre **por
tarea** y con la explicación del techo de Summary. Posición honesta: por encima de
GPT-4-turbo-juez (F1 0,63), ~10 pts bajo los fine-tuned (0,79–0,84), a coste cero,
CPU, interpretable y determinista.

---

## 9. Estado del código

- **Features**: `src/ragcheck/features.py` (18 features; LSA y `sent_novel`
  descartadas y documentadas).
- **Análisis de error**: `exploration/error_analysis.py` →
  `outputs/reports/error_analysis_summary.md` + 3 figuras.
- **Informes citables regenerados**: `outputs/reports/{e0_baseline,test_oficial,
  <modelo>}.md`.
- **Tests**: 35/35 verdes (incluidos `test_no_leakage` y `test_determinism`).
- **BITACORA**: hallazgo del camuflaje léxico, decisión de features relacionales,
  corrección del umbral y del AUC en la documentación.
- Rama `exp/error-analysis-summary` (PR a `main` pendiente).
