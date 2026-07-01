# BITÁCORA — Detección de alucinaciones en RAG con ML clásico

Cuaderno de campo del proyecto. Documenta el proceso y las decisiones
metodológicas, no el progreso trivial. Formato de cada entrada en `CLAUDE.md` §8.

---

## [2026-07-01] Hallazgo — El cuello de botella de Summary es camuflaje léxico, no tipo sutil

**Contexto.** Tras las features de nivel frase, Summary seguía siendo la tarea
débil (AUC 0,702 vs QA 0,809, Data2txt 0,793). Análisis de error para saber por
qué, antes de invertir en más features (`exploration/error_analysis.py`).

**Detalle.** La hipótesis intuitiva —"Summary es difícil porque tiene más
alucinación *sutil*"— es **falsa** y los datos la refutan: Summary es ~86%
*evidente* (Evident Baseless 0,51 + Evident Conflict 0,34) y **QA tiene más tipo
sutil (0,33) que Summary (0,14) pero puntúa mejor**. El tipo anotado no explica
la dificultad.

Lo que sí la explica es el **camuflaje léxico**, medido con el containment del
*texto alucinado* contra la fuente: Data2txt 0,43 · QA 0,48 · **Summary 0,62**
(mediana 0,67). Un resumen reutiliza el vocabulario del documento, así que el
span alucinado comparte palabras con la fuente incluso cuando el error es
evidente, y el solape léxico no lo separa del texto fiel. Confirmación directa:
dentro de Summary, los **falsos negativos tienen mayor containment (0,641) que
los aciertos (0,555)** — el modelo falla justo en los spans camuflados. Ejemplo
canónico: *"completed the Boston Marathon in 26.2 hours"* (son 26,2 *millas*),
containment de span = 1,00: error evidente, léxicamente invisible.

**Implicaciones para la memoria.** (1) El capítulo de resultados puede afirmar,
con figura, que el techo del clásico en Summary es *estructural* (camuflaje
léxico), no un artefacto del tipo de alucinación —refuta una explicación fácil y
la sustituye por una medida. (2) Marca la dirección de features futuras: para
Summary hace falta señal **relacional** (consistencia número-unidad,
entidad-relación, polaridad/negación), no más solape de vocabulario; añadir
features de overlap ahí tiene techo. (3) Refuerza el marco: es la frontera entre
minería léxica y representación semántica.

**Referencias.** `exploration/error_analysis.py`;
`outputs/reports/error_analysis_summary.md`; figuras `err_prob_por_tarea`,
`err_span_containment`, `err_tipo_por_tarea`.

---

## [2026-07-01] Decisión — Features de nivel frase; LSA descartada por redundante

**Contexto.** El diagnóstico por tarea mostró que el gap con el estado del arte
(RAG-HAT 0,839 F1; Llama-2-13B 0,787; LettuceDetect 0,792, todos nivel respuesta)
no era uniforme: Data2txt ya rozaba el SOTA (F1 0,83) y QA estaba en mitad de
tabla, pero **Summary era el agujero** (AUC 0,676, casi azar). La causa: las cinco
features originales miden solape léxico **respuesta-entera**, y una alucinación
suele ser *una* frase sin respaldo que el promedio global diluye —justo el patrón
de los resúmenes (paráfrasis incorrecta).

**Detalle.** Se añaden dos bloques a `features.py`:
- **Nivel frase** (`SENTENCE_FEATURES`, 6 columnas): se trocea la respuesta en
  frases y se agrega el soporte de cada una contra la fuente con **mínimo, media,
  desviación y fracción de frases mal sostenidas**. Dos señales por frase:
  containment léxico y mejor coseno TF-IDF contra las frases de la fuente (TF-IDF
  ajustado por par, sin fuga). El mínimo delata la frase peor sostenida.
- **`task_type` one-hot** (`task_{QA,Summary,Data2txt}`): deja que el modelo
  especialice por tarea. Solo se añade si la columna existe, para no romper el
  contrato de `inference.score(response, source)`.

Efecto (OOF GroupKFold, xgboost, hiperparámetros fijos): AUC global **0,804 →
0,819**; por tarea Data2txt 0,763 → 0,793, QA 0,800 → 0,809, Summary 0,676 →
0,702.

**LSA descartada.** Se implementó y probó `LsaSimilarity` (TF-IDF + TruncatedSVD
ajustado **por fold**, sin fuga, como manda §10). Aporte marginal nulo: AUC global
0,819 → 0,820 y Summary 0,702 → **0,701** (plano/negativo). Las features de
similitud TF-IDF a nivel frase ya capturan esa señal semántica, así que LSA es
redundante. Se **elimina el código** (regla "sin código muerto") en vez de dejarlo
inerte. Es un resultado negativo defendible ante el tribunal.

**Implicaciones para la memoria.** El capítulo de features distingue ahora
features respuesta-entera vs. **nivel frase**, y el arco de resultados puede
mostrar el salto de Summary. La LSA aparece como vía **explorada y descartada con
datos**, no omitida —refuerza el marco (el techo de las features de superficie en
alucinación sutil). Pendiente: re-lanzar grid search + test oficial sobre el nuevo
conjunto de features para actualizar las cifras citables (`e0_baseline.md`,
`test_oficial.md`, `<modelo>.md`).

**Referencias.** `src/ragcheck/features.py` (`_sentence_features`,
`SENTENCE_FEATURES`, `TASKS`); `tests/test_features.py`.

---

## [2026-07-01] Milestone — Estructura del proyecto definida

**Contexto.** El repositorio estaba en verde (sin código Python, sin git) y con
`tfg-kit/` anidado sin integrar. Antes de arrancar la Fase 0 se fijó la
arquitectura completa en una entrevista de diseño.

**Detalle.** Decisiones tomadas:
- Entregable = **librería instalable `ragcheck`** con `src-layout`; alcance doble
  (API de inferencia `score(response, source)` + pipeline reproducible).
- **Un fichero por modelo** en `scripts/` (atómicos y citables en la memoria),
  con la lógica pesada en la librería. Cinco modelos: LogReg, KNN, SVM, Random
  Forest, XGBoost. `scripts/xgb.py` se llama así para no ensombrecer al paquete
  `xgboost`.
- Separación **`scripts/` (final, citable)** vs **`exploration/` (borrador)**.
- **Dos repositorios hermanos**: código PÚBLICO (`ALUCINACIONES_RAG`, portfolio)
  y tesis PRIVADO (`../tfg-tesis`, kit de redacción + LaTeX en `tesis/`). El repo
  de código es la fuente única de cifras y figuras.
- **Métricas por secciones A–F**; todas las figuras con seaborn y estilo común.
- **Dos descriptivos**: dataset (Nivel 1) y spans alucinados (Nivel 2, con
  posición, longitud, POS y NER vía spacy usado solo como herramienta de análisis).
- **Orden de trabajo**: código y resultados primero (Fase A), memoria después
  (Fase B). La estructura de capítulos se fija en la Fase B combinando
  `estructura.md`, el TFG de Sergio Gil y la transcripción del tutor.

**Implicaciones para la memoria.** El capítulo de metodología describirá la
librería y la separación motor/scripts; el de datos, los dos descriptivos; el de
resultados, la comparativa de los cinco modelos con las secciones A–F.

**Referencias.** `CLAUDE.md` §3, §6, §9; `GUIA_TFG_ALUCINACIONES.md` §7, §9;
plan de estructura aprobado.

---

## [2026-07-01] Hallazgo — SVM y KNN exigen estandarizar las features

**Contexto.** Primera comparativa de los cinco modelos (Fase 0) con las cinco
features léxicas sobre el train completo (15 090 filas), GroupKFold por `source`.

**Detalle.** Sin escalar, SVM daba AUC-ROC 0.656 (peor que el baseline trivial de
`containment`, 0.761) y KNN 0.741. La causa es que `answer_len` (decenas o
centenas de tokens) domina el núcleo RBF y la distancia euclídea frente a las
cuatro features acotadas en [0, 1]. Al meter `StandardScaler` en el pipeline de
SVM y KNN: SVM 0.656 → 0.792 y KNN 0.741 → 0.788. La regresión logística (0.769)
y los árboles (RF 0.788, XGBoost 0.804) no dependen de la escala, por eso no se
tocaron y la logística conserva la línea base ~0.78 de la viabilidad.

Ranking E0 (AUC-ROC, IC95%): XGBoost 0.804 [0.798, 0.811] > SVM 0.792 >
RF 0.788 ≈ KNN 0.788 > LogReg 0.769 > baseline containment 0.761.

**Implicaciones para la memoria.** El capítulo de metodología debe justificar el
preprocesado por modelo: escalado obligatorio en los métodos basados en distancia
o margen, innecesario en los de árbol. Es un punto de rigor que el tutor valora.

**Pendiente.** `SVC(probability=True)` está deprecado en scikit-learn 1.9 (se
elimina en 1.11); migrar a `CalibratedClassifierCV(SVC(), ensemble=False)` cuando
se aborde la calibración (Fase 2/3).

**Referencias.** `src/ragcheck/models.py` (build_svm, build_knn);
`scripts/compare_models.py`; `outputs/reports/e0_baseline.md`.

---

## [2026-07-01] Decisión — Grid search por modelo y evaluación en test oficial

**Contexto.** Los hiperparámetros estaban fijos a mano. Siguiendo el molde del TFG
de Sergio Gil (grid search por modelo maximizando F1), se añade búsqueda de
hiperparámetros a cada script de modelo y una evaluación final en el test oficial.

**Detalle.** `training.grid_search` usa `GridSearchCV` con `cv=GroupKFold` por
`source` y `scoring="f1"` (misma partición honesta, sin fuga de documento; la
propia `GroupKFold` exige `groups`, lo que garantiza que se usan). Cada script de
modelo expone su `PARAM_GRID`, reporta la rejilla (top-5) y las métricas A–F del
mejor, y guarda el modelo. El SVM se busca sin `probability` (F1 usa `predict`,
mucho más rápido) y solo el ganador se reentrena con `probability=True`.

Sesgo asumido y cómo se cierra: seleccionar hiperparámetros y reportar sobre los
mismos folds de CV es levemente optimista (no es fuga, es sesgo de selección). El
número **definitivo y defendible** se toma en el **test oficial** de RAGTruth
(2 700 ejemplos intactos), vía `scripts/evaluate_test.py`.

Resultados en test oficial (AUC-ROC, IC95%): XGBoost 0.821 [0.805, 0.837] >
KNN 0.819 > RF 0.817 > SVM 0.795 > LogReg 0.793. Todos por encima del umbral 0.80
de la hipótesis (salvo los dos lineales, muy cerca). El ECE empeora en test
(0.07–0.11; LogReg 0.114): los modelos están sobre-confiados fuera de muestra, lo
que motiva la calibración (Platt/isotónica) en la Fase 2.

**Implicaciones para la memoria.** El capítulo de resultados sigue el arco
línea base (parámetros por defecto, CV) → ajuste por grid search (CV) →
evaluación final (test oficial), como en los TFG ejemplares. La calibración pasa
a ser una palanca justificada por datos, no por intuición.

**Referencias.** `src/ragcheck/training.py` (grid_search, top_configs);
`src/ragcheck/report.py`; `scripts/{log_reg,knn,svm,random_forest,xgb}.py`;
`scripts/evaluate_test.py`; `outputs/reports/test_oficial.md` y `<modelo>.md`.
