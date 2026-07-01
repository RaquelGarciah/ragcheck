# BITÁCORA — Detección de alucinaciones en RAG con ML clásico

Cuaderno de campo del proyecto. Documenta el proceso y las decisiones
metodológicas, no el progreso trivial. Formato de cada entrada en `CLAUDE.md` §8.

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
