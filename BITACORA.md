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
