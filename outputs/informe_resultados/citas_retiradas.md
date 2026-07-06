# Trazabilidad de citas — capítulo de resultados

Registro de las citas del capítulo de resultados. Criterio del tutor: **en este capítulo solo
las imprescindibles** (el dataset y los métodos con los que se compara); las **definiciones de
método** (regresión logística, SVM, TF-IDF, ROC, etc.) van al **marco teórico**.

**Todas las entradas siguen en `referencias.bib`** (no se borra nada del `.bib`), así que cualquiera
es recuperable escribiendo `\cite{clave}` en el texto. Este fichero es la red de seguridad
anti-plagio: si una afirmación necesita respaldo, aquí está la clave y qué cubre.

Actualizado: 2026-07.

---

## Citadas AHORA en el capítulo (6 — imprescindibles)

| Clave | Qué es | Por qué se queda |
|---|---|---|
| `niu2024ragtruth` | Corpus RAGTruth (Niu et al., 2024) | Es el dataset que se usa. |
| `guyon2002rfe` | Recursive Feature Elimination (Guyon et al., 2002) | Método aplicado en la selección de variables. |
| `guyon2003selection` | Introducción a la selección de variables (Guyon & Elisseeff, 2003) | Encuadre de filtro vs envolvente, aplicado aquí. |
| `kovacs2025lettucedetect` | LettuceDetect (Kovács & Recski, 2025) | Fuente de las F1 por tarea del estado del arte. |
| `song2024raghat` | RAG-HAT (Song et al., 2024) | Método SOTA de comparación (cota alta). |
| `ravi2024lynx` | Lynx (Ravi et al., 2024) | Fuente de las accuracy del estado del arte. |

---

## Retiradas del capítulo (motivo: definición de método → marco teórico)

Retiradas explícitamente durante la edición, por petición de la autora, porque son definiciones de
método que van al marco teórico (o que la autora no repetirá allí, pero prefiere no citar aquí):

| Clave | Qué es | Cuándo / motivo |
|---|---|---|
| `cox1958logreg` | Regresión logística (Cox, 1958) | 2026-07. Definición de método (sección logreg). No se re-explica en teórico, pero se retira aquí. |
| `breiman2001rf` | Random Forests (Breiman, 2001) | 2026-07. Definición de método (sección bosque aleatorio). |

---

## En `referencias.bib` pero NO citadas en este capítulo (reserva)

Disponibles por si una afirmación las necesita (la mayoría pertenecen al marco teórico). Cada una,
qué cubre:

| Clave | Cubre |
|---|---|
| `lewis2020rag` | RAG original (Lewis et al., 2020). |
| `ji2023survey` | Survey de alucinaciones en NLG (Ji et al., 2023). |
| `zheng2023judge` | LLM como juez / MT-Bench (Zheng et al., 2023). |
| `salton1988tfidf` | TF-IDF (Salton & Buckley, 1988). |
| `deerwester1990lsa` | LSA (Deerwester et al., 1990). |
| `halko2011svd` | SVD aleatorizado (Halko et al., 2011). |
| `cortes1995svm` | Support-Vector Networks (Cortes & Vapnik, 1995). |
| `cover1967knn` | Nearest Neighbor (Cover & Hart, 1967). |
| `friedman2001gbm` | Gradient boosting (Friedman, 2001). |
| `chen2016xgboost` | XGBoost (Chen & Guestrin, 2016). |
| `fawcett2006roc` | ROC / AUC (Fawcett, 2006). |
| `efron1979bootstrap` | Bootstrap (Efron, 1979). |
| `delong1988` | Comparación de AUCs (DeLong et al., 1988). |
| `platt1999` | Platt scaling / calibración (Platt, 1999). |
| `saerens2002` | Corrección de prior a posteriori (Saerens et al., 2002). |
| `chawla2002smote` | SMOTE (Chawla et al., 2002). |
| `faithful2026` | **Placeholder** (paper sobre la falta de fiabilidad del LLM-juez, PENDIENTE de sustituir). |

---

## Cómo recuperar una cita

1. Escribe `\cite{clave}` donde toque en `resultados.tex`.
2. Recompila: con `\bibliographystyle{plain}` solo aparecen en la bibliografía las que estén citadas.
