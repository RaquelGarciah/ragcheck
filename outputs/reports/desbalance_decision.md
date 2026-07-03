# Tratamiento del desbalance de clases — decisión del punto de operación

Fuente única citable de las tablas de la sesión sobre desbalance. Protocolo honesto:
umbrales y priors de referencia se fijan en el OOF de `GroupKFold` por `source` sobre
train; la evaluación es en el **test oficial** de RAGTruth (n=2700, prev=0,349),
intacto. Modelo: XGBoost (semilla 42). Reproducible con
`exploration/{imbalance_thresholds,resampling_check,smote_check,figuras_informe}.py`.

## 1. El desbalance es heterogéneo por tarea y se desplaza train→test

| Tarea | prev. train | prev. test |
|---|---|---|
| Data2txt | 0,694 | 0,643 |
| QA | 0,311 | 0,178 |
| Summary | 0,311 | 0,227 |
| GLOBAL | 0,445 | 0,349 |

Un único umbral no puede ser óptimo para prevalencias tan distintas. QA separa muy
bien (AUC 0,822) pero un umbral global calla sus positivos (recall 0,54).

## 2. Estrategias de umbral (test oficial)

Umbrales fijados en el OOF de train: Youden global = 0,487; por tarea max-F1
= {Data2txt 0,49, QA 0,313, Summary 0,308}; por tarea max-F2 = {Data2txt 0,151,
QA 0,112, Summary 0,129}.

| Estrategia | ámbito | F1 | precisión | recall | accuracy | FN | FP |
|---|---|---|---|---|---|---|---|
| **Youden global** | GLOBAL (pooled) | 0,682 | 0,684 | 0,680 | 0,779 | 302 | 296 |
| | Data2txt | 0,821 | 0,765 | 0,884 | 0,751 | 67 | 157 |
| | QA | 0,513 | 0,486 | 0,544 | 0,817 | 73 | 92 |
| | Summary | 0,287 | 0,472 | 0,206 | 0,768 | 162 | 47 |
| **Por tarea (max-F1)** | GLOBAL (pooled) | 0,656 | 0,547 | 0,821 | 0,700 | **169** | 642 |
| | Data2txt | 0,821 | 0,766 | 0,884 | 0,752 | 67 | 156 |
| | QA | 0,492 | 0,357 | 0,794 | 0,709 | **33** | 229 |
| | Summary | **0,453** | 0,344 | 0,662 | 0,638 | **69** | 257 |
| **Por tarea (max-F2)** | GLOBAL (pooled) | 0,577 | 0,409 | 0,982 | 0,497 | 17 | 1340 |
| | Data2txt | 0,786 | 0,648 | 1,000 | 0,650 | 0 | 315 |
| | QA | 0,391 | 0,244 | 0,981 | 0,457 | 3 | 486 |
| | Summary | 0,407 | 0,261 | 0,931 | 0,386 | 14 | 539 |

F1 global: **pooled** (micro) y **macro** (media por tarea).

| Estrategia | F1 pooled | F1 macro |
|---|---|---|
| Youden global | 0,682 | 0,540 |
| Por tarea (max-F1) | 0,656 | **0,589** |
| Por tarea (max-F2) | 0,577 | 0,528 |

El umbral por tarea (max-F1) recorta los falsos negativos de 302 a **169 (−44%)**,
sube el F1 de Summary de 0,287 a **0,453** y el F1 macro de 0,540 a 0,589, a costa de
precisión (más falsos positivos) y de bajar el F1 pooled a 0,656. F2 es demasiado
agresivo: recall casi 1 pero la accuracy se hunde.

## 3. Corrección de prior (Saerens et al., 2002): negativo

El EM que estima el prior del test sin etiquetas colapsa a valores degenerados,
incluso sobre probabilidades calibradas por isotónica:

| Tarea | prior EM estimado | prior real |
|---|---|---|
| Data2txt | 1,000 | 0,643 |
| QA | 0,000 | 0,178 |
| Summary | 0,000 | 0,227 |

Con priors 0/1 la decisión colapsa (todo alucina en Data2txt, nada en QA/Summary).
Las distribuciones de score se solapan demasiado para que el EM identifique la
proporción de mezcla. Descartada.

## 4. Remuestreo y coste-sensible: la frontera no se mueve

AUC-PR por tarea (invariante al umbral: mide la separación real) según cómo se trate
el desbalance en el entrenamiento:

| Estrategia | Data2txt | QA | Summary |
|---|---|---|---|
| base (sin tratar) | 0,858 | 0,528 | 0,410 |
| oversample 50/50 por tarea | 0,853 | 0,532 | 0,390 |
| undersample 50/50 por tarea | 0,854 | 0,543 | 0,409 |
| SMOTE 50/50 por tarea | 0,855 | 0,533 | 0,357 |
| coste-sensible (`scale_pos_weight`) | 0,857 | 0,528 | 0,403 |

Ninguna variante mueve la frontera por tarea; SMOTE la empeora en Summary
(0,410 → 0,357). El AUC-PR global cae con el remuestreo por tarea (0,752 → 0,67)
porque distorsiona el ranking entre tareas. Reequilibrar clases no crea señal.

### SMOTE dominado por el umbral por tarea

A umbral Youden global, SMOTE parece ayudar en Summary (F1 0,287 → 0,429, FN 162 →
97), pero es puro deslizamiento del punto de operación sobre una frontera peor. El
modelo limpio con umbral por tarea lo domina:

| Vía | Summary F1 | Summary FN | Summary accuracy | AUC-PR |
|---|---|---|---|---|
| **base + umbral por tarea** | **0,453** | 69 | 0,638 | **0,410** |
| SMOTE + Youden global | 0,429 | 97 | 0,683 | 0,357 |
| SMOTE + umbral por tarea | 0,435 | 35 | 0,512 | 0,357 |

## 5. Decisión

**Punto de operación oficial: umbral por tarea (max-F1) fijado en el OOF de train.**
Recupera el 44% de los falsos negativos y hace competitivo el F1 de Summary sin tocar
las features ni el modelo. Se descartan, con prueba: la corrección de prior de
Saerens (colapso del EM), el remuestreo over/under/SMOTE y el aprendizaje
coste-sensible (frontera plana). Se reportan siempre F1 pooled y F1 macro, y las
métricas por tarea, para no esconder el efecto de la prevalencia.
