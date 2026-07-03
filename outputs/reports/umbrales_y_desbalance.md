# Punto de operación: umbrales y desbalance

Cómo se elige el umbral del detector y por qué balancear/umbral-por-tarea no
aportan. Modelo: xgboost global; evaluación en **test oficial** a prevalencia
natural. Umbrales fijados en la OOF de train (GroupKFold por `source`).

## 1. Umbral global, tres reglas (top-7, top-10, 18)

| Set | Umbral | thr | F1 | acc | recall | prec | bal_acc |
|---|---|---|---|---|---|---|---|
| **top-7** (AUC 0,835) | **Youden** | 0,497 | **0,676** | **0,781** | 0,653 | 0,700 | 0,751 |
| | max-F1 | 0,375 | 0,667 | 0,717 | 0,811 | 0,566 | 0,739 |
| | break-even | 0,435 | 0,673 | 0,756 | 0,720 | 0,632 | 0,747 |
| **top-10** (AUC 0,835) | Youden | 0,448 | 0,673 | 0,757 | 0,716 | 0,636 | 0,748 |
| | max-F1 | 0,370 | 0,668 | 0,717 | 0,813 | 0,566 | 0,739 |
| | break-even | 0,440 | 0,670 | 0,751 | 0,722 | 0,624 | 0,744 |
| **18** (AUC 0,837) | Youden | 0,487 | 0,682 | 0,779 | 0,680 | 0,684 | 0,756 |
| | max-F1 | 0,360 | 0,677 | 0,723 | 0,831 | 0,571 | 0,748 |
| | break-even | 0,440 | 0,670 | 0,753 | 0,719 | 0,628 | 0,745 |

**Decisión: Youden global.** En test da a la vez el mejor F1 y la mejor
accuracy. El max-F1 se elige en train (prevalencia 0,445) y no generaliza al test
(0,349): baja demasiado el corte, hunde la precisión. Reglas alternativas por si
se prioriza recall: max-F1 (recall ~0,81) o break-even (equilibrado).

## 2. Umbral por tarea: NO ayuda (y una regla se rompe)

Umbral calibrado por tarea (3 reglas) sobre el modelo global top-7, test:

- **F1 agregado con umbral por tarea: 0,61–0,64**, frente a **0,676 con umbral
  global único**. Calibrar por tarea RESTA ~7 puntos (cada umbral se sobreajusta a
  la prevalencia de train de su tarea).
- **El break-even se ROMPE en Summary**: elige 0,905 en train → en test predice 0
  positivos → **F1 = 0**. Curva PR plana + baja prevalencia = punto P=R degenerado.
- Motivo de fondo: el one-hot (`task_QA`, `task_Summary`) ya mete la conciencia de
  tarea en las probabilidades → el umbral por tarea es redundante y frágil.

## 3. Balancear a 50/50 = mover el umbral (demostrado)

Balancear (SMOTE, undersampling, `scale_pos_weight`) **no crea señal**: reequilibra
el prior, que es deslizarse por la MISMA curva PR. Prueba en test (top-7):

| Config | AUC | **AUC-PR** | thr | F1 | recall | prec |
|---|---|---|---|---|---|---|
| base (natural, Youden) | 0,835 | **0,750** | 0,497 | 0,676 | 0,653 | 0,700 |
| scale_pos_weight (~50/50) @0,5 | 0,834 | **0,749** | 0,500 | 0,669 | 0,699 | 0,642 |
| undersample 50/50 @0,5 | 0,834 | **0,750** | 0,500 | 0,675 | 0,716 | 0,639 |
| base con umbral 0,46 (=recall del balanceado) | 0,835 | **0,750** | 0,455 | 0,675 | 0,698 | 0,653 |

- **El AUC-PR no se mueve (0,749–0,750).** Balancear no levanta la curva; solo
  sube recall y baja precisión (desliza el punto).
- **Equivalencia demostrada:** el modelo base con umbral 0,46 (igualando el recall
  del balanceado) reproduce al `scale_pos_weight`@0,5. Balancear ≡ bajar el umbral.

**Por qué es una técnica válida en general (pero no aquí):** (1) da un punto de
operación sensato con el corte 0,5 por defecto, útil si no ajustas umbral —nosotras
sí lo ajustamos (Youden); (2) en desbalance *severo* con modelos regularizados,
rebalancear cambia la frontera aprendida y puede levantar la curva un poco —nuestro
desbalance global es leve (44,5%, `scale_pos_weight`=1,25), de ahí AUC invariante.
El desbalance real es *por tarea*, pero balancear por tarea = umbral por tarea (§2),
que no generaliza; y el techo de Summary es de señal, no de balance.

## Conclusión

Umbral **global por Youden**. Umbral por tarea, clasificador por tarea y
remuestreo **deslizan por la curva PR sin levantarla** y no generalizan
(prevalencia train≠test). El recall se gestiona, si se quiere, moviendo
conscientemente el umbral global.

Scripts: `exploration/{global_thresholds,threshold_calibration_per_task,
resampling_vs_threshold}.py`. Figuras `pr_top7`, `pr_top10`.
