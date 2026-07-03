# Weak supervision a nivel frase (con los spans de RAGTruth)

Idea: usar los spans anotados como **etiquetas de frase** (no como features —eso
sería fuga), entrenar un clasificador de frases (GroupKFold por source) y agregar
a respuesta. Ataca los falsos negativos por dilución: la frase inventada dentro de
una respuesta fiel deja de diluirse en el promedio.

## Resultados (OOF, xgboost, umbral Youden)

| Modelo | AUC | F1 | recall | prec | Data2txt AUC | QA AUC | Summ AUC |
|---|---|---|---|---|---|---|---|
| base 18 (resp-level) | 0,824 | 0,700 | 0,657 | 0,750 | 0,813 | 0,812 | 0,701 |
| frase→max (reemplazo) | 0,744 | 0,685 | 0,750 | 0,630 | 0,525 | 0,786 | 0,679 |
| híbrido 18 + frase(5) | 0,827 | 0,706 | 0,659 | 0,760 | 0,814 | 0,817 | 0,702 |

## Lecturas

- **frase→max** sube recall (0,66→0,75, QA 0,53→0,78) pero **hunde el AUC**
  (0,824→0,744) y **colapsa Data2txt** (0,525): el `max` es un OR demasiado
  agresivo; en respuestas largas con muchas frases se satura.
- **híbrido** (response-level + estadísticos de frase por stacking OOF) no colapsa
  nada y gana poco: AUC +0,003, F1 +0,006. La feature de frase (`sent_top2`) es la
  #1 en importancia, pero el AUC apenas sube → **es redundante con las features de
  frase artesanales** (`sent_cont_min`, `sent_sim_min`) que no necesitan spans.
- **El recall es una palanca de umbral**, no de features: el salto de recall del
  `max` venía con el colapso del AUC (punto de operación agresivo), reproducible en
  cualquier modelo bajando el umbral.

## Conclusión

La weak supervision **confirma** que la señal de localización es real, pero **no
aporta sobre las features de frase ya existentes**, y añade mucha complejidad
(dataset de frases, mapeo span→frase, modelo de dos etapas, dependencia de las
etiquetas de span). No se adopta como modelo de producción; queda como validación
metodológica (los spans no dan más de lo que ya captura `sent_*_min`).

Scripts: `exploration/weak_supervision_{frase,hibrido}.py`.
