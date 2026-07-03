# Eficacia del detector por LLM generador

RAGTruth anota qué LLM generó cada respuesta (columna `model`). Aquí, la eficacia
del detector definitivo (top-7, xgboost, umbral Youden global 0,497) desglosada
por generador, en el test oficial.

## Métricas por generador

| Generador | n | prev | AUC | AUC-PR | F1 | acc | recall | prec |
|---|---|---|---|---|---|---|---|---|
| llama-2-13b | 450 | 0,46 | **0,857** | 0,859 | **0,790** | 0,811 | 0,773 | 0,808 |
| llama-2-70b | 450 | 0,38 | 0,824 | 0,766 | 0,718 | 0,782 | 0,731 | 0,706 |
| mistral-7b | 450 | 0,56 | 0,817 | 0,826 | 0,675 | 0,696 | 0,566 | 0,835 |
| gpt-4 | 450 | **0,09** | 0,812 | 0,257 | **0,314** | 0,844 | 0,381 | 0,267 |
| gpt-3.5 | 450 | **0,10** | 0,795 | 0,294 | 0,330 | 0,864 | 0,326 | 0,333 |
| llama-2-7b | 450 | 0,50 | 0,765 | 0,785 | 0,693 | 0,689 | 0,699 | 0,687 |
| **GLOBAL** | 2700 | 0,35 | 0,835 | 0,750 | 0,676 | 0,781 | 0,653 | 0,700 |

## Lecturas

1. **La discriminación (AUC) es robusta ante el generador: 0,77–0,86.** El detector
   no está sobreajustado a ningún LLM; funciona venga el texto de GPT-4 o de un
   Llama-7B.
2. **El F1 lo manda la tasa de alucinación de cada modelo, no el detector.**
   GPT-4 y GPT-3.5 casi no alucinan (9–10%) → con pocos positivos el F1 se desploma
   (0,31–0,33) aunque el AUC sea 0,81/0,80. Los modelos abiertos alucinan mucho
   (Mistral 56%, Llama 46–50%) → F1 bueno (0,68–0,79).

## El F1 bajo en GPT-4 NO se arregla con el umbral

Mejor F1 posible por generador (optimizando el umbral sobre su propio test, cota
superior que "hace trampa"):

| Generador | prev | AUC-PR | F1 (umbral global) | F1 máximo posible |
|---|---|---|---|---|
| gpt-4 | 0,09 | 0,257 | 0,314 | **0,367** |
| gpt-3.5 | 0,10 | 0,294 | 0,330 | 0,364 |
| llama-70b | 0,38 | 0,766 | 0,718 | 0,721 |
| llama-13b | 0,46 | 0,859 | 0,790 | 0,795 |
| mistral | 0,56 | 0,826 | 0,675 | 0,808 |

En GPT-4, aun con el mejor umbral posible, el F1 sube a 0,37 y ahí se queda: el
techo lo pone la curva PR (AUC-PR 0,26), no el corte. El umbral desliza por la
curva; no la levanta. En los generadores de alta prevalencia el umbral global ya
es casi óptimo (llama-13b 0,790 vs 0,795 máximo).

## Titular para la memoria

El detector es **más útil justo donde más falta hace**: en los LLM abiertos y
baratos que alucinan mucho (F1 0,7–0,8). En GPT-4 el F1 bajo no es un fallo del
detector, sino que GPT-4 apenas alucina —y aun así lo rankea con AUC 0,81. No
sobre-interpretar las diferencias de AUC como "GPT-4 alucina más sutil": son
modestas y están confundidas con el mix de tareas de cada modelo.

Scripts: `exploration/{efficacy_by_generator,threshold_ceiling_by_generator}.py`.
