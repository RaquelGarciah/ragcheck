# Análisis de error — Summary como cuello de botella

Umbral (Youden global) = 0.485

## AUC-ROC OOF por tarea

- Data2txt: 0.793
- QA: 0.809
- Summary: 0.702

## Containment medio del texto alucinado vs fuente (por tarea)

- Data2txt: media 0.427, mediana 0.429 (n=8236)
- QA: media 0.482, mediana 0.500 (n=2692)
- Summary: media 0.620, mediana 0.667 (n=1828)

## Composición de tipos de alucinación por tarea (fracción)

| task_type   |   Evident Baseless Info |   Evident Conflict |   Subtle Baseless Info |   Subtle Conflict |
|:------------|------------------------:|-------------------:|-----------------------:|------------------:|
| Data2txt    |                   0.383 |              0.447 |                  0.163 |             0.007 |
| QA          |                   0.522 |              0.146 |                  0.314 |             0.018 |
| Summary     |                   0.514 |              0.343 |                  0.1   |             0.043 |

Nota: Summary es ~86% *evidente* y QA tiene MÁS tipo *sutil* que Summary, pero QA puntúa mejor. Por tanto el tipo anotado NO explica la dificultad de Summary.

## Summary: containment de span en errores vs aciertos

- Falsos negativos (n=1151): span_cont medio 0.641
- Aciertos positivos (n=331): span_cont medio 0.555

## Muestra de falsos negativos de Summary (spans alucinados)

- p=0.28, span_cont=1.00 → «age of 55 | NSYNC and Chris Kirkpatrick,»
- p=0.37, span_cont=1.00 → «majority of black members»
- p=0.16, span_cont=1.00 → «completed the Boston Marathon in 26.2 hours»
- p=0.32, span_cont=1.00 → «two men»
- p=0.23, span_cont=1.00 → «six children.»

## Conclusión

El cuello de botella de Summary NO es el tipo de alucinación (es mayoría *evidente*), sino el **camuflaje léxico**: un resumen reutiliza el vocabulario del documento, así que el span alucinado comparte palabras con la fuente (containment 0.62 vs 0.43 en Data2txt) y el solape léxico no lo distingue del texto fiel. Los falsos negativos son precisamente los spans de mayor containment. Es el techo estructural de las features de superficie: para separar en Summary hace falta señal *relacional* (consistencia número-unidad, entidad-relación, polaridad), no más solape de vocabulario.
