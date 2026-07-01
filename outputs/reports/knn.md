# knn — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.677**
- Mejores hiperparámetros: `{'kneighborsclassifier__n_neighbors': 51, 'kneighborsclassifier__p': 1, 'kneighborsclassifier__weights': 'distance'}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_kneighborsclassifier__n_neighbors |   param_kneighborsclassifier__p | param_kneighborsclassifier__weights   |   mean_test_score |   std_test_score |
|------------------------------------------:|--------------------------------:|:--------------------------------------|------------------:|-----------------:|
|                                        51 |                               1 | distance                              |            0.6773 |           0.0081 |
|                                       101 |                               2 | distance                              |            0.6767 |           0.0077 |
|                                        51 |                               1 | uniform                               |            0.6767 |           0.0075 |
|                                        25 |                               1 | uniform                               |            0.6763 |           0.0088 |
|                                       101 |                               2 | uniform                               |            0.6758 |           0.0066 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.802 (IC95% [0.795, 0.809]) · AUC-PR 0.769
- **B** F1 0.678 · precision 0.727 · recall 0.636 · accuracy 0.731 · specificity 0.808
- **C** balanced accuracy 0.722
- **D** Brier 0.179 · ECE 0.015
- Umbral (Youden) 0.499