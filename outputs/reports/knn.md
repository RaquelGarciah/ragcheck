# knn — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.693**
- Mejores hiperparámetros: `{'kneighborsclassifier__n_neighbors': 25, 'kneighborsclassifier__p': 1, 'kneighborsclassifier__weights': 'distance'}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_kneighborsclassifier__n_neighbors |   param_kneighborsclassifier__p | param_kneighborsclassifier__weights   |   mean_test_score |   std_test_score |
|------------------------------------------:|--------------------------------:|:--------------------------------------|------------------:|-----------------:|
|                                        25 |                               1 | distance                              |            0.6934 |           0.0073 |
|                                        25 |                               1 | uniform                               |            0.6928 |           0.0074 |
|                                        25 |                               2 | distance                              |            0.6905 |           0.0066 |
|                                        25 |                               2 | uniform                               |            0.6895 |           0.0077 |
|                                        51 |                               1 | distance                              |            0.6893 |           0.0072 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.814 (IC95% [0.808, 0.821]) · AUC-PR 0.790
- **B** F1 0.701 · precision 0.740 · recall 0.666 · accuracy 0.747 · specificity 0.812
- **C** balanced accuracy 0.739
- **D** Brier 0.173 · ECE 0.024
- Umbral (Youden) 0.479