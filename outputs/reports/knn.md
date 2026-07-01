# knn — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.685**
- Mejores hiperparámetros: `{'kneighborsclassifier__n_neighbors': 51, 'kneighborsclassifier__p': 2, 'kneighborsclassifier__weights': 'distance'}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_kneighborsclassifier__n_neighbors |   param_kneighborsclassifier__p | param_kneighborsclassifier__weights   |   mean_test_score |   std_test_score |
|------------------------------------------:|--------------------------------:|:--------------------------------------|------------------:|-----------------:|
|                                        51 |                               2 | distance                              |            0.6846 |           0.0076 |
|                                       101 |                               2 | distance                              |            0.6832 |           0.0076 |
|                                        25 |                               2 | distance                              |            0.683  |           0.0062 |
|                                        25 |                               2 | uniform                               |            0.6822 |           0.0054 |
|                                        25 |                               1 | distance                              |            0.6814 |           0.0075 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.811 (IC95% [0.805, 0.818]) · AUC-PR 0.783
- **B** F1 0.696 · precision 0.725 · recall 0.668 · accuracy 0.739 · specificity 0.797
- **C** balanced accuracy 0.732
- **D** Brier 0.174 · ECE 0.016
- Umbral (Youden) 0.464