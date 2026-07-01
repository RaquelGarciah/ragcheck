# random_forest — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.695**
- Mejores hiperparámetros: `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'n_estimators': 600}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_max_depth | param_max_features   |   param_min_samples_leaf |   param_n_estimators |   mean_test_score |   std_test_score |
|------------------:|:---------------------|-------------------------:|---------------------:|------------------:|-----------------:|
|                   | sqrt                 |                        5 |                  600 |            0.6945 |           0.0069 |
|                   | sqrt                 |                        5 |                  300 |            0.6934 |           0.0059 |
|                12 | 0.5                  |                        1 |                  300 |            0.6929 |           0.006  |
|                   | sqrt                 |                        1 |                  600 |            0.6928 |           0.0076 |
|                   | 0.5                  |                        5 |                  600 |            0.6927 |           0.0076 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.823 (IC95% [0.817, 0.830]) · AUC-PR 0.798
- **B** F1 0.712 · precision 0.717 · recall 0.706 · accuracy 0.745 · specificity 0.776
- **C** balanced accuracy 0.741
- **D** Brier 0.168 · ECE 0.016
- Umbral (Youden) 0.448