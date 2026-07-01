# random_forest — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.689**
- Mejores hiperparámetros: `{'max_depth': 12, 'max_features': 0.5, 'min_samples_leaf': 5, 'n_estimators': 600}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_max_depth | param_max_features   |   param_min_samples_leaf |   param_n_estimators |   mean_test_score |   std_test_score |
|------------------:|:---------------------|-------------------------:|---------------------:|------------------:|-----------------:|
|                12 | 0.5                  |                        5 |                  600 |            0.6892 |           0.006  |
|                   | sqrt                 |                        1 |                  300 |            0.6891 |           0.0089 |
|                12 | 0.5                  |                        5 |                  300 |            0.6891 |           0.0072 |
|                   | sqrt                 |                        5 |                  600 |            0.6891 |           0.0094 |
|                   | 0.5                  |                        5 |                  300 |            0.6891 |           0.0071 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.817 (IC95% [0.810, 0.824]) · AUC-PR 0.786
- **B** F1 0.709 · precision 0.712 · recall 0.705 · accuracy 0.742 · specificity 0.771
- **C** balanced accuracy 0.738
- **D** Brier 0.172 · ECE 0.017
- Umbral (Youden) 0.446