# xgboost — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.695**
- Mejores hiperparámetros: `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 600, 'reg_lambda': 1, 'subsample': 0.8}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_learning_rate |   param_max_depth |   param_n_estimators |   param_reg_lambda |   param_subsample |   mean_test_score |   std_test_score |
|----------------------:|------------------:|---------------------:|-------------------:|------------------:|------------------:|-----------------:|
|                  0.1  |                 3 |                  600 |                  1 |               0.8 |            0.6955 |           0.0106 |
|                  0.03 |                 6 |                  600 |                  1 |               0.8 |            0.6948 |           0.0103 |
|                  0.03 |                 6 |                  300 |                  2 |               0.8 |            0.6945 |           0.0101 |
|                  0.03 |                 6 |                  600 |                  2 |               0.8 |            0.6936 |           0.0094 |
|                  0.1  |                 4 |                  300 |                  2 |               0.8 |            0.6932 |           0.0083 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.815 (IC95% [0.808, 0.822]) · AUC-PR 0.784
- **B** F1 0.696 · precision 0.741 · recall 0.655 · accuracy 0.745 · specificity 0.816
- **C** balanced accuracy 0.736
- **D** Brier 0.173 · ECE 0.023
- Umbral (Youden) 0.502