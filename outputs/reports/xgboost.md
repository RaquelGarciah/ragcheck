# xgboost — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.675**
- Mejores hiperparámetros: `{'learning_rate': 0.03, 'max_depth': 4, 'n_estimators': 300, 'reg_lambda': 1, 'subsample': 0.8}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_learning_rate |   param_max_depth |   param_n_estimators |   param_reg_lambda |   param_subsample |   mean_test_score |   std_test_score |
|----------------------:|------------------:|---------------------:|-------------------:|------------------:|------------------:|-----------------:|
|                  0.03 |                 4 |                  300 |                  1 |               0.8 |            0.6748 |           0.0126 |
|                  0.03 |                 4 |                  300 |                  2 |               0.8 |            0.6743 |           0.012  |
|                  0.03 |                 4 |                  300 |                  1 |               1   |            0.6737 |           0.0111 |
|                  0.03 |                 3 |                  600 |                  1 |               1   |            0.6736 |           0.0103 |
|                  0.03 |                 6 |                  300 |                  2 |               0.8 |            0.6735 |           0.0122 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.806 (IC95% [0.799, 0.813]) · AUC-PR 0.771
- **B** F1 0.680 · precision 0.721 · recall 0.643 · accuracy 0.730 · specificity 0.800
- **C** balanced accuracy 0.722
- **D** Brier 0.177 · ECE 0.006
- Umbral (Youden) 0.486