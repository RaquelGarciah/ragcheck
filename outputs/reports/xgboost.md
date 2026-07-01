# xgboost — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.699**
- Mejores hiperparámetros: `{'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 600, 'reg_lambda': 2, 'subsample': 0.8}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_learning_rate |   param_max_depth |   param_n_estimators |   param_reg_lambda |   param_subsample |   mean_test_score |   std_test_score |
|----------------------:|------------------:|---------------------:|-------------------:|------------------:|------------------:|-----------------:|
|                  0.03 |                 6 |                  600 |                  2 |               0.8 |            0.6992 |           0.0097 |
|                  0.1  |                 3 |                  600 |                  2 |               0.8 |            0.6992 |           0.011  |
|                  0.1  |                 4 |                  300 |                  2 |               0.8 |            0.699  |           0.0111 |
|                  0.03 |                 6 |                  600 |                  2 |               1   |            0.6983 |           0.0064 |
|                  0.03 |                 6 |                  300 |                  2 |               0.8 |            0.6983 |           0.011  |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.823 (IC95% [0.816, 0.829]) · AUC-PR 0.797
- **B** F1 0.706 · precision 0.736 · recall 0.679 · accuracy 0.749 · specificity 0.804
- **C** balanced accuracy 0.742
- **D** Brier 0.169 · ECE 0.022
- Umbral (Youden) 0.475