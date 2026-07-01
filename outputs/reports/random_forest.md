# random_forest — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.674**
- Mejores hiperparámetros: `{'max_depth': 12, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'n_estimators': 300}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_max_depth | param_max_features   |   param_min_samples_leaf |   param_n_estimators |   mean_test_score |   std_test_score |
|------------------:|:---------------------|-------------------------:|---------------------:|------------------:|-----------------:|
|                12 | 0.5                  |                        5 |                  300 |            0.6736 |           0.0139 |
|                12 | sqrt                 |                        5 |                  300 |            0.6736 |           0.0139 |
|                12 | 0.5                  |                        5 |                  600 |            0.671  |           0.0138 |
|                12 | sqrt                 |                        5 |                  600 |            0.671  |           0.0138 |
|                   | sqrt                 |                        5 |                  300 |            0.6701 |           0.0128 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.803 (IC95% [0.796, 0.810]) · AUC-PR 0.768
- **B** F1 0.680 · precision 0.718 · recall 0.646 · accuracy 0.729 · specificity 0.796
- **C** balanced accuracy 0.721
- **D** Brier 0.179 · ECE 0.012
- Umbral (Youden) 0.482