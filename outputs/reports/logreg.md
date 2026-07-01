# logreg — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.689**
- Mejores hiperparámetros: `{'C': 1.0, 'class_weight': 'balanced', 'solver': 'liblinear'}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_C | param_class_weight   | param_solver   |   mean_test_score |   std_test_score |
|----------:|:---------------------|:---------------|------------------:|-----------------:|
|       1   | balanced             | liblinear      |            0.6895 |           0.0055 |
|       1   | balanced             | lbfgs          |            0.6891 |           0.0058 |
|      10   | balanced             | lbfgs          |            0.6888 |           0.0078 |
|      10   | balanced             | liblinear      |            0.688  |           0.0077 |
|       0.1 | balanced             | lbfgs          |            0.6816 |           0.0063 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.790 (IC95% [0.783, 0.797]) · AUC-PR 0.752
- **B** F1 0.692 · precision 0.653 · recall 0.735 · accuracy 0.708 · specificity 0.686
- **C** balanced accuracy 0.711
- **D** Brier 0.188 · ECE 0.049
- Umbral (Youden) 0.491