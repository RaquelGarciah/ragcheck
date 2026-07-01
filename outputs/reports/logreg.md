# logreg — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.692**
- Mejores hiperparámetros: `{'C': 10.0, 'class_weight': 'balanced', 'solver': 'liblinear'}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_C | param_class_weight   | param_solver   |   mean_test_score |   std_test_score |
|----------:|:---------------------|:---------------|------------------:|-----------------:|
|      10   | balanced             | liblinear      |            0.6915 |           0.0068 |
|       1   | balanced             | liblinear      |            0.6912 |           0.0075 |
|      10   | balanced             | lbfgs          |            0.691  |           0.0072 |
|       1   | balanced             | lbfgs          |            0.6909 |           0.007  |
|       0.1 | balanced             | lbfgs          |            0.6806 |           0.0069 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.790 (IC95% [0.783, 0.797]) · AUC-PR 0.747
- **B** F1 0.695 · precision 0.650 · recall 0.746 · accuracy 0.708 · specificity 0.678
- **C** balanced accuracy 0.712
- **D** Brier 0.188 · ECE 0.044
- Umbral (Youden) 0.485