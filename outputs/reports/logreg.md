# logreg — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.677**
- Mejores hiperparámetros: `{'C': 1.0, 'class_weight': 'balanced', 'solver': 'lbfgs'}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_C | param_class_weight   | param_solver   |   mean_test_score |   std_test_score |
|----------:|:---------------------|:---------------|------------------:|-----------------:|
|       1   | balanced             | lbfgs          |            0.6773 |           0.0075 |
|      10   | balanced             | lbfgs          |            0.6773 |           0.0075 |
|      10   | balanced             | liblinear      |            0.6772 |           0.0076 |
|       1   | balanced             | liblinear      |            0.6771 |           0.0073 |
|       0.1 | balanced             | lbfgs          |            0.6736 |           0.0066 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.769 (IC95% [0.762, 0.776]) · AUC-PR 0.713
- **B** F1 0.677 · precision 0.645 · recall 0.713 · accuracy 0.697 · specificity 0.685
- **C** balanced accuracy 0.699
- **D** Brier 0.196 · ECE 0.043
- Umbral (Youden) 0.505