# svm — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.690**
- Mejores hiperparámetros: `{'svc__C': 10.0, 'svc__gamma': 0.1}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_svc__C | param_svc__gamma   |   mean_test_score |   std_test_score |
|---------------:|:-------------------|------------------:|-----------------:|
|             10 | 0.1                |            0.6904 |           0.0122 |
|             10 | scale              |            0.6894 |           0.0123 |
|              1 | 0.1                |            0.6835 |           0.0106 |
|              1 | 1.0                |            0.6834 |           0.0089 |
|              1 | scale              |            0.6796 |           0.0092 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.799 (IC95% [0.791, 0.806]) · AUC-PR 0.765
- **B** F1 0.702 · precision 0.728 · recall 0.677 · accuracy 0.743 · specificity 0.797
- **C** balanced accuracy 0.737
- **D** Brier 0.179 · ECE 0.022
- Umbral (Youden) 0.451