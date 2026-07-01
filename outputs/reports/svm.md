# svm — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.675**
- Mejores hiperparámetros: `{'svc__C': 1.0, 'svc__gamma': 1.0}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_svc__C | param_svc__gamma   |   mean_test_score |   std_test_score |
|---------------:|:-------------------|------------------:|-----------------:|
|            1   | 1.0                |            0.6755 |           0.0066 |
|           10   | scale              |            0.6748 |           0.0099 |
|            0.5 | 1.0                |            0.6745 |           0.0087 |
|            1   | scale              |            0.6708 |           0.0077 |
|           10   | 0.1                |            0.6707 |           0.0077 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.776 (IC95% [0.769, 0.784]) · AUC-PR 0.733
- **B** F1 0.693 · precision 0.694 · recall 0.692 · accuracy 0.727 · specificity 0.755
- **C** balanced accuracy 0.723
- **D** Brier 0.187 · ECE 0.027
- Umbral (Youden) 0.392