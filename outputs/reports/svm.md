# svm — grid search y evaluación

- Mejor F1 (CV GroupKFold por `source`): **0.698**
- Mejores hiperparámetros: `{'svc__C': 10.0, 'svc__gamma': 'scale'}`

## Rejilla de hiperparámetros (top-5 por F1)

|   param_svc__C | param_svc__gamma   |   mean_test_score |   std_test_score |
|---------------:|:-------------------|------------------:|-----------------:|
|           10   | scale              |            0.698  |           0.0087 |
|           10   | 0.1                |            0.6955 |           0.0095 |
|            1   | 0.1                |            0.694  |           0.0105 |
|            0.5 | 0.1                |            0.6885 |           0.0077 |
|            1   | scale              |            0.6867 |           0.0069 |


## Métricas del modelo ajustado (secciones A–F)

- **A** AUC-ROC 0.808 (IC95% [0.801, 0.815]) · AUC-PR 0.779
- **B** F1 0.715 · precision 0.712 · recall 0.718 · accuracy 0.745 · specificity 0.767
- **C** balanced accuracy 0.743
- **D** Brier 0.174 · ECE 0.020
- Umbral (Youden) 0.395