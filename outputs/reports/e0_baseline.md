# E0 — Línea base: comparativa de modelos

n = 15090 respuestas, 2514 documentos.

Validación GroupKFold por `source`. Umbral por índice de Youden.

| modelo                      |   AUC-ROC | IC95%          |   AUC-PR |    F1 |   precision |   recall |   bal_acc |   Brier |     ECE |
|:----------------------------|----------:|:---------------|---------:|------:|------------:|---------:|----------:|--------:|--------:|
| logreg                      |     0.789 | [0.782, 0.796] |    0.749 | 0.695 |       0.653 |    0.744 |     0.713 |   0.186 |   0.021 |
| xgboost                     |     0.819 | [0.812, 0.826] |    0.788 | 0.697 |       0.739 |    0.659 |     0.736 |   0.171 |   0.006 |
| random_forest               |     0.813 | [0.806, 0.820] |    0.782 | 0.695 |       0.732 |    0.662 |     0.734 |   0.174 |   0.018 |
| svm                         |     0.799 | [0.792, 0.806] |    0.769 | 0.702 |       0.707 |    0.698 |     0.733 |   0.18  |   0.028 |
| knn                         |     0.801 | [0.795, 0.809] |    0.755 | 0.696 |       0.69  |    0.702 |     0.724 |   0.18  |   0.031 |
| baseline: solo containment  |     0.761 | [0.753, 0.768] |    0.662 | 0.667 |       0.669 |    0.664 |     0.7   |   0.199 |   0.039 |
| baseline: clase mayoritaria |     0.5   | --             |    0.445 | 0     |       0     |    0     |     0.5   |   0.247 | nan     |

Figuras: `outputs/figures/roc_modelos.*`, `pr_modelos.*`.
