# E0 — Línea base: comparativa de modelos

n = 15090 respuestas, 2514 documentos.

Validación GroupKFold por `source`. Umbral por índice de Youden.

| modelo                      |   AUC-ROC | IC95%          |   AUC-PR |    F1 |   precision |   recall |   bal_acc |   Brier |     ECE |
|:----------------------------|----------:|:---------------|---------:|------:|------------:|---------:|----------:|--------:|--------:|
| logreg                      |     0.769 | [0.762, 0.776] |    0.713 | 0.682 |       0.636 |    0.736 |     0.699 |   0.194 |   0.018 |
| xgboost                     |     0.804 | [0.798, 0.811] |    0.77  | 0.695 |       0.682 |    0.709 |     0.722 |   0.178 |   0.013 |
| random_forest               |     0.788 | [0.781, 0.796] |    0.751 | 0.673 |       0.691 |    0.656 |     0.71  |   0.186 |   0.034 |
| svm                         |     0.792 | [0.785, 0.799] |    0.754 | 0.695 |       0.687 |    0.703 |     0.723 |   0.185 |   0.029 |
| knn                         |     0.788 | [0.781, 0.795] |    0.737 | 0.671 |       0.706 |    0.639 |     0.713 |   0.186 |   0.038 |
| baseline: solo containment  |     0.761 | [0.753, 0.768] |    0.662 | 0.667 |       0.669 |    0.664 |     0.7   |   0.199 |   0.039 |
| baseline: clase mayoritaria |     0.5   | --             |    0.445 | 0     |       0     |    0     |     0.5   |   0.247 | nan     |

Figuras: `outputs/figures/roc_modelos.*`, `pr_modelos.*`.
