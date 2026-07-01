# E0 — Línea base: comparativa de modelos

n = 15090 respuestas, 2514 documentos.

Validación GroupKFold por `source`. Umbral por índice de Youden.

| modelo                      |   AUC-ROC | IC95%          |   AUC-PR |    F1 |   precision |   recall |   bal_acc |   Brier |     ECE |
|:----------------------------|----------:|:---------------|---------:|------:|------------:|---------:|----------:|--------:|--------:|
| logreg                      |     0.79  | [0.783, 0.797] |    0.752 | 0.688 |       0.659 |    0.721 |     0.71  |   0.186 |   0.027 |
| xgboost                     |     0.824 | [0.818, 0.831] |    0.798 | 0.7   |       0.75  |    0.657 |     0.741 |   0.168 |   0.01  |
| random_forest               |     0.819 | [0.813, 0.826] |    0.795 | 0.692 |       0.757 |    0.638 |     0.737 |   0.17  |   0.02  |
| svm                         |     0.807 | [0.801, 0.814] |    0.778 | 0.708 |       0.718 |    0.698 |     0.739 |   0.175 |   0.021 |
| knn                         |     0.806 | [0.799, 0.813] |    0.765 | 0.702 |       0.692 |    0.712 |     0.729 |   0.177 |   0.028 |
| baseline: solo containment  |     0.761 | [0.753, 0.768] |    0.662 | 0.667 |       0.669 |    0.664 |     0.7   |   0.199 |   0.039 |
| baseline: clase mayoritaria |     0.5   | --             |    0.445 | 0     |       0     |    0     |     0.5   |   0.247 | nan     |

Figuras: `outputs/figures/roc_modelos.*`, `pr_modelos.*`.
