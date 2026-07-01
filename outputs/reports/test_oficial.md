# Evaluación en el test oficial de RAGTruth

n = 2700 respuestas; fracción que alucina = 0.349.

Modelos ajustados por grid search (GroupKFold por `source`) sobre train, evaluados sobre el test oficial. Umbral por Youden.

| modelo        |   AUC-ROC | IC95%          |   AUC-PR |    F1 |   precision |   recall |   bal_acc |   Brier |   ECE |
|:--------------|----------:|:---------------|---------:|------:|------------:|---------:|----------:|--------:|------:|
| random_forest |     0.836 | [0.820, 0.851] |    0.745 | 0.68  |       0.651 |    0.71  |     0.753 |   0.158 | 0.069 |
| xgboost       |     0.835 | [0.819, 0.849] |    0.748 | 0.684 |       0.698 |    0.671 |     0.758 |   0.157 | 0.063 |
| knn           |     0.826 | [0.810, 0.841] |    0.735 | 0.678 |       0.628 |    0.737 |     0.752 |   0.16  | 0.065 |
| svm           |     0.819 | [0.801, 0.837] |    0.732 | 0.692 |       0.7   |    0.685 |     0.764 |   0.16  | 0.068 |
| logreg        |     0.816 | [0.800, 0.832] |    0.711 | 0.67  |       0.643 |    0.7   |     0.746 |   0.177 | 0.111 |

Figuras: `outputs/figures/roc_test.*`, `pr_test.*`.
