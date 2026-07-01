# Evaluación en el test oficial de RAGTruth

n = 2700 respuestas; fracción que alucina = 0.349.

Modelos ajustados por grid search (GroupKFold por `source`) sobre train, evaluados sobre el test oficial. Umbral por Youden.

| modelo        |   AUC-ROC | IC95%          |   AUC-PR |    F1 |   precision |   recall |   bal_acc |   Brier |   ECE |
|:--------------|----------:|:---------------|---------:|------:|------------:|---------:|----------:|--------:|------:|
| xgboost       |     0.821 | [0.805, 0.837] |    0.718 | 0.66  |       0.647 |    0.673 |     0.738 |   0.165 | 0.07  |
| knn           |     0.819 | [0.803, 0.836] |    0.711 | 0.66  |       0.646 |    0.676 |     0.738 |   0.166 | 0.066 |
| random_forest |     0.817 | [0.801, 0.834] |    0.714 | 0.654 |       0.689 |    0.621 |     0.736 |   0.166 | 0.07  |
| svm           |     0.795 | [0.776, 0.813] |    0.683 | 0.66  |       0.625 |    0.7   |     0.737 |   0.175 | 0.074 |
| logreg        |     0.793 | [0.777, 0.810] |    0.658 | 0.65  |       0.572 |    0.751 |     0.725 |   0.187 | 0.114 |

Figuras: `outputs/figures/roc_test.*`, `pr_test.*`.
