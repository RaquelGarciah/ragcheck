# Evaluación en el test oficial de RAGTruth

n = 2700 respuestas; fracción que alucina = 0.349.

Modelos ajustados por grid search (GroupKFold por `source`) sobre train, evaluados sobre el test oficial. Umbral por Youden.

| modelo        |   AUC-ROC | IC95%          |   AUC-PR |    F1 |   precision |   recall |   bal_acc |   Brier |   ECE |
|:--------------|----------:|:---------------|---------:|------:|------------:|---------:|----------:|--------:|------:|
| random_forest |     0.83  | [0.813, 0.845] |    0.733 | 0.676 |       0.656 |    0.698 |     0.751 |   0.16  | 0.068 |
| xgboost       |     0.828 | [0.811, 0.843] |    0.727 | 0.676 |       0.637 |    0.721 |     0.75  |   0.162 | 0.066 |
| knn           |     0.826 | [0.811, 0.842] |    0.724 | 0.678 |       0.595 |    0.787 |     0.75  |   0.161 | 0.058 |
| logreg        |     0.817 | [0.800, 0.832] |    0.706 | 0.671 |       0.615 |    0.738 |     0.745 |   0.177 | 0.11  |
| svm           |     0.805 | [0.787, 0.823] |    0.697 | 0.686 |       0.67  |    0.702 |     0.758 |   0.166 | 0.071 |

Figuras: `outputs/figures/roc_test.*`, `pr_test.*`.
