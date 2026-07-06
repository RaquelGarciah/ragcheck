# Evaluación en test oficial — conjunto RFE (11 variables)

Variables: containment, task_QA, task_Summary, task_Data2txt, num_context, answer_len, jaccard, sent_cont_min, novel_bigram, num_overlap, sent_sim_min.

Grid search GroupKFold por `context` (F1); umbral de Youden fijado sobre las OOF de train (no sobre test). n test = 2700.

## Comparativa y ganador

| modelo        |   AUC |    F1 |   precisión |   recall |   accuracy |   bal_acc |   prom_F1_AUC |
|:--------------|------:|------:|------------:|---------:|-----------:|----------:|--------------:|
| xgboost       | 0.837 | 0.685 |       0.673 |    0.698 |      0.776 |     0.758 |         0.761 |
| random_forest | 0.833 | 0.677 |       0.656 |    0.699 |      0.767 |     0.751 |         0.755 |
| svm           | 0.819 | 0.685 |       0.67  |    0.701 |      0.775 |     0.758 |         0.752 |
| knn           | 0.826 | 0.669 |       0.646 |    0.695 |      0.76  |     0.745 |         0.748 |
| logreg        | 0.798 | 0.654 |       0.594 |    0.726 |      0.731 |     0.73  |         0.726 |

**Ganador (promedio F1+AUC): xgboost.**


## Rejilla — logreg (mejor: {'C': 0.01, 'class_weight': 'balanced', 'penalty': 'l2', 'solver': 'liblinear'})

|      C | class\_weight   | penalty   |   F1 (CV) |   desv. |
|-------:|:----------------|:----------|----------:|--------:|
|   0.01 | balanced        | l2        |     0.675 |   0.003 |
|   0.01 | balanced        | l1        |     0.674 |   0.005 |
|   0.1  | balanced        | l1        |     0.674 |   0.008 |
|  10    | balanced        | l1        |     0.673 |   0.008 |
| 100    | balanced        | l1        |     0.673 |   0.008 |
|   1    | balanced        | l2        |     0.673 |   0.008 |
|   1    | balanced        | l1        |     0.673 |   0.007 |
|  10    | balanced        | l2        |     0.673 |   0.008 |
| 100    | balanced        | l2        |     0.672 |   0.008 |
|   0.1  | balanced        | l2        |     0.672 |   0.008 |
|   0.1  | None            | l1        |     0.662 |   0.007 |
|   0.1  | None            | l2        |     0.662 |   0.005 |
|   0.01 | None            | l1        |     0.66  |   0.006 |
|   1    | None            | l2        |     0.659 |   0.009 |
|   0.01 | None            | l2        |     0.658 |   0.002 |
|   1    | None            | l1        |     0.657 |   0.009 |
| 100    | None            | l2        |     0.657 |   0.008 |
| 100    | None            | l1        |     0.657 |   0.008 |
|  10    | None            | l1        |     0.656 |   0.008 |
|  10    | None            | l2        |     0.656 |   0.008 |

## Rejilla — knn (mejor: {'kneighborsclassifier__n_neighbors': 25, 'kneighborsclassifier__p': 1, 'kneighborsclassifier__weights': 'distance'})

|   k |   p | pesos    |   F1 (CV) |   desv. |
|----:|----:|:---------|----------:|--------:|
|  25 |   1 | distance |     0.689 |   0.008 |
|  15 |   1 | distance |     0.689 |   0.009 |
|  15 |   1 | uniform  |     0.688 |   0.01  |
|  25 |   1 | uniform  |     0.686 |   0.007 |
|  11 |   1 | uniform  |     0.685 |   0.008 |
|  15 |   2 | distance |     0.684 |   0.011 |
|  11 |   1 | distance |     0.684 |   0.006 |
|  25 |   2 | distance |     0.684 |   0.011 |
|  25 |   2 | uniform  |     0.684 |   0.011 |
|  11 |   2 | distance |     0.683 |   0.009 |
|  15 |   2 | uniform  |     0.683 |   0.012 |
|  11 |   2 | uniform  |     0.683 |   0.009 |
|   5 |   1 | uniform  |     0.674 |   0.008 |
|   5 |   1 | distance |     0.673 |   0.008 |
|   5 |   2 | distance |     0.672 |   0.01  |
|   5 |   2 | uniform  |     0.671 |   0.01  |
|   3 |   1 | uniform  |     0.663 |   0.01  |
|   3 |   1 | distance |     0.662 |   0.01  |
|   3 |   2 | uniform  |     0.661 |   0.013 |
|   3 |   2 | distance |     0.66  |   0.013 |

## Rejilla — svm (mejor: {'svc__C': 10.0, 'svc__class_weight': 'balanced', 'svc__gamma': 'scale'})

|     C | class\_weight   | gamma   |   F1 (CV) |   desv. |
|------:|:----------------|:--------|----------:|--------:|
|  10   | balanced        | scale   |     0.699 |   0.021 |
|  10   | balanced        | auto    |     0.699 |   0.021 |
| 100   | balanced        | scale   |     0.696 |   0.027 |
| 100   | balanced        | auto    |     0.696 |   0.027 |
|   1   | balanced        | scale   |     0.692 |   0.027 |
|   1   | balanced        | auto    |     0.692 |   0.027 |
|  10   | None            | scale   |     0.685 |   0.025 |
|  10   | None            | auto    |     0.685 |   0.025 |
| 100   | None            | scale   |     0.682 |   0.027 |
| 100   | None            | auto    |     0.682 |   0.027 |
|   0.1 | balanced        | scale   |     0.681 |   0.027 |
|   0.1 | balanced        | auto    |     0.681 |   0.027 |
|   1   | None            | scale   |     0.675 |   0.023 |
|   1   | None            | auto    |     0.675 |   0.023 |
|   0.1 | None            | scale   |     0.672 |   0.026 |
|   0.1 | None            | auto    |     0.672 |   0.026 |

## Rejilla — random_forest (mejor: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4})

| prof.\ máx   | max\_features   |   mín.\ hoja |   F1 (CV) |   desv. |
|:-------------|:----------------|-------------:|----------:|--------:|
| 20           | log2            |            4 |     0.695 |   0.008 |
| 20           | sqrt            |            4 |     0.695 |   0.008 |
| 20           | log2            |            1 |     0.695 |   0.007 |
| 20           | sqrt            |            1 |     0.695 |   0.007 |
| 15           | sqrt            |            2 |     0.694 |   0.008 |
| 15           | log2            |            2 |     0.694 |   0.008 |
| 15           | log2            |            8 |     0.694 |   0.009 |
| 15           | sqrt            |            8 |     0.694 |   0.009 |
| 15           | log2            |            4 |     0.694 |   0.008 |
| 15           | sqrt            |            4 |     0.694 |   0.008 |
| None         | sqrt            |            4 |     0.694 |   0.007 |
| None         | log2            |            4 |     0.694 |   0.007 |
| 20           | log2            |            8 |     0.693 |   0.009 |
| 20           | sqrt            |            8 |     0.693 |   0.009 |
| None         | sqrt            |            1 |     0.693 |   0.008 |
| None         | log2            |            1 |     0.693 |   0.008 |
| None         | sqrt            |            8 |     0.693 |   0.009 |
| None         | log2            |            8 |     0.693 |   0.009 |
| 20           | sqrt            |            2 |     0.693 |   0.008 |
| 20           | log2            |            2 |     0.693 |   0.008 |
| None         | log2            |            2 |     0.693 |   0.006 |
| None         | sqrt            |            2 |     0.693 |   0.006 |
| 15           | sqrt            |            1 |     0.692 |   0.007 |
| 15           | log2            |            1 |     0.692 |   0.007 |
| 10           | sqrt            |            2 |     0.691 |   0.007 |
| 10           | log2            |            2 |     0.691 |   0.007 |
| 10           | sqrt            |            4 |     0.69  |   0.006 |
| 10           | log2            |            4 |     0.69  |   0.006 |
| 10           | sqrt            |            8 |     0.69  |   0.007 |
| 10           | log2            |            8 |     0.69  |   0.007 |
| 10           | sqrt            |            1 |     0.688 |   0.008 |
| 10           | log2            |            1 |     0.688 |   0.008 |

## Rejilla — xgboost (mejor: {'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8})

|   tasa aprend. |   prof.\ máx |   subsample |   F1 (CV) |   desv. |
|---------------:|-------------:|------------:|----------:|--------:|
|           0.05 |            6 |         0.8 |     0.696 |   0.008 |
|           0.1  |            4 |         0.8 |     0.695 |   0.009 |
|           0.1  |            5 |         0.8 |     0.694 |   0.009 |
|           0.1  |            4 |         1   |     0.694 |   0.009 |
|           0.03 |            6 |         0.8 |     0.694 |   0.009 |
|           0.05 |            6 |         1   |     0.694 |   0.012 |
|           0.1  |            3 |         0.8 |     0.694 |   0.008 |
|           0.2  |            3 |         0.8 |     0.693 |   0.01  |
|           0.2  |            4 |         1   |     0.693 |   0.01  |
|           0.03 |            6 |         1   |     0.692 |   0.01  |
|           0.1  |            6 |         0.8 |     0.692 |   0.007 |
|           0.05 |            5 |         0.8 |     0.692 |   0.011 |
|           0.1  |            3 |         1   |     0.691 |   0.008 |
|           0.05 |            5 |         1   |     0.691 |   0.009 |
|           0.1  |            6 |         1   |     0.691 |   0.011 |
|           0.05 |            4 |         0.8 |     0.691 |   0.008 |
|           0.1  |            5 |         1   |     0.691 |   0.009 |
|           0.2  |            3 |         1   |     0.691 |   0.007 |
|           0.03 |            5 |         1   |     0.691 |   0.011 |
|           0.2  |            4 |         0.8 |     0.69  |   0.006 |
|           0.03 |            5 |         0.8 |     0.69  |   0.007 |
|           0.05 |            4 |         1   |     0.689 |   0.01  |
|           0.2  |            5 |         1   |     0.688 |   0.009 |
|           0.03 |            4 |         0.8 |     0.688 |   0.01  |
|           0.2  |            5 |         0.8 |     0.688 |   0.007 |
|           0.05 |            3 |         0.8 |     0.687 |   0.01  |
|           0.2  |            6 |         1   |     0.687 |   0.009 |
|           0.2  |            6 |         0.8 |     0.685 |   0.007 |
|           0.05 |            3 |         1   |     0.685 |   0.01  |
|           0.03 |            4 |         1   |     0.684 |   0.01  |
|           0.03 |            3 |         0.8 |     0.679 |   0.012 |
|           0.03 |            3 |         1   |     0.678 |   0.008 |
