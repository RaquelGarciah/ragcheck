# Evaluación en test oficial — conjunto RFE (11 variables)

Variables: containment, task_QA, task_Summary, task_Data2txt, num_context, answer_len, jaccard, sent_cont_min, novel_bigram, num_overlap, sent_sim_min.

Grid search GroupKFold por `source` (F1); umbral por Youden. n test = 2700.

## Comparativa y ganador

| modelo        |   AUC |    F1 |   precisión |   recall |   accuracy |   bal_acc |   prom_F1_AUC |
|:--------------|------:|------:|------------:|---------:|-----------:|----------:|--------------:|
| xgboost       | 0.837 | 0.688 |       0.685 |    0.691 |      0.781 |     0.76  |         0.763 |
| knn           | 0.826 | 0.681 |       0.665 |    0.698 |      0.771 |     0.754 |         0.754 |
| random_forest | 0.83  | 0.676 |       0.679 |    0.672 |      0.774 |     0.751 |         0.753 |
| svm           | 0.813 | 0.69  |       0.693 |    0.686 |      0.784 |     0.762 |         0.751 |
| logreg        | 0.806 | 0.658 |       0.587 |    0.749 |      0.729 |     0.733 |         0.732 |

**Ganador (promedio F1+AUC): xgboost.**


## Rejilla — logreg (mejor: {'C': 0.3, 'class_weight': 'balanced'})

|    C | class\_weight   |   F1 (CV) |   desv. |
|-----:|:----------------|----------:|--------:|
|  0.3 | balanced        |     0.673 |   0.007 |
|  1   | balanced        |     0.672 |   0.007 |
| 10   | balanced        |     0.672 |   0.006 |
|  3   | balanced        |     0.672 |   0.007 |
|  0.1 | balanced        |     0.672 |   0.007 |
|  0.1 | nan             |     0.663 |   0.007 |
|  0.3 | nan             |     0.661 |   0.006 |
|  1   | nan             |     0.658 |   0.005 |
|  3   | nan             |     0.657 |   0.006 |
| 10   | nan             |     0.657 |   0.006 |

## Rejilla — knn (mejor: {'kneighborsclassifier__n_neighbors': 25, 'kneighborsclassifier__weights': 'distance'})

|   k | pesos    |   F1 (CV) |   desv. |
|----:|:---------|----------:|--------:|
|  25 | distance |     0.685 |   0.011 |
|  35 | distance |     0.684 |   0.013 |
|  35 | uniform  |     0.684 |   0.012 |
|  25 | uniform  |     0.684 |   0.012 |
|  15 | uniform  |     0.684 |   0.01  |
|  15 | distance |     0.683 |   0.009 |
|  11 | distance |     0.681 |   0.003 |
|  11 | uniform  |     0.681 |   0.003 |
|   5 | uniform  |     0.672 |   0.008 |
|   5 | distance |     0.672 |   0.007 |

## Rejilla — svm (mejor: {'svc__C': 10.0, 'svc__gamma': 'scale'})

|   C | gamma   |   F1 (CV) |   desv. |
|----:|:--------|----------:|--------:|
|  10 | scale   |     0.688 |   0.019 |
|  10 | auto    |     0.688 |   0.019 |
| 100 | scale   |     0.682 |   0.019 |
| 100 | auto    |     0.682 |   0.019 |
|   1 | scale   |     0.68  |   0.025 |
|   1 | auto    |     0.68  |   0.025 |

## Rejilla — random_forest (mejor: {'max_depth': 20})

|   prof.\ máx |   F1 (CV) |   desv. |
|-------------:|----------:|--------:|
|           20 |     0.693 |   0.006 |
|              |     0.691 |   0.008 |
|           12 |     0.689 |   0.008 |
|            8 |     0.684 |   0.009 |

## Rejilla — xgboost (mejor: {'learning_rate': 0.05, 'max_depth': 6})

|   tasa aprend. |   prof.\ máx |   F1 (CV) |   desv. |
|---------------:|-------------:|----------:|--------:|
|           0.05 |            6 |     0.692 |   0.01  |
|           0.1  |            4 |     0.692 |   0.009 |
|           0.1  |            6 |     0.692 |   0.01  |
|           0.05 |            4 |     0.691 |   0.01  |
|           0.1  |            3 |     0.69  |   0.011 |
|           0.05 |            3 |     0.689 |   0.009 |
