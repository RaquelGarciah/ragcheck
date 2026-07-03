# Evaluación en test oficial — conjunto RFE (11 variables)

Variables: containment, task_QA, task_Summary, task_Data2txt, num_context, answer_len, jaccard, sent_cont_min, novel_bigram, num_overlap, sent_sim_min.

n test = 2700, prevalencia = 0.349. Grid search GroupKFold por `source` (F1); umbral por Youden.

| modelo        |   AUC |    F1 |   precision |   recall |   accuracy |   bal_acc |   prom_F1_AUC |
|:--------------|------:|------:|------------:|---------:|-----------:|----------:|--------------:|
| xgboost       | 0.837 | 0.688 |       0.685 |    0.691 |      0.781 |     0.76  |         0.763 |
| knn           | 0.826 | 0.681 |       0.665 |    0.698 |      0.771 |     0.754 |         0.754 |
| random_forest | 0.83  | 0.676 |       0.679 |    0.672 |      0.774 |     0.751 |         0.753 |
| svm           | 0.813 | 0.69  |       0.693 |    0.686 |      0.784 |     0.762 |         0.751 |
| logreg        | 0.806 | 0.658 |       0.587 |    0.749 |      0.729 |     0.733 |         0.732 |

**Ganador (promedio F1+AUC): xgboost.**

Hiperparámetros óptimos por modelo:

- logreg: {'C': 0.3, 'class_weight': 'balanced'}
- knn: {'kneighborsclassifier__n_neighbors': 25, 'kneighborsclassifier__weights': 'distance'}
- svm: {'svc__C': 10.0, 'svc__gamma': 'scale'}
- random_forest: {'max_depth': 20}
- xgboost: {'learning_rate': 0.05, 'max_depth': 6}
