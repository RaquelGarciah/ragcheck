# Selección de variables — ¿sobran features?

Objetivo: comprobar si las 18 features causan sobreajuste o redundancia, y fijar
un subconjunto parco. Método: (1) forward stepwise por modelo (greedy F1,
GroupKFold por `source`); (2) correlación entre features; (3) validación
podado-vs-completo en el **test oficial**.

## Veredicto

**Sobran features, pero no hacen daño.** En test, el rendimiento **no cae** al
añadir features (no hay sobreajuste peligroso), pero **se aplana muy pronto**:
~7 features capturan el 99% del rendimiento y las 11 restantes son redundantes.

**Rodilla (99% del mejor F1 OOF) por modelo:** logreg 10 · xgboost 8 ·
random_forest 12 · svm 6 · knn 7 → **mediana ≈ 8**.

**Podado por importancia, xgboost en test oficial:**

| k features | AUC | F1 | acc |
|---|---|---|---|
| 1 (`containment`) | 0,790 | 0,642 | 0,740 |
| 5 | 0,820 | 0,662 | 0,750 |
| **7** | **0,829** | **0,669** | **0,765** |
| 10 | 0,832 | 0,671 | 0,768 |
| 18 (todas) | 0,835 | 0,673 | 0,770 |

De 7 a 18 features: **+0,006 AUC, +0,004 F1, +0,005 acc** — ruido.

## Redundancia (Spearman)

Pares casi idénticos: `novel_bigram`~`novel_trigram` (0,98),
`containment`~`sent_cont_mean` (0,92), `containment`~`novel_bigram` (−0,92), y
todo el bloque de solape entre 0,8–0,98. Hay ~2–3 señales independientes
codificadas en 13 features.

## Subconjunto recomendado (7 features, una por cluster de señal)

`containment`, `answer_len`, `num_overlap`, `jaccard`, `neg_diff`,
`num_context`, `sent_sim_min`.

Descartables por redundantes: `novel_bigram` (≈`novel_trigram`), `sent_cont_mean`
(≈`containment`), `tfidf_cos`, y 4 de las 6 de nivel frase, más las one-hot de
tarea (aportan <0,004 de importancia por permutación).

**Figuras:** `fsel_{logreg,knn,svm,random_forest,xgboost}` (curvas por modelo),
`feat_correlacion` (redundancia), `feat_importancia_perm` (importancia),
`feat_podado_test` (rendimiento vs nº features).

---

## Detalle: forward stepwise por modelo

Curva OOF; SVM sobre submuestra de 5000. Umbral Youden por paso.


## logreg

- Mejor F1 = 0.696 con 12 features (AUC 0.788, acc 0.710).
- **Rodilla (99% del mejor F1): 10 features.** Orden de entrada: containment, answer_len, num_overlap, sent_sim_mean, neg_diff, task_Data2txt, task_QA, sent_cont_mean, novel_trigram, sent_cont_std, sent_frac_low, tfidf_cos, task_Summary, num_context, novel_bigram, sent_sim_min, sent_cont_min, jaccard.

|   n | added          |   auc |    f1 |   recall |   precision |   acc |   bal_acc |
|----:|:---------------|------:|------:|---------:|------------:|------:|----------:|
|   1 | containment    | 0.761 | 0.667 |    0.664 |       0.669 | 0.704 |     0.7   |
|   2 | answer_len     | 0.765 | 0.684 |    0.739 |       0.637 | 0.696 |     0.701 |
|   3 | num_overlap    | 0.766 | 0.687 |    0.758 |       0.628 | 0.692 |     0.699 |
|   4 | sent_sim_mean  | 0.766 | 0.685 |    0.749 |       0.632 | 0.694 |     0.699 |
|   5 | neg_diff       | 0.767 | 0.686 |    0.749 |       0.632 | 0.694 |     0.699 |
|   6 | task_Data2txt  | 0.771 | 0.686 |    0.751 |       0.631 | 0.693 |     0.699 |
|   7 | task_QA        | 0.775 | 0.686 |    0.752 |       0.631 | 0.693 |     0.699 |
|   8 | sent_cont_mean | 0.778 | 0.687 |    0.744 |       0.638 | 0.698 |     0.703 |
|   9 | novel_trigram  | 0.778 | 0.689 |    0.752 |       0.635 | 0.697 |     0.703 |
|  10 | sent_cont_std  | 0.785 | 0.689 |    0.715 |       0.665 | 0.713 |     0.713 |
|  11 | sent_frac_low  | 0.787 | 0.695 |    0.738 |       0.656 | 0.711 |     0.714 |
|  12 | tfidf_cos      | 0.788 | 0.696 |    0.744 |       0.653 | 0.71  |     0.713 |
|  13 | task_Summary   | 0.788 | 0.695 |    0.743 |       0.653 | 0.71  |     0.713 |
|  14 | num_context    | 0.789 | 0.695 |    0.749 |       0.649 | 0.708 |     0.712 |
|  15 | novel_bigram   | 0.789 | 0.694 |    0.742 |       0.652 | 0.709 |     0.712 |
|  16 | sent_sim_min   | 0.789 | 0.694 |    0.744 |       0.65  | 0.708 |     0.711 |
|  17 | sent_cont_min  | 0.79  | 0.693 |    0.742 |       0.65  | 0.707 |     0.711 |
|  18 | jaccard        | 0.79  | 0.689 |    0.724 |       0.657 | 0.709 |     0.71  |

## xgboost

- Mejor F1 = 0.717 con 15 features (AUC 0.820, acc 0.738).
- **Rodilla (99% del mejor F1): 8 features.** Orden de entrada: sent_sim_mean, novel_bigram, num_overlap, sent_cont_min, neg_diff, task_Summary, task_Data2txt, answer_len, novel_trigram, containment, sent_frac_low, tfidf_cos, sent_sim_min, sent_cont_std, task_QA, sent_cont_mean, num_context, jaccard.

|   n | added          |   auc |    f1 |   recall |   precision |   acc |   bal_acc |
|----:|:---------------|------:|------:|---------:|------------:|------:|----------:|
|   1 | sent_sim_mean  | 0.741 | 0.67  |    0.722 |       0.624 | 0.683 |     0.687 |
|   2 | novel_bigram   | 0.764 | 0.678 |    0.727 |       0.635 | 0.692 |     0.696 |
|   3 | num_overlap    | 0.775 | 0.691 |    0.766 |       0.63  | 0.695 |     0.702 |
|   4 | sent_cont_min  | 0.786 | 0.702 |    0.785 |       0.635 | 0.704 |     0.712 |
|   5 | neg_diff       | 0.792 | 0.7   |    0.765 |       0.645 | 0.708 |     0.713 |
|   6 | task_Summary   | 0.794 | 0.703 |    0.762 |       0.651 | 0.712 |     0.717 |
|   7 | task_Data2txt  | 0.796 | 0.703 |    0.758 |       0.656 | 0.715 |     0.72  |
|   8 | answer_len     | 0.803 | 0.71  |    0.77  |       0.66  | 0.72  |     0.725 |
|   9 | novel_trigram  | 0.803 | 0.71  |    0.771 |       0.659 | 0.72  |     0.725 |
|  10 | containment    | 0.814 | 0.712 |    0.739 |       0.687 | 0.734 |     0.734 |
|  11 | sent_frac_low  | 0.814 | 0.713 |    0.749 |       0.68  | 0.731 |     0.733 |
|  12 | tfidf_cos      | 0.816 | 0.713 |    0.747 |       0.682 | 0.732 |     0.734 |
|  13 | sent_sim_min   | 0.82  | 0.716 |    0.743 |       0.692 | 0.738 |     0.738 |
|  14 | sent_cont_std  | 0.82  | 0.714 |    0.731 |       0.697 | 0.739 |     0.738 |
|  15 | task_QA        | 0.82  | 0.717 |    0.748 |       0.689 | 0.738 |     0.738 |
|  16 | sent_cont_mean | 0.82  | 0.715 |    0.735 |       0.696 | 0.739 |     0.739 |
|  17 | num_context    | 0.823 | 0.712 |    0.711 |       0.714 | 0.744 |     0.741 |
|  18 | jaccard        | 0.825 | 0.706 |    0.678 |       0.735 | 0.748 |     0.741 |

## random_forest

- Mejor F1 = 0.714 con 15 features (AUC 0.819, acc 0.745).
- **Rodilla (99% del mejor F1): 12 features.** Orden de entrada: sent_frac_low, num_overlap, task_Data2txt, task_QA, task_Summary, num_context, sent_cont_min, containment, sent_sim_min, tfidf_cos, sent_cont_std, sent_cont_mean, novel_bigram, answer_len, neg_diff, jaccard, novel_trigram, sent_sim_mean.

|   n | added          |   auc |    f1 |   recall |   precision |   acc |   bal_acc |
|----:|:---------------|------:|------:|---------:|------------:|------:|----------:|
|   1 | sent_frac_low  | 0.709 | 0.669 |    0.742 |       0.609 | 0.673 |     0.68  |
|   2 | num_overlap    | 0.705 | 0.68  |    0.801 |       0.591 | 0.665 |     0.678 |
|   3 | task_Data2txt  | 0.732 | 0.689 |    0.821 |       0.594 | 0.671 |     0.685 |
|   4 | task_QA        | 0.739 | 0.675 |    0.733 |       0.625 | 0.685 |     0.69  |
|   5 | task_Summary   | 0.738 | 0.676 |    0.74  |       0.622 | 0.684 |     0.69  |
|   6 | num_context    | 0.754 | 0.665 |    0.678 |       0.651 | 0.695 |     0.693 |
|   7 | sent_cont_min  | 0.759 | 0.657 |    0.655 |       0.659 | 0.695 |     0.691 |
|   8 | containment    | 0.784 | 0.678 |    0.678 |       0.678 | 0.714 |     0.71  |
|   9 | sent_sim_min   | 0.799 | 0.689 |    0.671 |       0.707 | 0.73  |     0.724 |
|  10 | tfidf_cos      | 0.805 | 0.699 |    0.707 |       0.69  | 0.728 |     0.726 |
|  11 | sent_cont_std  | 0.809 | 0.701 |    0.7   |       0.702 | 0.734 |     0.731 |
|  12 | sent_cont_mean | 0.809 | 0.707 |    0.726 |       0.689 | 0.732 |     0.731 |
|  13 | novel_bigram   | 0.812 | 0.706 |    0.708 |       0.705 | 0.738 |     0.735 |
|  14 | answer_len     | 0.817 | 0.702 |    0.671 |       0.737 | 0.747 |     0.739 |
|  15 | neg_diff       | 0.819 | 0.714 |    0.712 |       0.715 | 0.745 |     0.742 |
|  16 | jaccard        | 0.821 | 0.711 |    0.709 |       0.713 | 0.743 |     0.74  |
|  17 | novel_trigram  | 0.82  | 0.708 |    0.702 |       0.715 | 0.743 |     0.739 |
|  18 | sent_sim_mean  | 0.82  | 0.708 |    0.701 |       0.715 | 0.742 |     0.738 |

## svm (submuestra 5000)

- Mejor F1 = 0.721 con 8 features (AUC 0.801, acc 0.728).
- **Rodilla (99% del mejor F1): 6 features.** Orden de entrada: containment, answer_len, sent_frac_low, sent_sim_min, num_overlap, task_QA, neg_diff, sent_cont_min, sent_cont_std, tfidf_cos, novel_bigram, sent_sim_mean, task_Data2txt, sent_cont_mean, task_Summary, novel_trigram, num_context, jaccard.

|   n | added          |   auc |    f1 |   recall |   precision |   acc |   bal_acc |
|----:|:---------------|------:|------:|---------:|------------:|------:|----------:|
|   1 | containment    | 0.742 | 0.687 |    0.726 |       0.651 | 0.705 |     0.707 |
|   2 | answer_len     | 0.773 | 0.704 |    0.759 |       0.656 | 0.716 |     0.72  |
|   3 | sent_frac_low  | 0.778 | 0.702 |    0.754 |       0.657 | 0.715 |     0.719 |
|   4 | sent_sim_min   | 0.783 | 0.705 |    0.758 |       0.658 | 0.717 |     0.721 |
|   5 | num_overlap    | 0.785 | 0.706 |    0.754 |       0.663 | 0.72  |     0.724 |
|   6 | task_QA        | 0.8   | 0.715 |    0.773 |       0.665 | 0.726 |     0.73  |
|   7 | neg_diff       | 0.799 | 0.72  |    0.787 |       0.663 | 0.727 |     0.733 |
|   8 | sent_cont_min  | 0.801 | 0.721 |    0.79  |       0.663 | 0.728 |     0.734 |
|   9 | sent_cont_std  | 0.803 | 0.719 |    0.77  |       0.675 | 0.732 |     0.736 |
|  10 | tfidf_cos      | 0.804 | 0.72  |    0.77  |       0.675 | 0.733 |     0.737 |
|  11 | novel_bigram   | 0.805 | 0.72  |    0.778 |       0.67  | 0.73  |     0.735 |
|  12 | sent_sim_mean  | 0.805 | 0.719 |    0.776 |       0.67  | 0.73  |     0.735 |
|  13 | task_Data2txt  | 0.805 | 0.716 |    0.763 |       0.675 | 0.731 |     0.734 |
|  14 | sent_cont_mean | 0.804 | 0.715 |    0.766 |       0.67  | 0.728 |     0.732 |
|  15 | task_Summary   | 0.803 | 0.715 |    0.765 |       0.67  | 0.728 |     0.732 |
|  16 | novel_trigram  | 0.802 | 0.716 |    0.772 |       0.667 | 0.727 |     0.732 |
|  17 | num_context    | 0.803 | 0.713 |    0.756 |       0.674 | 0.729 |     0.731 |
|  18 | jaccard        | 0.805 | 0.704 |    0.7   |       0.708 | 0.738 |     0.734 |

## knn

- Mejor F1 = 0.704 con 16 features (AUC 0.806, acc 0.732).
- **Rodilla (99% del mejor F1): 7 features.** Orden de entrada: sent_sim_min, novel_bigram, task_QA, num_overlap, neg_diff, answer_len, sent_cont_std, task_Summary, task_Data2txt, sent_sim_mean, containment, novel_trigram, sent_frac_low, jaccard, num_context, sent_cont_mean, sent_cont_min, tfidf_cos.

|   n | added          |   auc |    f1 |   recall |   precision |   acc |   bal_acc |
|----:|:---------------|------:|------:|---------:|------------:|------:|----------:|
|   1 | sent_sim_min   | 0.678 | 0.666 |    0.887 |       0.534 | 0.604 |     0.632 |
|   2 | novel_bigram   | 0.754 | 0.67  |    0.713 |       0.632 | 0.687 |     0.69  |
|   3 | task_QA        | 0.763 | 0.678 |    0.719 |       0.641 | 0.695 |     0.698 |
|   4 | num_overlap    | 0.774 | 0.685 |    0.719 |       0.654 | 0.705 |     0.707 |
|   5 | neg_diff       | 0.773 | 0.686 |    0.727 |       0.65  | 0.704 |     0.706 |
|   6 | answer_len     | 0.779 | 0.693 |    0.734 |       0.657 | 0.711 |     0.713 |
|   7 | sent_cont_std  | 0.784 | 0.697 |    0.734 |       0.664 | 0.716 |     0.717 |
|   8 | task_Summary   | 0.784 | 0.696 |    0.727 |       0.667 | 0.717 |     0.718 |
|   9 | task_Data2txt  | 0.784 | 0.696 |    0.727 |       0.667 | 0.717 |     0.718 |
|  10 | sent_sim_mean  | 0.784 | 0.696 |    0.728 |       0.666 | 0.716 |     0.717 |
|  11 | containment    | 0.795 | 0.699 |    0.717 |       0.681 | 0.724 |     0.724 |
|  12 | novel_trigram  | 0.795 | 0.699 |    0.713 |       0.686 | 0.727 |     0.725 |
|  13 | sent_frac_low  | 0.792 | 0.696 |    0.712 |       0.681 | 0.723 |     0.722 |
|  14 | jaccard        | 0.801 | 0.703 |    0.72  |       0.686 | 0.729 |     0.728 |
|  15 | num_context    | 0.804 | 0.702 |    0.714 |       0.691 | 0.73  |     0.729 |
|  16 | sent_cont_mean | 0.806 | 0.704 |    0.714 |       0.694 | 0.732 |     0.73  |
|  17 | sent_cont_min  | 0.807 | 0.689 |    0.645 |       0.739 | 0.741 |     0.731 |
|  18 | tfidf_cos      | 0.806 | 0.702 |    0.712 |       0.692 | 0.731 |     0.729 |
