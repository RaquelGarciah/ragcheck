# tfidf_cos vs containment, y sets decorrelados

## Por qué containment gana (una feature sola, OOF GroupKFold, xgboost)

| Feature sola | AUC | Data2txt | QA | Summary |
|---|---|---|---|---|
| containment | 0,774 | 0,712 | 0,760 | 0,652 |
| tfidf_cos | 0,695 | 0,560 | 0,601 | 0,548 |

`tfidf_cos` es ~8 pts peor de AUC, y en Data2txt casi azar (0,560): el coseno de
una respuesta corta contra un documento largo lo dominan las palabras comunes
compartidas, ciego a las cifras/entidades fabricadas. `containment` = |R∩F|/|R|
mide *anclaje* directo (¿qué fracción de lo dicho está en la fuente?), que es la
definición operativa de alucinación; `tfidf_cos` mide *parecido temático*, un
proxy más pobre.

## Modelo con tfidf_cos + features decorreladas con ella

Correlación con tfidf_cos: alta con novel_bigram (−0,83), sent_sim_mean (0,78),
containment (0,74); baja con num_context (0,27), num_overlap (0,09), answer_len.

| Modelo | k | AUC | F1 | Data2txt | QA | Summary |
|---|---|---|---|---|---|---|
| tfidf_cos sola | 1 | 0,695 | 0,612 | 0,560 | 0,601 | 0,548 |
| tfidf + decorr (\|rho\|<0,3) | 5 | 0,795 | 0,661 | 0,767 | 0,778 | 0,657 |
| tfidf + decorr (\|rho\|<0,5) | 10 | 0,812 | 0,700 | 0,793 | 0,795 | 0,687 |
| TODAS (referencia) | 18 | 0,824 | 0,700 | 0,813 | 0,812 | 0,701 |
| containment + decorr (\|rho\|<0,5) | 7 | 0,817 | 0,699 | 0,791 | 0,813 | 0,687 |

**Conclusiones.** (1) tfidf + sus decorreladas llega a 0,812, casi como las 18:
las features ortogonales aportan la señal que a tfidf le falta (selección por
mínima redundancia). (2) Pero anclar en containment da 0,817 con 7 features (< 10
y > 0,812): containment es mejor ancla. tfidf_cos es una versión más pobre de lo
que containment ya mide, y es podable sin pérdida.

Script: `exploration/tfidf_vs_containment.py`.
