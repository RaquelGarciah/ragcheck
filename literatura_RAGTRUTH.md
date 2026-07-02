# Literatura relevante para el TFG de detección de alucinaciones en RAG

Documento de referencia para el capítulo de Estado del Arte y para las citas
del resto de la memoria. Pegar en la carpeta del proyecto
(`alucinaciones_rag/`) antes de arrancar con Claude Code.

> Aviso: las cifras concretas están citadas de memoria y de los papers
> originales. Antes de meterlas en la memoria del TFG, **verifica cada número
> contra el paper original**. Los rangos y el orden relativo son fiables.

---

## 1. El dataset: RAGTruth

### Referencia principal

**Niu, C., Wu, Y., Zhu, J., Xu, S., Shum, K., Zhong, R., Song, J., & Zhang,
T. (2024).** *RAGTruth: A Hallucination Corpus for Developing Trustworthy
Retrieval-Augmented Language Models.* Proceedings of ACL 2024.
arXiv:2401.00396.

### Contenido del dataset

- 17.790 ejemplos anotados a mano a nivel de span.
- Tres tareas: **QA**, **Summary**, **Data2txt**.
- Seis modelos generadores: GPT-4, GPT-3.5-turbo, Llama-2-7B/13B/70B,
  Mistral-7B.
- Distribuciones de alucinación: global ~44,5% positivo; por tarea Data2txt
  69,4%, QA 31,1%, Summary 31,1%.
- Cuatro tipos de alucinación anotados: `Evident Conflict`,
  `Subtle Conflict`, `Evident Introduction of Baseless Information`,
  `Subtle Introduction of Baseless Information`.

### Baselines del paper original (aproximados)

| Enfoque | F1 aproximada | Nota |
|---|---|---|
| GPT-4 zero-shot como juez | 0,80-0,85 | Referencia comercial |
| GPT-3.5 zero-shot como juez | 0,60-0,70 | Baseline barato |
| Llama-2-70B zero-shot | 0,65-0,75 | Open-source grande |
| Fine-tuned Llama-2-13B | 0,78-0,85 | Especializado sobre RAGTruth |

### Descomposición por dificultad

- **Data2txt** es la tarea más fácil: alucinaciones típicamente gruesas
  (cifras y entidades inventadas).
- **QA** es la más difícil: alucinaciones semánticas sutiles.
- **Summary** es intermedia.
- **Evident Conflict** es el tipo más fácil; **Subtle Introduction of
  Baseless Information** es el techo del problema.

---

## 2. RAG y arquitecturas de recuperación aumentada

### Papers fundamentales

**Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
et al. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP
Tasks.* NeurIPS 2020. arXiv:2005.11401.

**Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., et al. (2024).**
*Retrieval-Augmented Generation for Large Language Models: A Survey.*
arXiv:2312.10997.

### Puntos para el TFG

- La motivación original de RAG: mitigar alucinaciones dando contexto
  documental fresco al LLM.
- La paradoja: RAG reduce alucinaciones globales pero introduce las
  alucinaciones "de fuente".
- Este segundo tipo es exactamente el que RAGTruth mide y el que este TFG
  detecta.

---

## 3. Alucinaciones en LLMs: surveys

### Referencias

**Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.
J., Madotto, A., & Fung, P. (2023).** *Survey of Hallucination in Natural
Language Generation.* ACM Computing Surveys, 55(12), 1-38.
doi:10.1145/3571730.

**Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., et al. (2023).**
*Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language
Models.* arXiv:2309.01219.

**Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., et al. (2023).**
*A Survey on Hallucination in Large Language Models: Principles, Taxonomy,
Challenges, and Open Questions.* arXiv:2311.05232.

### Puntos para el TFG

- Taxonomía de alucinaciones (factualidad vs fidelidad vs coherencia).
- Causas identificadas: sesgo de datos, exposición insuficiente a
  información actualizada, límites de contexto, sesgos de decodificación.
- El foco de este TFG está en **detección post-hoc**, no en mitigación.

---

## 4. LLM-as-a-judge

### Referencia canónica

**Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., et
al. (2023).** *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*
NeurIPS 2023. arXiv:2306.05685.

### Puntos para el TFG

- Formaliza el paradigma "usar un LLM grande para evaluar otro LLM".
- Documenta limitaciones: sesgo de posición, sesgo de longitud,
  autopreferencia, coste, falta de reproducibilidad.
- Coste real: entre 1 y 3 céntimos por juicio con GPT-4; varios segundos
  de latencia.

Es la técnica contra la que compite tu detector clásico. Las limitaciones
que documenta el paper son las que motivan tu TFG.

---

## 5. Detectores especializados fine-tuned

### Lynx (Patronus AI)

**Ravi, S. S., Mielczarek, B., Kannappan, A., Kiela, D., & Qian, R. (2024).**
*Lynx: An Open Source Hallucination Evaluation Model.* Patronus AI Technical
Report. arXiv:2407.08488.

**Puntos clave.**

- Dos variantes: **Lynx-8B** (basado en Llama-3-8B-Instruct) y **Lynx-70B**
  (basado en Llama-3-70B-Instruct).
- Fine-tuned con SFT + DPO sobre HaluBench.
- Salida encadenada: chain-of-thought + veredicto binario (PASS/FAIL).
- Hardware: 8B necesita ~16 GB VRAM; 70B requiere infraestructura seria.
- Rendimiento sobre RAGTruth:
  - Lynx-8B: **~82,9% accuracy**, AUC ~0,84.
  - Lynx-70B: **~85,7% accuracy**, AUC ~0,87.
- Comparación en su paper: GPT-4o alcanza ~87,3%, Claude 3 Sonnet ~84,1%,
  GPT-3.5 ~66,4%.
- Coste operativo estimado: ~10⁻³ céntimos por evaluación con Lynx-8B en
  GPU propia, frente a ~1 céntimo con GPT-4.

**Por qué importa para tu TFG.**

Lynx es el **detector open-source de referencia** en 2024. Es contra el que
te vas a comparar. Su pitch coincide con el tuyo: sustituir GPT-4 como
juez. La diferencia es que Patronus usa deep learning con 8B de parámetros
y GPU; tú usas features léxicas con XGBoost y CPU.

### HaluBench

Benchmark de Patronus AI que agrupa cinco datasets: RAGTruth,
FinanceBench, DROP, PubMedQA y CovidQA. Se introduce dentro del technical
report de Lynx (no hay paper standalone).

### Otros detectores comparables

- **Vectara Hallucination Evaluation Model (HHEM)**: open-source más
  pequeño (~1B parámetros), enfocado en resúmenes. AUC menor que Lynx pero
  coste más bajo.
- **SelfCheckGPT** (Manakul et al. 2023, EMNLP): no requiere modelo
  entrenado; detecta alucinaciones vía sampling y consistencia.
- **G-Eval** (Liu et al. 2023, EMNLP): usa GPT-4 con chain-of-thought para
  evaluar textos generados.

---

## 6. Referencias para el marco teórico

### Modelo de espacio vectorial y TF-IDF

**Salton, G., Wong, A., & Yang, C. S. (1975).** *A Vector Space Model for
Automatic Indexing.* Communications of the ACM, 18(11), 613-620.

**Salton, G., & Buckley, C. (1988).** *Term-Weighting Approaches in
Automatic Text Retrieval.* Information Processing & Management, 24(5),
513-523.

### LSA / TruncatedSVD

**Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., &
Harshman, R. (1990).** *Indexing by Latent Semantic Analysis.* Journal of
the American Society for Information Science, 41(6), 391-407.

**Halko, N., Martinsson, P. G., & Tropp, J. A. (2011).** *Finding Structure
with Randomness: Probabilistic Algorithms for Constructing Approximate
Matrix Decompositions.* SIAM Review, 53(2), 217-288.

### Modelos de ML

**Cox, D. R. (1958).** *The Regression Analysis of Binary Sequences.*
Journal of the Royal Statistical Society, Series B, 20(2), 215-232.

**Friedman, J. H. (2001).** *Greedy Function Approximation: A Gradient
Boosting Machine.* Annals of Statistics, 29(5), 1189-1232.

**Chen, T., & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting
System.* Proceedings of KDD 2016, 785-794.

**Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of
Statistical Learning* (2nd ed.). Springer.

### Evaluación

**Fawcett, T. (2006).** *An Introduction to ROC Analysis.* Pattern
Recognition Letters, 27(8), 861-874.

**DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).**
*Comparing the Areas Under Two or More Correlated Receiver Operating
Characteristic Curves.* Biometrics, 44(3), 837-845.

**Matthews, B. W. (1975).** *Comparison of the Predicted and Observed
Secondary Structure of T4 Phage Lysozyme.* Biochimica et Biophysica Acta,
405(2), 442-451.

### Calibración

**Platt, J. C. (1999).** *Probabilistic Outputs for Support Vector
Machines and Comparisons to Regularized Likelihood Methods.* Advances in
Large Margin Classifiers, 10(3), 61-74.

**Zadrozny, B., & Elkan, C. (2002).** *Transforming Classifier Scores
into Accurate Multiclass Probability Estimates.* Proceedings of KDD 2002,
694-699.

### Desbalance de clases

**Japkowicz, N., & Stephen, S. (2002).** *The Class Imbalance Problem: A
Systematic Study.* Intelligent Data Analysis, 6(5), 429-449.

**He, H., & Garcia, E. A. (2009).** *Learning from Imbalanced Data.*
IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.

### Bootstrap

**Efron, B. (1979).** *Bootstrap Methods: Another Look at the Jackknife.*
Annals of Statistics, 7(1), 1-26.

---

## 7. Tabla comparativa del estado del arte sobre RAGTruth

> ⚠️ **AVISO (corregido jul. 2026).** La columna **AUC-ROC de esta tabla NO
> procede de los papers**: son estimaciones hechas de memoria. La literatura de
> RAGTruth **no reporta AUC-ROC** como métrica principal (reporta F1 y accuracy),
> así que **no existe un AUC-ROC publicado de Lynx, HHEM, GPT-4o, etc. con el que
> compararse**. No usar esta columna en la memoria. Las **cifras reales y
> verificadas** (F1 y accuracy) están en **§11.3–§11.6**, y la situación del AUC
> en la literatura, en **§11.9**. La columna de accuracy también mezcla valores
> reales (Lynx 80,0%, GPT-4o 84,3% — ver §11.6) con estimaciones.

Tabla ilustrativa del espacio de compromisos (coste/latencia/tamaño); las cifras
de rendimiento son orientativas, **no citables**. Verificar contra §11.

| Sistema | Parámetros | Hardware | Latencia | Coste/eval | AUC-ROC | Accuracy |
|---|---|---|---|---|---|---|
| Predictor aleatorio | 0 | — | 0 ms | 0 | 0,50 | 0,55 |
| Predictor `containment` (una feature) | 1 | CPU | <1 ms | 0 | ~0,76 | ~0,72 |
| Baseline clásico 5 features (TFG) | ~10 | CPU | <1 ms | 0 | 0,78-0,82 | ~0,74 |
| Detector clásico extendido (TFG objetivo) | ~100 | CPU | ~1 ms | 0 | 0,85-0,88 | ~0,80 |
| Vectara HHEM (open source) | ~1×10⁹ | GPU 8 GB | ~500 ms | ~10⁻⁴ € | 0,80-0,83 | ~0,76 |
| Lynx-8B (Patronus, open source) | 8×10⁹ | GPU 16 GB | ~2 s | ~10⁻³ € | 0,84 | 0,83 |
| Lynx-70B (Patronus) | 70×10⁹ | 4×A100 | ~10 s | ~10⁻² € | 0,87 | 0,86 |
| GPT-3.5 como juez | ~175×10⁹ | API | ~2 s | ~0,001 € | 0,72 | 0,66 |
| Claude 3 Sonnet como juez | ~500×10⁹ (est.) | API | ~3 s | ~0,003 € | 0,83 | 0,84 |
| GPT-4o como juez | ~1,7×10¹² (est.) | API | ~5 s | ~0,01 € | 0,88 | 0,87 |
| Concordancia humana (upper bound) | — | — | ~30 s | ~0,50 € | ~0,92 | ~0,90 |

Fuentes: paper de RAGTruth (Niu et al. 2024), paper de Lynx (Ravi et al.
2024), technical reports de Vectara y Patronus, y estimaciones de coste
de las APIs oficiales.

**Nota**: los tamaños de GPT-3.5, GPT-4o y Claude 3 son estimaciones
porque no están publicados oficialmente.

---

## 8. Estructura sugerida del capítulo de estado del arte

**2.1 — La arquitectura RAG y sus limitaciones.**
- Cita: Lewis et al. 2020, Gao et al. 2024.
- Cierre: el problema de las alucinaciones en RAG específicamente.

**2.2 — Alucinaciones en modelos de lenguaje.**
- Cita: Ji et al. 2023, Zhang et al. 2023.
- Foco en la fidelidad como concepto clave para RAG.

**2.3 — Enfoques de detección: LLM-as-a-judge.**
- Cita: Zheng et al. 2023.
- Limitaciones documentadas: coste, latencia, opacidad.

**2.4 — Detectores especializados fine-tuned.**
- Cita: Ravi et al. 2024 (Lynx), Vectara HHEM.

**2.5 — Benchmark RAGTruth.**
- Cita: Niu et al. 2024.
- Descripción del corpus y estado del arte cuantitativo.

**2.6 — El hueco que llena este TFG.**
- Argumento: no hay estudios sistemáticos con ML clásico sobre RAGTruth.
- Este TFG explora esa vía.

---

## 9. Frases hechas útiles para la memoria

**Para el gap.**

"Hasta donde conocemos, no existe en la literatura un estudio sistemático
de detectores basados en Machine Learning clásico evaluados sobre
RAGTruth. La comunidad se ha centrado en detectores fine-tuned sobre
arquitecturas Transformer, dejando abierta la cuestión de cuánto
rendimiento es posible alcanzar con técnicas interpretables y de coste
cero."

**Para la comparación con Lynx.**

"El detector propuesto se sitúa en un espacio de compromisos distinto al
de los detectores fine-tuned como Lynx (Ravi et al. 2024): sacrifica
algunos puntos de rendimiento a cambio de eliminar por completo el coste
de inferencia, la dependencia de GPU y la opacidad del razonamiento."

**Para el techo teórico.**

"Los detectores clásicos basados en features léxicas tienen un techo
natural en el detector de alucinaciones sutiles, donde la señal semántica
escapa a las estadísticas de superficie. Este techo, estimado
empíricamente en el rango de AUC 0,88-0,90 sobre RAGTruth, define la
frontera entre lo que la minería de datos clásica puede capturar y lo
que requiere representaciones neuronales profundas."

**Para el pitch de la aportación.**

"La aportación central del trabajo es demostrar que un detector
construido sobre veintiséis features léxicas y estadísticas, entrenado
con regresión logística y XGBoost bajo un protocolo de evaluación
honesto (GroupKFold por documento fuente, test de permutación, intervalos
bootstrap), alcanza un rendimiento competitivo con detectores fine-tuned
de miles de millones de parámetros, con las ventajas añadidas de
interpretabilidad completa, coste de inferencia despreciable, y
determinismo absoluto."

---

## 10. Cómo usar este documento con Claude Code

1. Copia este fichero a la carpeta `alucinaciones_rag/` del proyecto.
2. Puedes renombrarlo a `LITERATURA.md`.
3. Cuando arranques Claude Code, dile "lee `LITERATURA.md` y úsalo como
   referencia bibliográfica al redactar el capítulo de estado del arte".
4. **Verifica cada cifra concreta contra el paper original** antes de
   meterla en la memoria.

---

## 11. Cifras verificadas contra fuentes primarias (búsqueda julio 2026)

Esta sección **corrige y sustituye** las cifras "de memoria" de las secciones
1, 5 y 7. Todos los números vienen de tablas de los papers originales
(consultados en arXiv/ACL). Las tres primeras secciones del documento se
escribieron de memoria; **usar estas cifras, no aquellas.**

### 11.1 Aviso metodológico crítico: la métrica es F1, no AUC-ROC

La literatura sobre RAGTruth **no reporta AUC-ROC**. Reporta **F1** en dos
tareas distintas:

- **Nivel respuesta (response/example-level):** clasificación binaria
  "¿esta respuesta contiene alguna alucinación?". Es nuestra tarea.
- **Nivel span:** localizar los tramos exactos alucinados dentro de la
  respuesta. Tarea más dura; nosotras **no** la abordamos (nuestro `label`
  es binario a nivel respuesta).

**Implicación para el TFG.** La tabla comparativa §7 (en AUC-ROC) no es
directamente comparable con el estado del arte, que está en F1. Para
posicionarnos frente a Lynx, LettuceDetect, RAG-HAT, etc., **hay que reportar
F1 a nivel respuesta** además del AUC-ROC. El AUC sigue siendo válido para
nuestra hipótesis falsable interna (≥0,80), pero la comparación externa exige
F1. Ojo también: los SOTA se evalúan sobre el **split oficial test** de
RAGTruth, no con GroupKFold — para comparar cifras hay que dar también nuestra
F1 sobre ese split oficial.

### 11.2 Estadísticas del dataset (Niu et al. 2024, tabla del paper)

- **17.790 respuestas** anotadas (algunas fuentes citan 17.838, incluyendo
  variantes); **2.965 prompts** distintos.
- Alucinación global: **44,4 %** de respuestas positivas.
- Por tarea: **Data2txt 69,2 %**, **QA 31,0 %**, **Summarization 32,6 %**
  (fuente de resumen: CNN/DailyMail).
- Data2txt es la tarea con más alucinación y la más "gruesa"; QA y Summary
  son las difíciles.

### 11.3 Baselines del paper RAGTruth — **F1 nivel respuesta** (Tabla 5)

| Método | F1 global (%) |
|---|---|
| GPT-3.5-turbo (prompt) | **1,0** |
| SelfCheckGPT (GPT-3.5-turbo) | 36,6 |
| SelfCheckGPT (GPT-4-turbo) | 60,5 |
| GPT-4-turbo (prompt) | 68,3 |
| **Fine-tuned Llama-2-13B** | **80,7** ← mejor del paper original |

Dato llamativo y citable: **GPT-3.5-turbo como juez zero-shot colapsa a 1,0 %
de F1** en RAGTruth (no sigue el formato de salida). Refuerza el argumento de
que "LLM barato como juez" no es fiable.

### 11.4 Baselines del paper RAGTruth — **F1 nivel span** (Tabla 6)

| Método | F1 global (%) |
|---|---|
| GPT-4-turbo (prompt) | 32,7 |
| Fine-tuned Llama-2-13B | 54,8 |

(Referencia de dificultad de la tarea span; nosotras no competimos aquí.)

### 11.5 Estado del arte posterior — **F1 nivel respuesta sobre RAGTruth**

Recopilado de la tabla comparativa de LettuceDetect (Kovács et al. 2025),
que reúne los métodos sobre el split oficial:

| Método | QA | Data2txt | Summ | **Global** |
|---|---|---|---|---|
| GPT-4-turbo (prompt) | 45,6 | 78,3 | 47,6 | 63,4 |
| Luna (encoder ligero) | 51,3 | 75,9 | 52,5 | 65,4 |
| LettuceDetect-base (150M) | 65,5 | 87,9 | 50,5 | 76,1 |
| Fine-tuned Llama-2-13B | 68,2 | 88,1 | 59,1 | 78,7 |
| **LettuceDetect-large (396M)** | 70,2 | 88,5 | 59,7 | **79,2** |
| **RAG-HAT (mejor)** | 74,8 | 91,6 | 67,6 | **83,9** |

Patrón que se repite en **todos** los sistemas: **Data2txt ~88–92 %**,
**QA ~46–75 %**, **Summary ~48–68 %**. Confirma empíricamente que Data2txt es
fácil y QA/Summary son el techo — exactamente lo que nuestras features léxicas
deberían mostrar (alta señal en Data2txt, baja en QA sutil).

### 11.6 Lynx sobre RAGTruth (Ravi et al. 2024, Tabla 3) — **accuracy**

Las cifras "~82,9 % / ~85,7 %" de §5 estaban mal: eran de **HaluBench global**,
no de RAGTruth. Sobre el **subset RAGTruth** las accuracy reales son:

| Modelo | Accuracy en RAGTruth (%) |
|---|---|
| GPT-3.5-turbo | 50,7 |
| Claude-3-Sonnet | 79,1 |
| **Lynx-8B** | 80,0 |
| **Lynx-70B** | 80,2 |
| **GPT-4o** | **84,3** ← mejor en RAGTruth |

Sobre **HaluBench global** (5 datasets): Lynx-8B **82,9 %**, Lynx-70B
**87,4 %** (aquí Lynx-70B sí supera a GPT-4o de media, pero no en RAGTruth).
Titular honesto: **en RAGTruth, GPT-4o (84,3 %) > Lynx-70B (80,2 %) ≈ Lynx-8B
(80,0 %)**; Lynx gana en el agregado de otros dominios, no en RAGTruth.

### 11.7 Lecturas nuevas que conviene citar (encontradas en la búsqueda)

- **LettuceDetect (Kovács et al., 2025), arXiv:2502.17125.** Detector token-level
  sobre **ModernBERT** (150M/396M params), contexto hasta 4k–8k tokens,
  **30–60 ejemplos/s en 1 GPU**, ~30× más pequeño que los mejores. F1 respuesta
  global 79,2 %. Es el **competidor "ligero" más directo** conceptualmente
  (encoder pequeño vs. LLM), aunque siga siendo red neuronal. Buen contraste:
  ellos ModernBERT en GPU, nosotras TF-IDF+XGBoost en CPU.
- **RAG-HAT (Song et al., EMNLP 2024 Industry), aclanthology 2024.emnlp-industry.113.**
  Pipeline de *hallucination-aware tuning*; **mejor F1 respuesta reportado (83,9 %)**.
  Es la cota alta actual sobre RAGTruth a nivel respuesta.
- **Luna.** Encoder ligero para detección en tiempo real; F1 global 65,4 %.
  Útil como punto medio en el eje coste/rendimiento.
- **RAGTruth++ / re-anotación.** Un subset QA/summarization de 408 ejemplos fue
  re-anotado: los spans alucinados pasaron de **86 a 865**, evidenciando
  **muchos falsos negativos** en las anotaciones span originales. Relevante como
  *limitación del dataset* a mencionar en la memoria (afecta sobre todo a la
  tarea span, menos a la binaria a nivel respuesta).

### 11.8 Dónde encaja nuestro detector (posicionamiento honesto)

Cifras **reales ya logradas** (18 features léxicas/relacionales + nivel frase +
one-hot de tarea; grid search con GroupKFold por `source`, evaluación en el
**split oficial test** de RAGTruth, n = 2 700, prevalencia 0,349):

- **AUC-ROC 0,836** (Random Forest) / 0,835 (XGBoost).
- **F1 nivel respuesta 0,692** (SVM) / 0,684 (XGBoost).
- **Accuracy / balanced-accuracy ≈ 0,74–0,76**.

Comparación **honesta** (métrica del SOTA = F1 y accuracy, §11.3–§11.6):
- **Por encima** de GPT-4-turbo-como-juez (F1 0,634) y de Luna (0,654).
- **Por debajo** de los fine-tuned: Llama-2-13B (F1 0,787), LettuceDetect (0,792),
  RAG-HAT (0,839); y en accuracy, de Lynx-8B (80,0%) y GPT-4o (84,3%) —a ~10 pts.
- El pitch NO es "ganar" a GPT-4o, sino **rendimiento competitivo a coste cero,
  CPU, interpretable y determinista**, y **acotar el techo** de lo léxico.

Reparto por tarea (OOF): Data2txt AUC ≈ 0,81 (casi SOTA), QA ≈ 0,81, **Summary
≈ 0,70** (cuello de botella estructural: camuflaje léxico —el texto fiel de un
resumen reutiliza el vocabulario de la fuente igual que el alucinado—). Este
hallazgo, con figuras, es aportación propia (ver `outputs/reports/
error_analysis_summary.md` en el repo de código).

### 11.9 El AUC-ROC en la literatura (situación verificada)

Punto delicado que corrige la tabla §7: **el AUC-ROC apenas se reporta en RAGTruth.**

- El **paper de RAGTruth y los detectores de referencia** (Lynx, LettuceDetect,
  RAG-HAT) reportan **F1 y accuracy, no AUC-ROC**. Por tanto **no hay un AUC-ROC
  publicado de Lynx/HHEM/GPT-4o sobre RAGTruth** con el que compararse; los AUC
  del §7 eran estimaciones, no cifras de paper.
- Las pocas cifras de AUROC sobre RAGTruth que sí existen vienen de trabajos
  secundarios y **en regímenes distintos** (no comparables de tú a tú):
  - **ECS-Span** (detección de spans): ROC-AUC ≈ **0,742** a nivel caso.
  - **PARALLAX** (detección por *activaciones* del LLM): los métodos caen entre
    **0,43 y 0,57** (≈ azar) en su reconstrucción; **no es** clasificación
    supervisada sobre features como la nuestra.
- **Consecuencia metodológica.** Reportar **AUC-ROC como nuestra métrica interna**
  (independiente del umbral, cumple la hipótesis ≥0,80, estándar en ML) es
  correcto, pero la **comparación externa** con el estado del arte **debe hacerse
  en F1 y accuracy**, no en AUC. Evitar frases del tipo "nuestro AUC iguala a
  Lynx": Lynx no publica AUC.
- **Matiz que es un regalo, no una amenaza.** PARALLAX sostiene que en RAGTruth
  "la etiqueta **no está correlacionada con diferencias de texto de superficie**"
  (marcan n/a una baseline de coseno TF-IDF). Que un clasificador clásico sobre
  features de superficie alcance 0,82–0,84 de AUC es, en ese contexto, un
  resultado **no trivial**; y nuestro desglose por tarea explica la aparente
  paradoja (funciona en Data2txt, fracasa en Summary). Es material de discusión
  fuerte para la memoria.

**Fuentes (verificadas jul. 2026):**
[RAGTruth, Niu et al. 2024 (ACL)](https://aclanthology.org/2024.acl-long.585/) ·
[arXiv:2401.00396](https://arxiv.org/html/2401.00396v1) ·
[Lynx, Ravi et al. 2024](https://arxiv.org/html/2407.08488v1) ·
[LettuceDetect, 2025](https://arxiv.org/html/2502.17125) ·
[RAG-HAT, EMNLP 2024](https://aclanthology.org/2024.emnlp-industry.113.pdf) ·
[PARALLAX, 2026 (arXiv:2605.17028)](https://arxiv.org/html/2605.17028) ·
[Real-Time Eval Models for RAG (arXiv:2503.21157)](https://arxiv.org/abs/2503.21157)

> Verificación jul. 2026: el rango "AUROC 0,814–0,837 en 7 modelos" que circula en
> buscadores **no se pudo confirmar sobre RAGTruth** (el benchmark que lo reporta
> —Real-Time Eval Models— usa FinQA/ELI5/CovidQA/DROP/FinanceBench/PubMedQA, no
> RAGTruth). No usarlo como AUC de RAGTruth.