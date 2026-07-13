# Correo del tutor (Daniel) — revisión del capítulo práctico y estructura SEMMA

Fecha: julio 2026. Fichero revisado: `../tex_revisados/resultados_revisado.tex` (49 comentarios
`% TUTOR p.XX:`). Esquema de metodologías: `../material/Metodologia_Analitica_ML.pdf` (CRISP-DM y SEMMA).

## Correo (literal)

> Ya hice una revisión más a fondo de tu capítulo 5. Lee los comentarios en rojo con atención. Aunque
> hay bastantes, no te preocupes, que en general son cosas "menores". Lo más relevante sería cambiar la
> estructura del capítulo (el texto es aprovechable).
>
> Metodologías de modelización (CRISP-DM de IBM o SEMMA de SAS). Fases SEMMA (nomenclatura SAS):
> - Exploración (E = Exploring) — descriptivo
> - Muestreo (S = Sampling) — generar muestras (training, validation, test), balancear clases
>   (sobre/bajomuestreo si es necesario)
> - Modificación de datos (M = Modifying) — tratamiento de missings y outliers y selección de
>   variables. Normalmente no se incluye aquí el feature engineering (pero podría ir)
> - Modelización (M = Modelling) — opciones / rejillas de hiperparámetros (podría ir la validación
>   cruzada por estar relacionada)
> - Evaluación / valoración (A = Assessment) — presentación de las métricas de los modelos ajustados
>
> Text Mining sobre los datos de partida (tokenizar, quitar stopwords, lematizar, tf-idf, etc.):
> ponerlo como parte del feature engineering o en un subapartado tras 4.2 Software utilizado. Si te lo
> dan hecho en los datasets, mencionarlo y ya está.

## Estructura pedida

### 3. MARCO TEÓRICO
- **3.1 Text mining** (3 pp) — solo teoría del text mining necesaria para crear las variables del
  práctico; NO feature engineering ni definir variables creadas. La teoría de selección de variables va
  a 3.2.3, más desarrollada.
- **3.2 Metodología** (3 pp) — explicar que seguimos SEMMA, describirla, listar las fases con una línea
  cada una y poner la figura del esquema SEMMA. Fases:
  - 3.2.1 Análisis exploratorio (Explore) — descriptivo
  - 3.2.2 Muestreo (Sample) — muestras train/validation/test, balanceo de clases
  - 3.2.3 Manipulación de datos (Modify) — missings, outliers y selección de variables
  - 3.2.4 Modelización (Model) — rejillas de hiperparámetros (+ validación cruzada)
  - 3.2.5 Valoración (Assess) — métricas, muy claro, con espacio, estilo Sergio
- **3.3 Fundamentos de los modelos utilizados**
  - 3.3.1 Introducción al aprendizaje automático (1,5 pp)
  - 3.3.2 Regresión logística (3 pp) · 3.3.3 KNN (2 pp) · 3.3.4 Árboles (3 pp) · 3.3.5 Random Forest
    (1,5-2 pp) · 3.3.6 Gradient Boosting (1,5-2 pp)

### 4. MARCO PRÁCTICO
- 4.1 Presentación de la base de datos (BIEN)
- 4.2 Software utilizado — librerías de Python, etc.
- 4.3 Procesamiento de datos (feature engineering)
- 4.4 Metodología de modelización analítica
  - 4.4.1 Análisis exploratorio (sobre las variables de 4.3)
  - 4.4.2 Muestreo
  - 4.4.3 Manipulación de datos (probablemente no hay missings; comentar outliers, que afectan a la
    regresión, no a los otros)
  - 4.4.4 Modelización (opciones/rejilla de hiperparámetros)
  - 4.4.5 Valoración (resultados de las métricas)
- 4.5 Discusión (comparativa de resultados / estado del arte)

## Instrucciones transversales (literal, resumidas)
- Marco teórico: da los fundamentos del capítulo siguiente. Muy preciso, riguroso, todo explicado con
  detalle y claridad, que no quede nada sin atar; el lector entiende en todo momento de dónde viene
  todo; estructura clara y fluida. Estilo formal.
- Ir al grano, no reiterar, categórica, sin subjetividad (penaliza en Matemáticas), huir de convencer
  por agotamiento. Más formal y riguroso. Ejemplos claros y que se entiendan mejor; más categórica.
- Primero responder la pregunta del umbral y que esa parte se entienda en la redacción
  (ver `respuesta_umbral.md`).
- Buscar más bibliografía y quedarse con lo mejor y más claro.
- No referenciar a secciones numéricas de continuo.
- La parte de resultados es la mejor redactada: seguir su estilo, espacios y claridad.
- No parecer IA. El TFG de Sergio es la inspiración de redacción/presentación/estructura, sin copiar
  (plagio).

## Los 49 comentarios del práctico (resumen para aplicar en el Deliverable 3)

Menores (forma): términos ingleses en cursiva (machine learning, prompts, tokens, token, folds,
GroupKFold, score, trade-off, accuracy, recall/precision sin acento); «tramo»→«tramos» (l. 221);
puntuación (l. 149-150); «caza»→«captura/identifica» (l. 371, 1217-1219); «exhaustividad»→
«cobertura/recall» (l. 1130, 1158); reformular containment alto/bajo (l. 345-346); registro formal en
l. 1217-1219; comas y «accuracy» (l. 1181-1182).

De fondo (estructura/rigor):
- Renombrar «Ingeniería de características»→«Procesamiento de datos» y llevarla tras «Software
  utilizado» (l. 271); «Marco de trabajo»→ base de datos a 4.1 y librerías a 4.2 Software utilizado
  (l. 645, 655).
- Reorganizar la metodología según SEMMA: 4.4.1-4.4.5 (l. 521-525, 382-383, 556-558, 583, 608).
- Sacar las descripciones de algoritmos al capítulo teórico y dejar SOLO los hiperparámetros que se
  ajustan (l. 716, 804-805, 889-890 [SVM tachada], 967-968, 1051, 1052-1054).
- Separar Modelización (rejillas) de Valoración (resultados/métricas) (l. 688-690, 781-782, 834,
  1042-1043, 1071-1072, 1125).
- Correlación: justificar Spearman y matizar que la multicolinealidad solo afecta a la regresión, no a
  la predicción (l. 404-405, 411-414).
- RFE: pensar si se deja; solo si con menos variables mejora de verdad (l. 433, 450-451, 457).
- Umbral: aclarar método (global Youden vs por tarea max-F1) (l. 613-614, 628-629, 1235-1236); explicar
  por qué el falso negativo es el error a minimizar (l. 1162-1163).
- SOTA: pensar si se deja; si las tablas no son el mismo split, omitirlas (l. 1252, 1293-1294).
- «svm.py» tachado (revisar) (l. 667).
