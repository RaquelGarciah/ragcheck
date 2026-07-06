# Sesión de trabajo — informe de resultados (hasta v4)

Registro de la conversación de esta sesión: qué se decidió, qué cambió y el estado final.
Fecha: 2026-07. Rama base tras la sesión: `main` = `docs/informe-resultados` = tag `resultados_v4`.

---

## 1. Sistema "aprende mi estilo"
Montado el comando `/aprende-estilo` en `ALUCINACIONES_RAG-informe/.claude/commands/`, que reutiliza
el kit de la tesis (`../tfg-tesis/memoria/correcciones_aprendidas.md`, `ESTILO_Y_ANTIIA.md`,
`conectores.md`, `estilo_autor/`). Al invocarlo: git-diff de las ediciones de Raquel → destila
reglas → guarda muestras de voz → ejecuta encargos `% CC:FIG/TAB/DATO/TXT` → deja su versión como
baseline. Sembradas muestras de voz en `estilo_autor/raquel_resultados.md`. Contrato anclado en
memoria nativa (`estilo-raquel.md`). *Nota: el comando vive en el worktree del informe; en el repo
principal no aparece como `/aprende-estilo`.*

## 2. Worktree vs repo principal (confusión de "qué PDF es el bueno")
El PDF/`.tex` bueno vive en el worktree `ALUCINACIONES_RAG-informe`; el repo principal tenía copias
viejas. Causa raíz: `main` llevaba obsoleto. Se resolvió al final moviendo `main` al informe (§13).

## 3. Rediseño de la sección 1.1.1 (Descriptivo y caracterización)
Reordenada y con figuras nuevas (título dentro, coma decimal, español, color por tarea):
- **Donuts de clase** (global + 3 tareas): Data2txt 69,4 % vs QA/Summary 31,1 %.
- **Slope train→test** de la prevalencia (QA cae de 0,311 a 0,178).
- **Barras apiladas** de tipos de alucinación por tarea (sustituye tabla de 12 números).
- **Anatomía del tramo**: categoría de lo inventado + longitud. Corrección de rigor: solo el
  **41 %** de los tramos son entidad nombrada (13 % numérico, 12,7 % temporal, 15,5 % nombres);
  el 59 % es texto libre (la prosa vieja "dominan los números" estaba inflada).
- Tabla de vistazo al dataset (3 filas reales) y definición de **prevalencia** al principio.

## 4. Renombrado de variables
`source → context`, `response → output` en 40 ficheros `.py` + `.tex`. `data.py` ya no renombra;
conserva los nombres originales del dataset. 35/35 tests verde.

## 5. Metodología (Problemas 2 y 3)
Esquemas nuevos (`figuras_conceptuales.py`): `concept_gridsearch` y `concept_umbral` (el del umbral
enseña el corte por tarea con valores reales). Figura de **generalización RFE** (`fsel_generaliza`):
los otros 4 modelos se aplanan hacia las mismas 6–8 variables → la selección no es específica de
XGBoost. Marco de trabajo reescrito estilo Sergio (Base de datos / Reentrenamiento con lista de
scripts / Librerías y reproducibilidad; repo real `github.com/RaquelGarciah/ragcheck`).

## 6. El cambio grande: protocolo honesto de umbral
**Problema detectado:** las tablas usaban el umbral de Youden fijado **sobre el test** (`ev.summary`
lo calcula sobre las probabilidades que recibe), lo que contradice la metodología (umbral en
entrenamiento) y no es trazable.

**Solución (aprobada):** `grid_tables.py` fija el umbral de Youden sobre las **OOF de train**
(`cross_validate`), incluida la SVM sobre train completo, y **regenera `test_oficial_rfe.md`**.
Resultado: **documento = informe en disco = script**. El AUC no cambia (independiente del umbral);
solo bajan un poco F1/precisión/recall/accuracy.

**Tabla comparativa final (test, umbral honesto):**

| Modelo | AUC | F1 | precisión | recall | accuracy | bal-acc | prom F1–AUC |
|---|---|---|---|---|---|---|---|
| **XGBoost** | **0,837** | 0,685 | 0,673 | 0,698 | 0,776 | 0,758 | **0,761** |
| Bosque aleatorio | 0,833 | 0,677 | 0,656 | 0,699 | 0,767 | 0,751 | 0,755 |
| SVM | 0,819 | 0,685 | 0,670 | 0,701 | 0,775 | 0,758 | 0,752 |
| KNN | 0,826 | 0,669 | 0,646 | 0,695 | 0,760 | 0,745 | 0,748 |
| Reg. logística | 0,798 | 0,654 | 0,594 | 0,726 | 0,731 | 0,730 | 0,726 |

**Cambios de narrativa** por el umbral honesto: el ganador sigue siendo **XGBoost**; el **bosque
aleatorio pasa a 2.º** y la **SVM a 3.ª** (ya no es "el mejor F1"/"la mejor accuracy", empata con
XGBoost en F1); **KNN deja de ser "el más preciso"**. Con el umbral honesto los cuatro primeros se
agrupan y la regresión logística es la excepción agresiva (mayor recall, peor precisión).

## 7. Tabla por tarea (`tab:portarea`) del ganador
XGBoost afinado + columnas con `*` para el umbral por tarea. Base (umbral global) y `*` (por tarea):

| Tarea | prevalencia | AUC | F1 | F1* | recall | recall* | accuracy | accuracy* | FN | FN* |
|---|---|---|---|---|---|---|---|---|---|---|
| Data2txt | 0,643 | 0,786 | 0,824 | 0,820 | 0,891 | 0,910 | 0,754 | 0,742 | 63 | 52 |
| QA | 0,178 | 0,817 | 0,490 | 0,463 | 0,525 | 0,831 | 0,806 | 0,657 | 76 | 27 |
| Summary | 0,227 | 0,702 | 0,358 | 0,454 | 0,284 | 0,775 | 0,769 | 0,578 | 146 | 46 |
| Global | 0,349 | 0,837 | 0,685 | 0,640 | 0,698 | 0,867 | 0,776 | 0,659 | 285 | 125 |

**Compromiso del umbral (mismo detector, dos puntos de operación):** el umbral por tarea recorta
los falsos negativos de **285 a 125** y sube el F1 por tarea de Summary (35,8→45,4), pero **baja la
accuracy global de 77,6 % a 65,9 %** (más falsas alarmas). Balancear la base fue descartado con
datos (la frontera AUC-PR no se mueve); Saerens colapsa.

## 8. Comparación con el estado del arte (reescrita)
- Quitadas repeticiones (el coste se decía 4 veces → una); fusionada la comparabilidad; fuera
  "Es importante mencionar".
- **Accuracy en tabla** (global, no por tarea): la nuestra 77,6 % entre GPT-3.5-turbo (50,7) y
  Claude-3-Sonnet (79,1).
- **Parámetros reales del modelo:** `~10^4` (300 árboles, ≈13.500 hojas), no `~10^2`. El contraste
  con los neuronales (10^8–10^12) sigue siendo enorme y ahora es correcto.
- F1 por tarea (46,3 / 82,0 / 45,4; global 68,5 con umbral global, 64,0 con umbral por tarea).

## 9. Bibliografía mínima (criterio del tutor)
En el capítulo solo las imprescindibles; las definiciones de método van al marco teórico. Quitadas
`cox1958logreg` y `breiman2001rf`. Quedan 6: `niu2024ragtruth`, `guyon2002rfe`,
`guyon2003selection`, `kovacs2025lettucedetect`, `song2024raghat`, `ravi2024lynx`. Trazabilidad
anti-plagio en `citas_retiradas.md` (retiradas + reserva; nada se borra del `.bib`).

## 10. Retirados del capítulo (guardados)
- Sección "Exploración: vías descartadas" + comandos de bibliografía → `descartado_del_tex.md`
  (la figura `fig:err` del camuflaje de Summary NO se retiró; se movió al cuerpo).
- 5 curvas ROC por modelo (se conserva la comparación conjunta ROC + PR).
- Sección "Conclusiones" (Hipótesis/Falsos negativos/Posicionamiento): la quitó Raquel (redundante).

## 11. Figuras: colocación
Todas las figuras/tablas fijadas con `[H]` (paquete `float`) para que aparezcan donde les
corresponde, no flotando a otra página.

## 12. Revisión final (verificada)
Compila limpio (30 páginas, 0 refs/citas sin resolver, 0 overfull, 0 "??"). Todas las figuras
existen. Coherencia numérica: las 5 tablas por modelo = `tab:modelos` = informe en disco. Sin
marcadores olvidados. "SOTA" no aparece en el texto (usar siempre "estado del arte").

## 13. Git — `main` es ya el tronco
- Tags: `resultados_v2`, `resultados_v3`, `resultados_v4_antes_de_revisar`, `resultados_v4`.
- `main` movido por **fast-forward** de `2450c82` a `14e51cc` (= `docs/informe-resultados` = v4),
  en local y en origin. No se borró nada (worktrees y ramas `exp/*`, `feat/fase-0` intactos).
- Trabajo único fuera del tronco (guardado en sus ramas): `feat/fase-0-datos-features` (8 commits),
  `exp/tfidf-decorrelado` (7), `exp/desbalance-decision` (1).

---

## Pendientes / cabos abiertos
- **Estilo durable:** las reglas de voz aprendidas esta sesión (interpretar no relistar cifras; no
  convencer por agotamiento; muletillas prohibidas; tabla/figura antes que prosa densa; categórica;
  intuición antes del formalismo) **no están aún destiladas** en `correcciones_aprendidas.md`.
  Varias ya existen en el kit; faltaba cruzar cuáles son nuevas.
- **Párrafo de SVM mejorado** (intuición del "pasillo", vectores de soporte, truco del núcleo): dado
  como texto, **no aplicado** al `.tex`.
- **`tab:lit`**: convive el global 68,5 (umbral global) con columnas por tarea (umbral por tarea),
  explicado en el pie. Si se quiere una sola convención, la opción es 64,0.
- **Coordinación de dos sesiones** sobre `resultados.tex`: causó la reversión de la SVM. Regla:
  una sola sesión edita el `.tex` a la vez.
- **`faithful2026`**: cita placeholder (LLM-juez poco fiable) pendiente de sustituir; no está citada
  en el capítulo.
