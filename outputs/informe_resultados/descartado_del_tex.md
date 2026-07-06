# Contenido retirado de `resultados.tex` (guardado para restaurar)

Retirado a petición de Raquel (2026-07). No aparece en el PDF de momento. Para volver a
meterlo, pegar el bloque LaTeX correspondiente en su sitio.

---

## 1. Sección «Exploración: vías descartadas»

Iba justo antes de `\section{Conclusiones}`. **Nota:** la figura `fig:err` (camuflaje léxico de
Summary) **no** se retiró; se movió al cuerpo principal (sección del ganador por tarea, tras el
párrafo «El techo de Summary»), porque el texto la referencia. Aquí queda el resto.

```latex
\section{Exploración: vías descartadas}

Estas vías se probaron y no funcionaron. Se recogen porque acotan el techo del problema.

\paragraph{Características que no aportaron.} Se probaron y descartaron, todas medidas con el
protocolo honesto: LSA (TF-IDF con descomposición en valores singulares truncada
\cite{deerwester1990lsa}), que aportó $+0{,}001$ de AUC; la novedad de $n$-gramas por frase,
$+0{,}001$; y características estructurales de spaCy (arcos de dependencia, sujeto-verbo-objeto,
enlace entidad-número), que dejaron Summary en $0{,}701\to0{,}700$: ni siquiera las features
estructurales clásicas cierran Summary, señal de que el residuo es semántico y de
conocimiento del mundo, fuera del alcance de la superficie.

\paragraph{Tratamiento del desbalance que no movió la frontera.} Reequilibrar las clases no
crea señal: el sobremuestreo, el submuestreo, SMOTE \cite{chawla2002smote} y el aprendizaje
sensible al coste dejan la frontera (AUC-PR por tarea) plana o peor
(Figura~\ref{fig:frontera}). La corrección de prior de Saerens \cite{saerens2002} colapsa: su
estimación por esperanza-maximización converge a prevalencias degeneradas (0 o 1)
(Figura~\ref{fig:saerens}). La única palanca honesta contra el falso negativo es el umbral
por tarea de la sección~\ref{sec:desb-trat}.

\begin{figure}[H]
\centering
\begin{subfigure}{0.52\textwidth}\includegraphics[width=\textwidth]{desb_frontera_remuestreo}\caption{Frontera por estrategia.}\label{fig:frontera}\end{subfigure}\hfill
\begin{subfigure}{0.44\textwidth}\includegraphics[width=\textwidth]{desb_saerens_colapso}\caption{Saerens: prior estimado vs.\ real.}\label{fig:saerens}\end{subfigure}
\caption{Reequilibrar clases no mueve la frontera; la corrección de prior de Saerens colapsa.
Resultados negativos.}
\end{figure}
```

---

## 2. Bibliografía

Iba justo antes de `\end{document}`. Las entradas viven en `referencias.bib` (no se toca ese
fichero). Si se retira esto del `.tex`, las 16 `\cite` del documento se imprimen como **[?]**.

```latex
\bibliographystyle{plain}
\bibliography{referencias}
```
