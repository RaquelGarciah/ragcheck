---
description: Aprende del último round de ediciones de Raquel sobre resultados.tex (git diff), destila reglas de estilo para su aprobación, guarda muestras de su voz, ejecuta sus encargos % CC: y deja su versión como nuevo baseline.
---

# /aprende-estilo — bucle "Raquel edita → Claude aprende → Claude escribe como ella"

Raquel corrige a mano `resultados.tex` y marca en comentarios `% CC:` dónde quiere una
figura/tabla/dato/texto. Este comando cierra el bucle: aprende de sus correcciones y ejecuta
sus encargos. La **fuente única de estilo vive en el repo de tesis** (`../tfg-tesis/`); aquí no
se duplica nada, se alimenta.

Rutas (worktree del informe = `ALUCINACIONES_RAG-informe/`; repo tesis hermano):
- TEX: `outputs/informe_resultados/resultados.tex`
- Libro de reglas: `../tfg-tesis/memoria/correcciones_aprendidas.md`
- Checklist voz/anti-IA: `../tfg-tesis/memoria/ESTILO_Y_ANTIIA.md`
- Conectores: `../tfg-tesis/tesis_assets/conectores.md`
- Muestras de voz: `../tfg-tesis/tesis_assets/estilo_autor/`
- Lógica de destilado: `../tfg-tesis/.claude/agents/aprendiz-correcciones.md`

## Paso 0 — Cargar contexto

Lee, en este orden: `correcciones_aprendidas.md` (reglas ya vigentes, para NO duplicar) →
`ESTILO_Y_ANTIIA.md` → `conectores.md` → todas las muestras de `estilo_autor/`. Lee también el
procedimiento de `aprendiz-correcciones.md` (lo aplicas inline; no invoques el agente
cross-repo).

## Paso 1 — Diff de las ediciones de Raquel

La versión de Claude está committeada; los cambios sin commitear son los de ella:

```
git -C <worktree> diff HEAD -- outputs/informe_resultados/resultados.tex
```

Si el diff está vacío, avisa y para (no hay nada que aprender). Si hay cambios ya committeados
por ella desde el último baseline, usa `git log` para encontrar el commit anterior de Claude y
haz el diff contra ahí.

## Paso 2 — Clasificar cada hunk

Reparte cada cambio en una de cuatro cajas:
- **Prosa/estilo** — reescribió una frase sin cambiar el dato. Candidato a **regla**.
- **Cifra/contenido** — cambió un número, una cita, un hecho. NO es regla de estilo; anótalo y,
  si contradice los datos reales, avísalo.
- **Encargo `% CC:`** — comentario nuevo pidiendo trabajo (ver gramática abajo).
- **Ruido** — reordenó, corrigió un typo LaTeX. Ignóralo.

## Paso 3 — Destilar reglas (procedimiento de `aprendiz-correcciones`)

Para cada cambio de prosa/estilo:
1. **Generaliza solo lo reutilizable.** Si el cambio solo tiene sentido en esa frase, es **caso
   puntual**: anótalo, NO lo generalices.
2. **Umbral por categoría:**
   - Exclusiones ("no menciones X", "no lo enmarques como Y") y énfasis/prioridades: regla
     **inmediata, prioridad alta**, desde una sola señal.
   - Rigor, cifras, citas: regla **inmediata**.
   - Estilo/voz y claridad: una vez → **tentativa**; visto ≥2-3 → **media/alta**. Marca "Visto Nx".
3. **Conserva su ejemplo real** antes→después (es el few-shot que te hará imitarla).
4. **Enuncia en imperativo y operativo** ("parte las frases de más de ~30 palabras", no "sé más
   conciso").
5. **Dedup:** si refuerza una regla existente, propón subir su frecuencia/prioridad en vez de
   crear otra.
6. **Asigna consumidor** (redactor-tesis, voz-autor, auditor-rigor, ...).

**Presenta las reglas propuestas a Raquel para su aprobación** en el formato canónico del libro:

```
- **Regla:** <imperativo>
  - Categoría · Prioridad (alta/media/tentativa) · Visto Nx
  - Ejemplo: «antes» → «después»
  - Afecta a: <agentes>
```

Solo **con su visto bueno** añade/refuerza las reglas en `correcciones_aprendidas.md` (respetando
sus secciones: "NO mencionar", "Énfasis y prioridades", estilo, etc.). No escribas el fichero
antes de la aprobación.

## Paso 4 — Guardar muestras de voz

Copia los párrafos que ella reescribió a `../tfg-tesis/tesis_assets/estilo_autor/raquel_resultados.md`
(añade, no sobreescribas; cada muestra con una línea de contexto). Son "oro" para su registro:
`redactor-tesis` y `voz-autor` los miden antes de escribir.

## Paso 5 — Ejecutar los encargos `% CC:`

Gramática de los comentarios que Raquel deja en el `.tex`:
- `% CC:FIG <qué>` — figura nueva, o «modificar fig X: …». Genera/edita el script de
  `exploration/` correspondiente, produce la figura en `outputs/figures/`, insértala con su
  `\caption` y `\label`. Semilla 42, sin datos inventados.
- `% CC:TAB <qué>` — tabla nueva o modificación. Los números salen de datos reales
  (`outputs/reports/*.md`, la librería `ragcheck`), nunca a mano.
- `% CC:DATO <qué leer>` — un número que hay que **leer de datos reales**. Ejecuta el cálculo con
  `PYTHONPATH=src` y el python del venv principal; sustituye el marcador por el valor con un
  comentario `% fuente: ...`.
- `% CC:TXT <qué>` — reescribir/añadir prosa. Aplica primero el contrato de escritura (Paso 0).

Tras ejecutar cada encargo, borra el comentario `% CC:` correspondiente del `.tex`.

## Paso 6 — Cerrar el baseline

Cuando Raquel apruebe reglas y encargos, commitea su versión (+ los encargos ejecutados) en la
rama `docs/informe-resultados` con un mensaje `docs:` o `feat:` que resuma qué aprendiste y qué
ejecutaste. Esa versión pasa a ser la base del próximo `/aprende-estilo`.

## Contrato de escritura (para escribir CUALQUIER capítulo como ella)

Antes de redactar prosa nueva, carga SIEMPRE, en orden: `correcciones_aprendidas.md` →
`ESTILO_Y_ANTIIA.md` → `conectores.md` → todas las muestras de `estilo_autor/`. Mide sus muestras
(longitud media de frase, primera persona, conectores que usa y los que NO) e imítalas. Respeta
las exclusiones y el vocabulario prohibido. Cifras solo desde datos trazables. Es el mismo
contrato de `redactor-tesis`; está anclado en la memoria nativa (`estilo-raquel.md`).
