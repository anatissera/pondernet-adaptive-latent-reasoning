# Consideraciones del paper

Notas internas del equipo sobre qué se decide **mostrar** y **omitir** en `paper.tex`, y
por qué. No forma parte del informe; es la memoria de las decisiones editoriales.

## Qué se omite deliberadamente del paper

### 1. El bug de `gradient_checkpointing`

Se eliminó toda referencia a este hallazgo. El paper se escribe **como si ese experimento
no hubiera existido**: no se menciona en el abstract, ni como contribución, ni como sección
de resultados, ni en limitaciones.

- Contexto (para nosotros, no para el paper): las primeras corridas quedaron capadas a
  ~15--19% porque `--gradient_checkpointing True` fuerza `use_cache=False` y rompe la
  KV-cache del bucle latente. El fix es dejarlo en `False`.
- Motivo de omitirlo: es un detalle de implementación/depuración, no un aporte científico
  del método adaptativo. Enturbia la narrativa de "adaptividad de cómputo".

### 2. El sesgo de evaluar/seleccionar sobre `test`

No se menciona en el paper la historia de que las métricas iniciales (exp 01--07) se
midieron y seleccionaron sobre el split de **test**. Regla operativa:

- **Si una métrica sólo existe validada en `test`, se considera inexistente y NO se reporta.**
  El paper no la ignora en el sentido de "no me importa": la trata como no disponible.
- Concretamente, esto ya afecta:
  - El histórico "42.23% @ ep2": era test + checkpoint borrado. **No se reporta.**
  - El "39.50%" del baseline (promedio de 5 muestreos sobre test): **no se reporta**; el
    baseline citado es 40.80% (greedy, K=6, validación).
  - Los números de **trunc-K (exp-06)**: todos test-biased. En el paper la dirección se
    menciona sólo de forma cualitativa ("no adoptada"), **sin cifras**.
- Todo lo que sí se reporta en el paper está medido en **validación (n=500, greedy, bs=1)**.

> Importante: el paper actualmente reporta **validación**, no test. Esto es una limitación
> conocida y está en la lista de TODO (ver abajo): el objetivo final es reportar sobre un
> `test` limpio (held-out, sin usar para selección).

### 3. Origen del baseline 40.80% (y su carácter provisional)

El número de referencia **40.80%** que aparece en el paper ("a la par del baseline de $K$
fijo") sale de nuestro propio experimento de baseline, no es inventado:

- Fuente: `docs/experiments/01-simcot-baselines/baseline-k6.md` (run `baseline-k6`).
- Es la evaluación del checkpoint CODI/SIM-CoT de **$K$ fijo (K=6)** de upstream, la barra
  canónica contra la que se compara cada corrida de PonderNet.
- **40.80%** = greedy, single-pass, sobre **validación (n=500)**, re-validado el 2026-06-23.
- Es un número de **validación, no de test** (por eso se reporta). El viejo **39.50%** era
  sobre `test` (1319 ej., promedio de 5 pasadas muestreadas) y por la regla de arriba se
  trata como inexistente: **no se menciona**.

> Provisional: el 40.80% es validación igual que el resto de las cifras del paper, así que
> cae bajo el TODO #1 (re-evaluar en `test` limpio). Cuando exista ese número hay que
> reemplazarlo. En el texto se cita como baseline plano, sin etiquetar el split, porque se
> quitó toda la narrativa test/validación.

## TODOs pendientes

- [ ] **Evaluar todos los modelos en `test`.** Hoy reportamos métricas de validación; no
  deberíamos. Correr la evaluación final de los checkpoints seleccionados sobre un split de
  `test` limpio (no usado para selección de hiperparámetros) y reemplazar las cifras del
  paper (baseline 40.80%, frontera exp-08, Spearman +0.675, desglose por dificultad).
- [ ] **Gráfico de la distribución de pasos de razonamiento en GSM8k-Aug.** Añadir una
  figura con el histograma de `n_expr` (número de operaciones / pasos de la anotación) sobre
  GSM8k-Aug, para justificar visualmente por qué el prior por instancia y el mapeo afín
  (más de la mitad de ejemplos con `n_i <= 1`) tienen sentido.
- [ ] **Nota al pie: diferencia de reasoning steps entre `train` y `test`.** Agregar un
  footnote que cuantifique cuánto difiere la distribución de pasos de razonamiento entre el
  split de entrenamiento y el de test, para contextualizar la generalización del halting
  aprendido.
