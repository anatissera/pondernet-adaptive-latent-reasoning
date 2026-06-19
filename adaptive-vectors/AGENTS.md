# Option-B — registro de trazabilidad (decisiones y cambios)

Bitácora fase por fase de la Option-B (eje `c`: vectores por paso adaptativos).
Cada fase registra **qué se decidió** y **qué se cambió**, con fecha. Léase junto a
`README.md` (la idea) y al plan aprobado.

> Convención: este archivo se actualiza como parte del commit de **cada** fase.

---

## Contexto (por qué existe esta rama)

El proyecto hace adaptativo el presupuesto latente de SIM-CoT en dos ejes ortogonales:

- **Eje `K` — número de pasos de razonamiento.** Lo trabaja la Option-C en
  `pondernet/` (ramas `feat/adaptive-k-from-scratch`, `feat/pondernet-*`) con una
  cabeza de halting estilo PonderNet que decide *cuándo dejar de razonar*.
- **Eje `c` — número de vectores por paso (ESTA rama, Option-B).** Mantiene fijo el
  número de pasos y adapta *cuántos sub-vectores* construyen cada paso, destilando la
  pérdida de reconstrucción por paso `L_step` del decoder auxiliar de SIM-CoT en un
  MLP que sobrevive a inferencia.

`pondernet/` y Option-B son **distintos**: comparten solo el *harness* SIM-CoT/CODI,
no la lógica de adaptación.

---

## Decisiones de diseño (acordadas con la usuaria)

- **D1 — Base del código.** Copiar el harness probado de `pondernet/` a `adaptive-vectors/`,
  remover el camino de halting de PonderNet (eje `K`) y agregar la lógica de Option-B
  (eje `c`) detrás de la flag `--option_b`. Motivo: el harness ya tiene el wiring de
  datos (`data/gsm8k_aug/train15k.jsonl`), el fetch del decoder y el manejo del gotcha
  de `gradient_checkpointing`. **No se toca** `k-classifier/`, `pondernet/`, ni el camino
  SIM-CoT original.
- **D2 — Rama.** `option-b` creada desde `main` (hermana de `option-a`), no desde las
  ramas de proposal-C, para no arrastrar el trabajo del eje `K`.
- **D3 — Secuencia: probe primero.** Antes de construir todo, una fase de diagnóstico
  read-only verifica el supuesto central (que `L_step` baja al agregar sub-vectores
  dentro de un mismo paso). Es un gate GO/NO-GO.
- **D4 — Documentación.** Mantener `README.md` (idea/contexto) y este `AGENTS.md`
  (log de decisiones y cambios) actualizados fase por fase para trazabilidad.
- **D5 — Confirmación por diff.** Todo cambio al forward pass o a la pérdida se muestra
  como diff y se espera confirmación de la usuaria antes de aplicarlo, incluso después
  de la aprobación del plan. Tras cada cambio: smoke test de overfit (4–8 ejemplos).

## Riesgos asumidos (a vigilar)

- **R1 — El supuesto de descenso intra-paso no está garantizado.** Re-alimentar el
  estado oculto puede derivar hacia el *próximo* paso en vez de refinar el actual.
  Mitigación: el gate de la Fase 1; y en entrenamiento, supervisar al decoder para
  reconstruir el texto del paso `k` desde *cada* sub-vector de ese paso, generando la
  presión hacia una curva decreciente.
- **R2 — Acople train/inferencia débil.** En entrenamiento se generan `M` sub-vectores
  fijos (para calcular los targets de regresión en batch); el umbral de halting solo
  actúa en inferencia. El penalty es un regularizador, no una pérdida-esperada-sobre-
  el-halt como en PonderNet. Aceptable para v1, anotado.
- **R3 — Estabilidad del target.** `L_step` es un CE escalar en bf16 de escala
  variable → se detacha el target de regresión y (configurable, `ob_detach_hk`) se
  hace stop-gradient de `h_k` hacia el MLP en v1.

---

## Bitácora de cambios

### Fase 0 — Scaffold + docs (sin cambio de comportamiento)
**Fecha:** 2026-06-19

- Rama `option-b` creada desde `main`.
- Copiado el harness de `pondernet/` a `adaptive-vectors/`: `src/model.py`, `train.py`,
  `test.py`, `smoke_optionb.py` (ex `smoke_pondernet.py`) y `scripts/` (solo helpers
  genéricos: `fetch_simcot_decoder.py`, `gcp_setup.sh`, `profile_batch_size.py`,
  `train_gpt2_gsm8k_pondernet.sh`, `eval_gpt2_gsm8k_pondernet.sh`). Se **descartaron**
  los scripts específicos del eje `K` (`sweep_k_recipe.sh`, `sweep_pondernet_gamma.sh`,
  `eval_gpt2_gsm8k_fixedk.sh`, etc.).
- Agregadas a `TrainingArguments` (en `src/model.py`) las flags `option_b`,
  `ob_subvectors_per_step`, `ob_mlp_hidden`, `ob_detach_hk`, `ob_lambda_ans`,
  `ob_lambda_step`, `ob_lambda_dist`, `ob_lambda_halt`, `ob_eps`, `ob_max_subvectors`,
  `ob_probe`. **Todas inertes por defecto** (`option_b=False`): el camino heredado no
  cambia hasta pasar `--option_b`.
- Escritos `README.md` y este `AGENTS.md`.
- Sin cambios al forward pass ni a la pérdida todavía.

### Fase 1 — Probe de factibilidad  →  **NO-GO** (sobre el checkpoint preentrenado)
**Fecha:** 2026-06-19 · GPU: RTX 3060 (PCI idx 2) · log: `outputs/optionb-probe/probe.log`

**Cambios (read-only, gateados por `--ob_probe`):** en `src/model.py` se agregaron
`_explain_loss_for` (extracción fiel del bloque de decoder de `forward`), `_ob_probe`
(genera M sub-vectores por paso re-alimentando el hidden y mide `L_step` del texto de
ESE paso desde cada sub-vector), y un hook al inicio de `forward`. Script:
`scripts/ob_probe.sh` (warm-start SIM-CoT, M=4, 10 batches en el 3060).

**Resultado (consistente en los 10 batches):** `L_step` **NO baja** al agregar
sub-vectores dentro de un paso — la curva media sube de forma monótona:

```
mean over non-pad steps (10 batches): subvec  0     1     2     3
                                              0.08–0.89 -> ~1.0–1.7 (sube siempre)
step 0 (típico):  0.19 -> 2.35 -> 3.17 -> 3.18   (el 1er vector reconstruye casi perfecto, luego colapsa)
steps 1–3:        ~0.8–1.2 PLANO (p.ej. 0.95 -> 0.88 -> 0.88 -> 0.95)
```

**Interpretación.** El checkpoint SIM-CoT/CODI fue entrenado con **exactamente 1
vector latente por paso**. Re-alimentar el hidden NO refina el paso actual: lo
mueve hacia el **siguiente** paso, así que reconstruir el texto del paso actual
empeora. El primer vector ya está "maduro" (step 0 ≈ 0.2); los pasos posteriores
quedan planos. → **El supuesto central de la propuesta B (que `L_step` baja a lo
largo de los sub-vectores, para que el MLP detecte madurez cuando "deja de bajar")
no se cumple en el modelo preentrenado.** El riesgo **R1** se materializó.

**Matiz importante (no entierra la idea, la recaracteriza).** El probe mide el
modelo *preentrenado*. La propuesta B inherentemente requiere **reentrenar** con
supervisión por sub-vector (Fase 2: el decoder reconstruye el texto del paso `k`
desde *cada* sub-vector). Por lo tanto B **no** es un "destilar gratis una señal que
ya existe": hay que **inducir** la dinámica multi-vector por entrenamiento. Eso
cambia el costo/riesgo y es una decisión que se elevó a la usuaria antes de invertir
en Fase 2.

**Observación adicional (eje `c` vs CODI).** El contexto de la propuesta describe
SIM-CoT con `c=2` vectores por paso, pero **este** código (CODI) usa 1 vector por
paso. El "eje c" del paper no mapea 1:1 sobre CODI; en SIM-CoT real un paso es un
bloque de `c` tokens latentes contiguos antes de que el decoder supervise ese paso.
Hacer `c` adaptativo ahí es distinto de "re-alimentar y mirar `L_step`".

**Decisión:** PAUSA en el gate GO/NO-GO (como preveía el plan). Opciones elevadas a
la usuaria: (A) seguir a Fase 2 e *inducir* el refinamiento multi-vector por
entrenamiento (apuesta de investigación, re-correr el probe después para ver si la
curva se invierte); (B) repensar la señal de madurez / cuestionar si el eje `c` tiene
headroom (los datos sugieren que c=1 ya es casi óptimo por paso); (C) reformular qué
es un "sub-vector" para acercarse al diseño real de bloques-de-`c` de SIM-CoT.

#### Probe variante BLOCK (acumulación) — segundo run
Para descartar que el resultado fuera un artefacto de medir solo el *último*
sub-vector, se agregó `_explain_loss_block`: reconstruye el texto del paso desde el
**bloque acumulado** de sub-vectores (B,j,dim), la medición estilo Option-B. El probe
ahora imprime SINGLE y BLOCK lado a lado (10 batches, 3060).

```
                subvec  0      1      2      3
SINGLE (media 10b):    ~0.51 -> 1.20 -> 1.34 -> 1.37   (sube)
BLOCK  (media 10b):    ~0.50 -> 1.07 -> 1.16 -> 1.37   (sube; ≈ SINGLE)
```

BLOCK ≈ SINGLE: el decoder (entrenado con 1 latente) **no** aprovecha los vectores
extra del bloque OOD. Confirma que el modelo preentrenado no da ventaja bajo ninguna
medición.

**Conclusión clave (validada con la usuaria):** como el checkpoint está fijado a
**c=1**, cualquier `c≠1` es OOD y el probe sobre el modelo preentrenado es a lo sumo
una pista débil — **reentrenar es obligatorio** se elija A o B. El probe no puede
predecir si funcionará; solo el entrenamiento puede.

**Dirección elegida:** **Option B (bloque-de-`c`, fiel a SIM-CoT) con reentrenamiento.**
La usuaria prefiere el rebuild de B y acepta el costo de reentrenar. Riesgo honesto a
vigilar: incluso tras reentrenar, el primer vector ya reconstruye bien cada paso
(subvec0 ≈ 0.5), así que puede haber poco *headroom* y el modelo podría aprender c=1
para todo; el penalty de ponder y la evaluación acc-vs-budget lo medirán. Esto se
re-planifica fuera del scaffold original de Fase 2/3 (ver plan de Option B abajo).

### Fase 2 — MLP head + L_dist
_(pendiente)_

### Fase 3 — Penalty de ponder + objetivo completo
_(pendiente)_

### Fase 4 — Inferencia adaptativa
_(pendiente)_

### Fase 5 — Entrenamiento + evaluación
_(pendiente)_
