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

### Fase 1 — Probe de factibilidad
_(pendiente)_

### Fase 2 — MLP head + L_dist
_(pendiente)_

### Fase 3 — Penalty de ponder + objetivo completo
_(pendiente)_

### Fase 4 — Inferencia adaptativa
_(pendiente)_

### Fase 5 — Entrenamiento + evaluación
_(pendiente)_
