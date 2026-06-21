# Option-B — registro de trazabilidad (decisiones y cambios)

Bitácora fase por fase de la Option-B (eje `c`: vectores por paso adaptativos).
Cada fase registra **qué se decidió** y **qué se cambió**, con fecha. Léase junto a
`README.md` (la idea) y al plan aprobado.

> Convención: este archivo se actualiza como parte del commit de **cada** fase.

---

## FASE NUEVA (2026-06-19) — Retrain SIN SESGO: cold-start + pasos gruesos

**Por qué.** El resultado negativo (c satura en c=2) pudo estar sesgado por: (1)
warm-start del checkpoint SIM-CoT c=1 + LoRA-only (base congelada) + solo 3 épocas →
el modelo nunca salió del régimen c=1; (2) granularidad atómica: 1 op = 1 paso, cada
paso trivial → 2 vectores saturan. Auditoría completa en `IMPLEMENTATION.md §6`.
Decisión con la usuaria: **cold-start desde GPT-2 plano (sin checkpoint SIM-CoT,
decoder fresco, ~30 épocas) + segmentación gruesa**, en una sola corrida. Trabajo en
worktree dedicado `/home/tpnlp/alr-anapaula-optionb` (symlinks data/models/outputs/
results al canónico; `.venv` symlinkeado al canónico, `UV_NO_SYNC=1` para no tocar el
venv compartido). Todo en la 3060.

**Cambios de código.**
- `get_steps_coarse` (src/model.py): encuentra los segmentos por-op (igual que
  `get_steps`) y los reparte en K buckets **parejos** (front-loaded) en vez de
  1-op-por-paso + merge-en-el-último. Cada paso = concatenación de su bucket (un eot
  final). `get_steps` heredado intacto. Verificado: 4 ops K=3 → [op0 op1][op2][op3]
  (parejo) vs atómico [op0][op1][op2 op3].
- Flag `ob_coarse_steps` (default False); `_forward_option_b` y `_ob_probe` eligen
  `get_steps_coarse` cuando está activo. Inferencia no se toca (no reconstruye texto).
- `scripts/train_gpt2_gsm8k_optionb_cold.sh`: GPT-2 plano, **sin `--simcot_ckpt` ni
  `--decoder_path`**, LR 3e-3, LoRA r128, 30 épocas, K=3 M=3, coarse, BS8×ACCUM8, 15k.
- `scripts/ob_smoke.sh`: env `COARSE` para smoke con pasos gruesos.

**Smoke (coarse, overfit 8, warm-start, 3060):** ce 0.039→0.002, l_step 0.92→0.001,
l_dist 0.14→0.04, L̂ sigue a L_step → wiring de segmentación gruesa correcto.

**Sanity cold (64 ej.):** sin warm-start (no carga simcot), sin OOM, ~1.4 batch/s.

**run en curso:** `optionb-cold-coarse` (tmux `ob-cold`), 7020 opt-steps (~11h),
λ_halt=0. Eval pendiente al terminar (c-curve + adaptive vs random, comparar con el
baseline warm+atómico).

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

### Fases 2 + 3 — MLP head + L_dist + penalty (objetivo Option-B completo)
**Fecha:** 2026-06-19 · GPU: RTX 3060

**Diseño elegido (rebuild self-contained).** En vez de cirugía sobre `forward()`,
se agregó `_forward_option_b()` al que se entra temprano (`if self.option_b: return
_forward_option_b(...)`). El camino SIM-CoT heredado **nunca** se ejecuta con
`--option_b`. Loop anidado: K pasos × M sub-vectores; en cada `(k,j)` el decoder
reconstruye el texto del paso desde el **bloque acumulado** (`_block_step_loss`,
per-example). El `ob_mlp` (2×Linear+ReLU) predice `L_step` por-ejemplo desde `h_k`.
Respuesta decodificada una vez tras todo el bloque (foco eje `c`, sin per-prefix
answer loss — eso es eje K).

Objetivo: `L = λ_ans·CE + distill + ref_ce + λ_step·L_step + λ_dist·(L_dist + λ_halt·Σσ(-L̂))`.
Flags nuevos: `ob_num_steps` (K). MLP en float32, `h_k` detached por defecto
(`ob_detach_hk`). `train.py`: warm-start permite `ob_mlp.*` newly-init; el grupo de
LR rápido (`HALT_HEAD_LR`) ahora cubre `ob_mlp` además de `halt_head`.

**Smoke (overfit 8 ej., 3060):**
- ce 0.22→0.003, ref_ce 0.32→0.03, l_step 1.57→~0 → arquitectura y bloque OK.
- **Bug encontrado:** con MSE, `L_dist` no convergía (target CE pesado/no-estacionario
  → MLP colapsa a la media, picos de 12.4). **Fix:** SmoothL1 (Huber) para `L_dist`
  (como el distill de CODI). Picos eliminados.
- **Segundo issue:** el MLP (head crítico en inferencia) aprendía lento en el grupo
  base-LR. **Fix:** sumar `ob_mlp` al grupo de LR rápido. Con `HALT_HEAD_LR=2e-3`:
  `L_dist` 0.42→0.03, `L̂` sigue a `L_step`, `halt_pen` sube 0.27→0.45 (σ(-L̂) registra
  madurez). Penalty-on (λ_halt=0.05) estable.

**Nota (rethink pendiente):** el overfit colapsa todos los `L_step` a ~0, así que NO
prueba si hay *headroom* (variación real de cuántos vectores necesita cada paso). Eso
solo se ve con entrenamiento real → siguiente: entrenar corto y medir si en inferencia
los vectores/paso VARÍAN entre instancias manteniendo accuracy. Si sale plano (todo
madura en j=1) → no hay nada que adaptar → rethink/report (la usuaria pidió repensar
si no hay resultados).

### Fase 4 — Inferencia adaptativa
**Fecha:** 2026-06-19

`test.py`: branch `option_b` (antes que pondernet). Por cada uno de los K pasos fijos,
agrega sub-vectores (re-alimentando el hidden) hasta que `|L̂_j - L̂_{j-1}| < ob_eps`
o se alcanza `ob_max_subvectors`; luego avanza al siguiente paso; decodifica la
respuesta tras los K pasos. La generación latente espeja el training (el sub-vector
actual se agrega al cache cuando se genera el siguiente; el último nunca se agrega →
mismo off-by-one). Fiel a `batch_size=1` (con bs>1 el cache compartido infla el
cómputo de filas que ya frenaron — mismo caveat que PonderNet). Reporta: accuracy,
avg vectores totales/instancia, media de vectores por posición de paso, distribución
de `n_k`, y tabla accuracy-vs-budget. Guarda detalle por instancia en `results/`.
Script: `scripts/eval_gpt2_gsm8k_optionb.sh` (bs=1, 3060). `__init__` ahora expone
`ob_eps`, `ob_max_subvectors`.

### Fase 5 — Entrenamiento + evaluación

**run1** (tmux `optionb-train`): K=4, M=3, BS=8×ACCUM4, 3 épocas, 8000 ej., LR 2e-5,
HALT_LR 1e-3, **λ_halt=0**. 3060, ~36 min. Loss 3.40→0.36. ckpt:
`models/checkpoints/optionb-run1/default/gpt2/ep_3/lr_2e-05/seed_42/checkpoint-747`.

**Bug en eval:** `model.to(bf16)` castea `ob_mlp` a bf16 pero le pasábamos `.float()`
→ dtype mismatch. Fix: castear input al dtype del MLP (`ob_mlp[0].weight.dtype`).

**Resultados sweep (300-subset, 3060):**

| config            | avg vecs | acc(%) |
|-------------------|----------|--------|
| fixed c=3 (eps=0) | 12.00    | 41.00  |
| adapt eps=0.02    | 10.69    | 41.00  |
| adapt eps=0.05    | 10.18    | 41.33  |
| adapt eps=0.15    | 9.23     | 41.00  |
| adapt eps=0.40    | 8.43     | 40.67  |
| **random**        | **8.01** | **41.00** |

Patrón de vectores/paso (adapt eps=0.05): s0=3.00, s1=2.66, s2=2.30, s3=2.22 — los
pasos tempranos usan más vectores; los tardíos frenan antes. Pero es mayormente un
patrón **posicional** (paso 0 = más), no per-instancia.

**Conclusión (HALLAZGO):**
1. ✅ El mecanismo Option-B funciona: entrena, halta adaptativo, recorta cómputo
   12→~9 vectores **sin pérdida de accuracy** (~41% vs baseline SIM-CoT ~39.5%).
2. ❌ **La accuracy es PLANA (~41%) en todo el rango 8–12 vectores, y el random a
   budget MENOR (8.0) iguala al adaptive.** El halting aprendido NO supera al random
   a budget igualado → **el eje `c` tiene poco headroom explotable** en GSM8K-Aug con
   GPT-2. Cada paso ya está casi saturado con 1 vector (lo anticipó el probe Fase 1).
   La diferencia 40.67–41.33% está dentro del ruido (300 ej., SE~2.8%).

Esto es exactamente el caso "rethink si no hay resultados" que pidió la usuaria: la
maquinaria es correcta pero la tarea no recompensa la adaptividad de `c`.

**Confirmación full-set (GSM8K test, 1319 ej., bs=1):**

| config            | avg vecs | acc(%) |
|-------------------|----------|--------|
| fijo c=3 (eps=0)  | 12.00    | 39.88  |
| adaptive eps=0.05 | 10.12    | 39.80  |
| random            | 8.05     | 39.35  |

Spread 0.53 % (SE~1.35 %) → plano. Adaptive = fijo con 16 % menos cómputo; random = ambos
con 33 % menos.

**¿El MLP da más vectores a lo difícil? NO.** Media de vectores: correctos=10.28,
incorrectos=10.01 (al revés de lo útil). Patrón posicional (s0=3 siempre, tardíos
frenan), no por dificultad. El halting aprendido no captura señal per-instancia.

**Curva accuracy-vs-`c` fijo (full set):** c=1(4v)=27.52 %, c=2(8v)=39.42 %,
c=3(12v)=39.88 %. Empinada 1→2, satura en c=2. → `c` importa pero su valor requerido es
~constante (2); varianza per-instancia baja = sin headroom adaptativo. Óptimo: fijo c=2.

**run2 (λ_halt=0.05, full set):** fijo c=2 = 39.65 % (8v); adaptive eps0.15 = 39.95 %
(9.2v); adaptive eps0.40 = 39.50 % (8.4v); random = 39.20 % (8v). La penalty NO desbloqueó
c=1 (eps0.40 → casi todo 2s, cero 1s). Fijo c=2 iguala/gana a todo.

**CONCLUSIÓN FINAL.** Option-B: implementación correcta y validada (entrena, halta,
recorta cómputo 16–33 % sin perder accuracy). Pero **sin headroom** para adaptividad
inteligente de `c` en GSM8K-Aug/GPT-2: la `c` requerida satura uniformemente en 2;
adaptive ≈ random ≈ fijo c=2 (todos 39.2–40.0 %, dentro del ruido); el MLP no detecta
dificultad per-instancia; la penalty no cambia nada. Recomendación: **fijar c=2** y
concentrar el esfuerzo de adaptividad en el eje `K` (Proposal C). Resumen completo en
`RESULTS.md`. Investigación cerrada (2 runs de entrenamiento + barridos full-set).

### Fase 6 — Retrain sin sesgo (cold + coarse) + diagnóstico (2026-06-21)

Para descartar que la saturación en c=2 fuera sesgo del warm-start, se reentrenó **cold
desde GPT-2 plano + segmentación gruesa** (`get_steps_coarse`, K=3 M=3, 30 ep, `train15k`).
- **Divergencia LR 3e-3** (receta CODI): explotó ep7→8 (loss 2.8→20). Fix: **LR 1e-3 +
  grad_norm 0.5 + warmup 0.05** → descenso limpio 7.5→0.77, sin spikes. `checkpoint-7020`.
- **Resultado: colapsó a ≈ azar.** c-curve full-set: c1=5.31, c2=5.46, c3=5.53 %.
  Adaptive/random (sub300) todo ~8 %, adaptive≈random.

**Diagnóstico (3 verificaciones):**
1. **No es bug de carga** — keys idénticas al ckpt warm que da 39 % (404 keys, cero diff);
   LoRA/prj/ob_mlp con pesos entrenados.
2. **Aprendió formato, no razonamiento** — genera "The answer is: N" bien formado pero con
   aritmética mal (3×3×60=540 → dice 360).
3. **Latentes inertes** — `fixed_k_eval` plano k1..k9 (~6 %); agregar vectores no ayuda.
   Contraste: warm SÍ usa latentes (c-curve sube 27.5→39.4).

**Modo de falla:** shortcut — el decoder imita el hidden del teacher y emite un número
plausible desde la pregunta sola, sin que los latentes se vuelvan load-bearing. CODI lo
evita porque su warm-start ya trae latentes funcionales (40 ep cold con objetivo simple c=1).

**Veredicto:** el ancla c=1 era **load-bearing, no solo sesgo**. El negativo del eje `c` es
**robusto en ambos regímenes**. Causa raíz = la tarea (pasos triviales uniformes), no el init.
**Próximo lever a explorar: densidad por-paso del dataset** (no el eje `c` del modelo).
Detalle completo en `RESULTS.md` § "Retrain sin sesgo".
