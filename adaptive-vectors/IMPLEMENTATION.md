# Option-B — Implementación de la `c` no-fija (vectores por paso adaptativos)

Documento de referencia para **trazar** la implementación de la adaptividad de `c` y poder
comparar resultados si se cambia algo. Todo el código vive en `adaptive-vectors/` y está gateado por
`--option_b` (apagado por defecto → camino SIM-CoT heredado intacto).

> **Glosario.** `K` = número de pasos de razonamiento (fijo). `M` = máximo de sub-vectores
> por paso (`ob_subvectors_per_step` en train; `ob_max_subvectors` en inferencia).
> `c` = número de sub-vectores efectivamente usados en un paso (n_k). `L_step` = pérdida de
> reconstrucción del texto del paso por el decoder auxiliar de SIM-CoT. `L̂` = predicción del
> MLP de `L_step`. `h_k` = último hidden state del backbone en la posición latente.

---

## 1. Idea en una frase

Cada paso de razonamiento se construye como un **bloque de hasta M sub-vectores** generados de
a uno (re-alimentando el hidden). El decoder de SIM-CoT reconstruye el texto del paso desde el
**bloque acumulado**, dando `L_step`. Un MLP **destila** `L_step` desde `h_k` y sobrevive a
inferencia (el decoder no). En inferencia, se agregan sub-vectores a un paso hasta que `L̂`
deja de bajar → el paso está "maduro" → se avanza al siguiente.

---

## 2. Entrenamiento — `CODI._forward_option_b` (`src/model.py`)

Método autocontenido; se entra por el hook al inicio de `forward()`:
`if self.option_b: return self._forward_option_b(...)`.

**Flujo:**
1. **Encode** de la pregunta → `past_key_values`, `latent_hidden` = último hidden (post-`prj`).
   Este primer hidden es el sub-vector 0 del paso 0.
2. **Textos por paso:** `get_steps(ref_input_ids, K, ...)` segmenta el CoT en K pasos usando
   los marcadores `<<` (start_ids=(16791,9959)) y `>>` (end_id=4211). `pad_steps` los alinea.
   *(Cada `<<op>>` de GSM8K-Aug = un paso; si hay >K ops se fusionan en el último; si hay <K
   se rellena con pad. Ver §6 sobre el sesgo de granularidad.)*
3. **Teacher pass** (ref) para `distill_loss` + `ref_ce` (heredados de CODI, se mantienen).
4. **Loop anidado** `for k in range(K): for j in range(M):`
   - Si no es el primer sub-vector: `codi(inputs_embeds=latent_hidden, past_key_values=...)`
     agrega el sub-vector previo al cache y produce el nuevo `latent_hidden` (post-`prj`).
   - `block.append(latent_hidden)`; `latent_block = cat(block)` (B, j+1, dim).
   - `per_ex, valid = _block_step_loss(latent_block, steps_pad_list, k)` → `L_step` por ejemplo
     reconstruyendo el texto del paso `k` desde el **bloque acumulado** (decoder).
   - `L̂ = ob_mlp(h_k)` (h_k detachado si `ob_detach_hk`).
   - Acumula: `l_step_sum += per_ex[valid].mean()`; `dist_terms += SmoothL1(L̂, sg(L_step))`;
     `halt_terms += sigmoid(-L̂)`.
5. **Respuesta:** se decodifica **una sola vez** tras el bloque completo (K·M sub-vectores) →
   `ce_loss` (answer CE) + `distill_loss` (hidden del estudiante vs teacher en la posición de
   respuesta, todas las capas). ⚠️ **La respuesta solo se entrena con el budget MÁXIMO (M por
   paso).** Ver §6 (sesgo metodológico).

**Objetivo:**
```
L = ob_lambda_ans·ce_loss + distill_loss + ref_ce
    + ob_lambda_step·L_step
    + ob_lambda_dist·( L_dist + ob_lambda_halt·Σ_k,j sigmoid(-L̂) )
L_step = mean_{k,j} per_example_CE(reconstrucción del paso k desde bloque de j vectores)
L_dist = mean_{k,j} SmoothL1( L̂_{k,j} , stopgrad(L_step_{k,j}) )
```
- **SmoothL1 (no MSE)** para `L_dist`: `L_step` es un CE de cola pesada y target no-estacionario;
  MSE colapsaba a la media / explotaba (bug encontrado en smoke; ver AGENTS.md Fase 2+3).
- `ob_mlp` = `Linear(dim,256)→ReLU→Linear(256,1)`, float32, última capa init a 0 / bias 1.0.
- El MLP regresa a `L_step` **por-ejemplo** (no batch-mean) para poder discriminar instancias.

## 3. Inferencia adaptativa (`test.py`, branch `option_b`)

Por cada uno de los K pasos: agrega sub-vectores (re-alimentando, espejando el training) y tras
cada uno evalúa `L̂`. **Frena el paso** cuando `|L̂_j − L̂_{j−1}| < ob_eps` o se alcanza
`ob_max_subvectors`; avanza al siguiente paso. Tras los K pasos, decodifica la respuesta desde
el prefijo latente completo. Reporta accuracy, vectores totales/instancia, media por posición de
paso, distribución de `n_k`, y tabla accuracy-vs-budget. **Fiel a `batch_size=1`** (con bs>1 el
cache compartido infla el cómputo de filas que ya frenaron — mismo caveat que PonderNet).

Baseline `--ob_random`: frena cada paso en `n_k ~ Uniforme[1, M_max]` (control a budget parecido).

## 4. Flags (todas en `TrainingArguments`, `src/model.py`)

| flag | default | qué hace |
|------|---------|----------|
| `option_b` | False | activa el camino Option-B |
| `ob_num_steps` (K) | 4 | nº de pasos de razonamiento |
| `ob_subvectors_per_step` (M, train) | 4 | sub-vectores fijos por paso en training |
| `ob_max_subvectors` (M_max, infer) | 4 | tope de sub-vectores por paso en inferencia |
| `ob_eps` | 0.01 | umbral de "dejó de bajar" para frenar un paso |
| `ob_mlp_hidden` | 256 | ancho del MLP |
| `ob_detach_hk` | True | stop-grad de h_k hacia el MLP (no corrompe el backbone) |
| `ob_lambda_ans/step/dist/halt` | 1/1/1/0.01 | pesos del objetivo |
| `ob_probe` | False | diagnóstico Fase-1 (curva L_step por sub-vector) |
| `ob_random` | False | baseline de halting aleatorio (eval) |
| `ob_coarse_steps` | False | segmentación **gruesa**: agrupa las ops en `ob_num_steps` buckets parejos (vía `get_steps_coarse`) en vez de 1-op-por-paso → variación de complejidad por paso para el test sin sesgo del eje c. Solo training/probe; inferencia no se afecta. |

## 5. Scripts y artefactos (dónde mirar)

- `scripts/train_gpt2_gsm8k_optionb.sh` — training (3060). Env: `K,M,BS,ACCUM,EPOCHS,MAXSAMPLES,LR,LAMBDA_HALT,HALT_HEAD_LR`.
- `scripts/eval_gpt2_gsm8k_optionb.sh` — eval (bs=1, 3060). Env: `K,MMAX,EPS`, `--ob_random`.
- `scripts/ob_eval_sweep.sh` / `ob_smoke.sh` / `ob_probe.sh` — barrido / smoke / probe.
- Checkpoints: `models/checkpoints/optionb-run{1,2}/default/gpt2/ep_3/lr_2e-05/seed_42/checkpoint-747`.
- Logs/resultados: `outputs/optionb-*`, `results/optionb-*`.
- Resultados + conclusiones: `RESULTS.md`. Bitácora por fase: `AGENTS.md`.

## 6. Decisiones que pueden sesgar el resultado (LEER antes de comparar)

Cosas de la implementación/setup actual que podrían empujar a "c satura en 2" y que un test
**sin sesgo** debería cambiar:

1. **Warm-start desde checkpoint c=1 + LoRA-only (base congelada).** En `option_b` no hay bloque
   de freezing (solo `pondernet` lo tiene), así que: GPT-2 base **congelada**, solo LoRA(r=128)
   la adapta; decoder/prj/ob_mlp entrenables. El modelo arranca en el régimen c=1 y se mueve poco.
   *Test sin sesgo:* full fine-tune (o cold start sin warm-start), más datos/épocas.
2. **Granularidad de pasos = 1 operación aritmética por paso.** `get_steps` parte por `<<...>>`.
   Cada paso de GSM8K-Aug es un `<<a op b=c>>` trivial → 2 vectores lo saturan. La variación del
   dataset está en el **nº de pasos** (1–6 ops), no en la complejidad por paso. *Test sin sesgo:*
   segmentación más **gruesa** (K menor, cada paso agrupa varias ops → complejidad variable), o
   un dataset con dificultad por-paso variable.
3. **La respuesta se entrena solo con budget máximo (M por paso).** El answer head nunca aprende a
   responder con menos vectores → sesga contra los configs de c bajo y contra que la adaptividad
   pague. *Test sin sesgo:* supervisar la respuesta a budgets variables (estilo PonderNet en el
   eje c), no solo en M.
4. **`L_step` = reconstrucción del TEXTO del paso, no "answer-readiness".** La madurez se mide por
   reconstrucción del texto (corto/trivial en GSM8K-Aug), que satura independientemente de si la
   respuesta necesita más cómputo. Puede que `L_step` no sea la señal correcta para ahorrar cómputo
   de la *respuesta*.

El plan de "retrain sin sesgo" ataca (1)-(3); (4) es conceptual y se discute en el plan.
