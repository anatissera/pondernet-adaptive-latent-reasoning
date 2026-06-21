# Option-B — Resultados (vectores por paso adaptativos, eje `c`)

**Fecha:** 2026-06-19 (actualizado 2026-06-21) · **GPU:** RTX 3060/5070/3090
**Backbone:** GPT-2 (+ SIM-CoT/CODI warm-start en el baseline; cold-start en el retrain)
**Datos:** GSM8K-Aug (train) / GSM8K test (1319 ej.)

## TL;DR

Option-B implementa y entrena de punta a punta la adaptividad del **número de vectores
por paso** (`c`): cada paso de razonamiento se construye como un bloque de hasta `M`
sub-vectores; el decoder de SIM-CoT reconstruye cada paso desde el bloque acumulado, y
un MLP destila ese `L_step` por-ejemplo para decidir en inferencia cuándo el paso está
"maduro". **El mecanismo funciona y recorta ~16–33 % del cómputo latente sin perder
accuracy.** Pero el halting aprendido **no supera al azar ni al fijo c=2 a budget
igualado** (2 runs, con y sin penalty), y el MLP **no asigna más cómputo a los problemas
más difíciles**. La curva accuracy-vs-`c` satura fuerte en c=2 (c1=27.5 %, c2=39.4 %,
c3=39.9 %): la `c` requerida es casi constante (~2) entre instancias → **poco headroom
explotable** en GSM8K-Aug con GPT-2. El óptimo simple es **fijar c=2**. Resultado
negativo limpio y reproducible: la maquinaria es correcta, la tarea no recompensa
adaptar `c`. El axis con headroom real es el número de pasos `K` (Proposal C).

> **Update 2026-06-21 (retrain sin sesgo).** Para descartar que la saturación fuera sesgo
> del warm-start c=1, se reentrenó **cold desde GPT-2 plano + segmentación gruesa**. Resultado:
> el modelo cold **colapsa a ≈ azar** (c1=5.3/c2=5.5/c3=5.5 %). Diagnóstico (ver sección
> "Retrain sin sesgo"): no es bug de carga; el modelo aprende el *formato* de la respuesta pero
> los **latentes quedan inertes** (`fixed_k_eval` plano k1..k9 ≈ 6 %) — el razonamiento latente
> no bootstrapea desde cero bajo el objetivo block-of-c. **El ancla c=1 era load-bearing, no
> solo un sesgo.** El negativo del eje `c` es robusto: donde el modelo funciona (warm), `c` es
> plano; el lever real es la **densidad por-paso del dataset**, no el eje `c`.

## Qué se construyó

- `_forward_option_b` (training): K pasos × M sub-vectores; reconstrucción por bloque
  acumulado (`_block_step_loss`, per-ejemplo); `ob_mlp` destila `L_step` (SmoothL1);
  penalty de ponder `λ·Σσ(-L̂)`. Camino SIM-CoT heredado intacto (gateado por `--option_b`).
- Inferencia adaptativa (`test.py`): por cada paso agrega sub-vectores hasta que
  `|L̂_j−L̂_{j−1}| < eps` o `M_max`; decodifica la respuesta tras K pasos. Fiel a bs=1.
- Baseline de halting **aleatorio** (`--ob_random`) para comparar a budget igualado.
- Diagnóstico Fase-1 (`--ob_probe`): mide si `L_step` baja dentro de un paso.

## Fase 1 — Probe (modelo preentrenado)

Sobre el checkpoint SIM-CoT (fijado a c=1), `L_step` **no baja** al agregar sub-vectores
dentro de un paso (sube, tanto en medición SINGLE como BLOCK). Confirmó que cualquier
`c≠1` es OOD y que **reentrenar es obligatorio**. (No es un go/no-go definitivo; solo
dice que el preentrenado no da ventaja.)

## Fase 5 — Entrenamiento + evaluación

**run1:** K=4, M=3, 3 épocas, 8 000 ej., LR 2e-5, HALT_LR 1e-3, **λ_halt=0**.
Loss 3.40 → 0.36. ckpt `…/optionb-run1/…/checkpoint-747`.

### Adaptive vs fijo vs azar (GSM8K test completo, 1319 ej., bs=1)

| config              | avg vectores | accuracy |
|---------------------|--------------|----------|
| fijo c=3 (eps=0)    | 12.00        | 39.88 %  |
| **adaptive** eps=0.05 | 10.12      | 39.80 %  |
| **random** halting  | 8.05         | 39.35 %  |

Spread = 0.53 % en todo el rango (SE ≈ 1.35 % con 1319 ej.) → **estadísticamente plano**.
Adaptive iguala a fijo con 16 % menos cómputo; random iguala a ambos con 33 % menos.

### Barrido de eps (300-subset) — tradeoff budget/accuracy

| eps   | avg vectores | accuracy |
|-------|--------------|----------|
| 0.00  | 12.00 | 41.00 % |
| 0.02  | 10.69 | 41.00 % |
| 0.05  | 10.18 | 41.33 % |
| 0.15  | 9.23  | 41.00 % |
| 0.40  | 8.43  | 40.67 % |

### ¿El MLP da más vectores a lo difícil? — NO

Media de vectores usados: **correctos = 10.28, incorrectos = 10.01** (¡al revés de lo
útil, y diferencia despreciable!). El patrón de vectores/paso es **posicional**
(s0=3.00 siempre; pasos tardíos frenan antes: s1≈2.7, s2≈2.3, s3≈2.2), no por dificultad
de la instancia. → El halting aprendido no captura una señal per-instancia.

### Curva accuracy-vs-`c` fijo (GSM8K test completo)

¿Importa `c` para nada? Forzando un `c` fijo por paso:

| c fijo | vectores totales | accuracy |
|--------|------------------|----------|
| c=1    | 4   | **27.52 %** |
| c=2    | 8   | **39.42 %** |
| c=3    | 12  | 39.88 %     |

**Hallazgo clave:** la curva es **empinada de 1→2 (+11.9 %) y plana de 2→3 (+0.46 %)**.
Satura fuerte en **c=2**. Así que `c` SÍ importa, pero la `c` *requerida* es **casi
constante (~2)** entre instancias: casi todo problema necesita 2 vectores y ninguno se
beneficia de 3. Por eso no hay headroom adaptativo — no porque `c` sea irrelevante, sino
porque su varianza per-instancia es muy baja. El óptimo es **fijo c=2** (8 vec, 39.42 %),
y el adaptive (10 vec, 39.80 %) gasta más para lo mismo.

## ¿Por qué satura en c=2? ¿Es sesgo o es la tarea?

Pregunta clave: ¿`c` satura de verdad, o nuestro setup lo sesgó a saturar? Auditoría
(detalle e implementación en `IMPLEMENTATION.md` §6):

**Contexto de los papers (agente de investigación):** CODI (el código base) usa **1 vector
por paso**; los papers SIM-CoT/Coconut usan `c_thought=2` (`SIM-CoT/Coconut/args/*.yaml`).
El checkpoint del que hicimos warm-start (`models/pretrained/simcot-gpt2-codi`) fue entrenado
con **c=1**. Los pasos se segmentan por marcadores de texto (`<<…>>`), supervisados desde el CoT.

**Sesgos en nuestro setup (a remover para un test limpio):**
1. **Warm-start c=1 + LoRA-only (base congelada).** El modelo arranca en el régimen c=1 y solo
   LoRA(r=128) lo adapta; la circuitería de razonamiento del backbone sigue siendo la de c=1.
   → **El usuario tiene razón: no deberíamos usar el checkpoint SIM-CoT; hay que reentrenar sin
   ese ancla** (cold start / full fine-tune desde GPT-2 plano).
2. **Granularidad: 1 operación aritmética = 1 paso.** En GSM8K-Aug cada paso es un `<<a op b=c>>`
   trivial → 2 vectores lo saturan. La **varianza de la tarea está en el nº de pasos (1–6 ops),
   no en la complejidad por paso**. Por eso el eje `c` es plano y el eje `K` tiene headroom.
3. **La respuesta se entrena solo a budget máximo (M/paso).** El answer head nunca aprende a
   responder con menos vectores → sesga contra c bajo y contra que la adaptividad pague.

**Veredicto del análisis:** la saturación en c=2 es **mayormente la tarea** (pasos triviales y
uniformes), con el warm-start/LoRA como ancla **secundaria**. Un retrain sin sesgo (puntos 1-3)
es necesario para afirmarlo con rigor — es exactamente lo que se hizo a continuación.

## Retrain sin sesgo: cold + coarse (resultado + diagnóstico)

**Fecha:** 2026-06-21 · **GPU:** RTX 5070 (training) + 3090 (eval). Plan: remover los sesgos
controlables (#1 warm-start c=1, #2 granularidad atómica) en una sola corrida y volver a medir.

**Setup:** GPT-2 plano (sin checkpoint SIM-CoT, decoder fresco) + segmentación **gruesa**
(`get_steps_coarse`: agrupa las ops en K=3 buckets parejos → complejidad por-paso variable),
K=3, M=3, 30 épocas, `train15k`. Script: `scripts/train_gpt2_gsm8k_optionb_cold.sh`.
- **Divergencia con LR 3e-3** (la receta cold de CODI): explotó en la época 7→8 (loss 2.8→20,
  se quedó en ~10). Re-lanzado con **LR 1e-3 + max_grad_norm 0.5 + warmup 0.05**: descenso
  monótono limpio 7.5 → **0.77**, sin spikes. Checkpoint final: `checkpoint-7020`.

**Resultado — c-curve (GSM8K test completo, 1319 ej., eps=0.0):**

| c (M_max) | vecs/inst | acc (%) cold | acc (%) warm (baseline) |
|-----------|-----------|--------------|--------------------------|
| 1         | 3.00      | **5.31**     | 27.52                    |
| 2         | 6.00      | **5.46**     | 39.42                    |
| 3         | 9.00      | **5.53**     | 39.88                    |

Adaptive/random (300-subset): **todo ~8 %**, adaptive ≈ random en todos los eps (el halting
recorta vectores 9→6 pero la accuracy no se mueve). **El modelo cold colapsó a ≈ azar.**

### Diagnóstico: por qué el cold-start falla (3 verificaciones)

1. **No es bug de carga.** El checkpoint cold tiene **estructura de keys idéntica** (404 keys,
   cero diff, cero mismatch de shape) al checkpoint warm que da 39 % por el mismo `test.py`.
   Los pesos LoRA/prj/ob_mlp están entrenados (normas grandes, no init). El loss bajó a 0.77.
2. **Aprendió el formato, no el razonamiento.** Las generaciones son respuestas **bien formadas**
   ("The answer is: N") con números de magnitud plausible, pero la aritmética está mal
   (16−3−4=9 ⇒ $18, dice 96; 3×3×60=540, dice 360). No es basura: es un modelo que da respuestas
   format-correctas pero incorrectas.
3. **Los latentes son inertes** (`fixed_k_eval`, decodifica la respuesta a cada prefijo k=1..9):

   ```
   accuracy@k (cold):  k1=7.3  k2=6.7  k3=6.0  k4=5.3  k5=5.7  k6=5.7  k7=6.0  k8=6.0  k9=5.7
   ```
   **Plana (incluso decreciente):** agregar vectores latentes no ayuda. La cadena de razonamiento
   no transporta computación. Contraste: el modelo **warm SÍ usa los latentes** — su c-curve
   **sube** (c1=27.5 → c2=39.4, +12 puntos al 2º vector).

**Modo de falla (shortcut):** desde cero, bajo el objetivo block-of-c, la optimización encuentra
un atajo — el decoder imita el hidden del teacher (`distill_loss`) y emite un número plausible
*desde la pregunta sola*, sin que los latentes se vuelvan load-bearing. El `L_step` entrena los
latentes a reconstruir el *texto* del paso, pero eso queda desacoplado de computar la respuesta.
CODI evita esto porque su warm-start **ya trae latentes funcionales** (entrenados 40 épocas en
cold con un objetivo simple: c=1, una reconstrucción).

**Veredicto actualizado:** el ancla c=1 **no era solo un sesgo — era load-bearing**. No se puede
separar "modelo que funciona" de "anclado en c=1" quitando el warm-start, porque el razonamiento
latente no bootstrapea desde cero bajo este objetivo. El negativo del eje `c` es **robusto en los
dos regímenes**: donde el modelo funciona (warm) `c` es plano; en cold no hay modelo funcional.
La causa raíz no es el init sino **la tarea**: cada paso de GSM8K-Aug es una op trivial que 2
vectores saturan. El lever real es la **densidad por-paso del dataset**, no el eje `c` del modelo.

## Conclusión y recomendación

1. **Implementación validada.** Option-B entrena, halta adaptativo por paso, y baja el
   cómputo latente ~16–33 % sin costo de accuracy. Competitivo con el baseline SIM-CoT
   (~39.5 %).
2. **Sin headroom para adaptividad *inteligente* de `c`.** La accuracy satura en c=2 de
   forma casi uniforme; adaptive ≈ random ≈ fijo c=2 a budget igualado; el MLP no asigna
   más cómputo a lo difícil (correctos 10.28 vs incorrectos 10.01). El óptimo simple es
   **fijar c=2**.
3. **Por qué.** Cada paso de razonamiento de CODI satura en ~2 vectores de forma pareja;
   el eje `c` tiene varianza per-instancia baja, a diferencia del rango dinámico del eje
   `K` (número de pasos). Refuerza la premisa del proyecto: **el axis con headroom real
   es el número de pasos (Proposal C)**.

### run2 — entrenado CON penalty (λ_halt=0.05)

¿La penalty empuja el subconjunto resoluble-con-c=1 hacia 1 vector, bajando de 8 vec a
~39 %? **No.** (GSM8K test completo)

| run2 config        | avg vecs | acc(%) |
|--------------------|----------|--------|
| fijo c=3           | 12.00    | 39.58  |
| **fijo c=2**       | 8.00     | **39.65** |
| adaptive eps=0.15  | 9.19     | 39.95  |
| adaptive eps=0.40  | 8.40     | 39.50  |
| random             | 8.05     | 39.20  |

La penalty NO desbloqueó eficiencia sub-c=2: incluso en eps=0.40 la distribución es
prácticamente todo 2s (`2:4745, 3:531`, **cero 1s**). Solo recortó un poco el paso 0
(s0=2.40). **Fijo c=2 (8 vec, 39.65 %) iguala o gana a todo el adaptive.** Confirma que
la `c` requerida es ~constante en 2 y no hay subconjunto c=1 identificable que explotar.

**Pivotes posibles** (si se quiere insistir en `c`): tareas con pasos más densos
(multi-hop real, no aritmética GSM8K), backbones mayores, o presupuestos `c` más altos
donde la saturación por-paso varíe. Pero la evidencia recomienda concentrar el esfuerzo
de adaptividad en `K`.
