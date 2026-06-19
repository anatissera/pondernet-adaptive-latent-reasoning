# Option-B — Resultados (vectores por paso adaptativos, eje `c`)

**Fecha:** 2026-06-19 · **GPU:** RTX 3060 · **Backbone:** GPT-2 + SIM-CoT/CODI warm-start
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
