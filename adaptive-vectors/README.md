# Option-B: Vectores por paso adaptativos (eje `c`)

La Option-B hace **adaptativa la cantidad de vectores latentes que componen cada
paso de razonamiento** (`c`), en lugar de fijarla como hiperparámetro. Es el eje
**ortogonal** al de la Option-C / `pondernet/`, que adapta la *cantidad de pasos* `K`.

```
                    eje K (Option-C / pondernet)  ->  ¿cuántos pasos razonar?
problema  ->  SIM-CoT  ->                                                      ->  respuesta
                    eje c (Option-B / esta carpeta) ->  ¿cuántos vectores por paso?
```

- **Option-C / pondernet** decide *cuándo dejar de razonar* (número de pasos `K`).
- **Option-B** decide *cuántos sub-vectores gastar para construir cada paso* (`c`),
  sin cambiar la cantidad de pasos.

Las dos son complementarias y operan sobre ejes distintos del presupuesto latente.

---

## La idea

SIM-CoT entrena un **decoder auxiliar** que reconstruye el texto de cada paso de
razonamiento a partir del vector latente `z_k`. Esa reconstrucción produce una
pérdida por paso, `L_step,k`, que mide **qué tan bien el vector representa
semánticamente su paso**: alta = todavía no lo captura; baja = ya lo capturó.

Normalmente `L_step` se descarta en inferencia junto con el decoder. La Option-B
**destila esa señal en un MLP chico** que sí sobrevive a inferencia:

```
L̂_k = MLP(h_k)            # predice L_step,k a partir del estado oculto h_k
L_dist = (L̂_k - L_step,k)²  # el MLP aprende a imitar al decoder
```

En inferencia se descarta el decoder y se conserva el MLP. Cada paso se construye
como una secuencia de sub-vectores `z_{k,1}, z_{k,2}, …` generados de a uno; después
de cada sub-vector se evalúa `L̂`. Cuando `L̂` deja de bajar, el paso se considera
**maduro** y el modelo pasa al siguiente paso.

A diferencia de ACT/PonderNet (que aprenden a frenar solo desde la señal de tarea),
el MLP no infiere la madurez desde cero: **imita lo que el decoder ya sabe**. A
diferencia de la convergencia geométrica, sigue la **madurez semántica** en vez del
movimiento en el espacio latente.

---

## Objetivo de entrenamiento

```
L_total = λ_ans · L_ans + λ_step · L_step + λ_halt · L_halt
L_halt  = L_dist + λ · Σ_k n_k · σ(-L̂_k)
```

`L_step` (la pérdida del decoder de SIM-CoT) **se mantiene**: estabiliza las
representaciones latentes y es parte central de SIM-CoT. El término
`λ · Σ_k n_k · σ(-L̂_k)` penaliza gastar sub-vectores de más una vez que el paso ya
parece maduro (`σ(-L̂)` es alto cuando `L̂` es bajo). Todos los pesos son
configurables por línea de comandos (flags `ob_*` en `src/model.py`).

---

## Estado

Implementación incremental, gateada por la flag `--option_b` (apagada por defecto, de
modo que el camino SIM-CoT heredado queda intacto). El registro de decisiones y
cambios fase por fase está en `AGENTS.md`.

| Fase | Qué hace | Estado |
|------|----------|--------|
| 0 | Scaffold + flags inertes + docs | hecho |
| 1 | Probe de factibilidad (¿baja `L_step` dentro de un paso?) | pendiente |
| 2 | MLP head + `L_dist` | pendiente |
| 3 | Penalty de ponder + objetivo completo | pendiente |
| 4 | Inferencia adaptativa por sub-vectores | pendiente |
| 5 | Entrenar + evaluar en GSM8K-Aug | pendiente |

---

## Estructura

```
adaptive-vectors/
├── src/model.py        # CODI/SIM-CoT + (próximamente) MLP de destilación + loop de sub-vectores
├── train.py            # entrenamiento (heredado del harness de pondernet)
├── test.py             # evaluación; (próximamente) halting adaptativo por sub-vector
├── smoke_optionb.py    # smoke test de overfit en pocos ejemplos
├── scripts/            # fetch del decoder, setup, training/eval
├── README.md           # este archivo
└── AGENTS.md           # log de trazabilidad (decisiones + cambios por fase)
```

## Base del código

El harness (ensamblado backbone+LoRA+decoder, wiring de datos, loop de
entrenamiento/eval, y el manejo del gotcha de `gradient_checkpointing`) se copió del
directorio `pondernet/` por ser la copia probada de SIM-CoT/CODI. La cabeza de
halting de PonderNet y su objetivo (eje `K`) **no** se reutilizan: se reemplazan por
la lógica del eje `c`. No se toca `k-classifier/`, `pondernet/`, ni el camino SIM-CoT
original.
