# Frontera exactitud–cómputo - conjunto de TEST held-out (GSM8k-Aug)

Números finales del informe sobre el **conjunto de test held-out** (n=1319). Protocolo:
adaptive halting fiel `batch_size=1`, greedy, K_max=12, GPT-2, checkpoint ep5. GPU RTX 3090.
Cada celda: **exactitud (%) @ pasos latentes promedio**.

## Nomenclatura (M0/M1/M2)

Los tres puntos de operación son el **mismo modelo adaptativo**; difieren sólo en la fuerza
γ del regularizador KL y la forma del prior por instancia (`geom_mean_i = α·n_i + β`):

| etiqueta | γ | α | β | experimento / run (repo) |
|:--------:|:--:|:--:|:--:|---|
| **M0** | 0.05 | 1.0 | 1.5 | exp-07 · `fullscope-adaptive-g0.05-b1.5-k12-ep5` |
| **M1** | 0.10 | 1.0 | 1.5 | exp-08 · `fullscope-adaptive-g0.10-b1.5-k12-ep5` |
| **M2** | 0.10 | 0.6 | 1.5 | exp-08 · `fullscope-adaptive-g0.10-a0.6-b1.5-k12-ep5` |

`baseline` es el **baseline de K fijo** (SIM-CoT/CODI, K=6, sin halting) - referencia distinta,
no una config del modelo adaptativo.

## Frontera (test, n=1319)

**baseline (K fijo):** SIM-CoT/CODI, K=6, greedy, bs=1 → **39.50% @ 6.0 pasos**.

| Umbral | M0 (γ0.05) | M1 (γ0.10, α1.0) | M2 (γ0.10, α0.6) |
|-------:|:----------:|:----------------:|:----------------:|
| 0.3 | 39.1 @ 3.34 | 39.1 @ 2.60 | 38.4 @ 2.10 |
| 0.4 | 39.6 @ 3.86 | 39.4 @ 3.16 | 39.4 @ 2.52 |
| 0.5 | 40.0 @ 4.46 | 39.4 @ 3.73 | 39.9 @ 3.01 |
| 0.8 | 40.2 @ 6.97 | 39.7 @ 6.39 | 40.0 @ 4.98 |

**Seguimiento de dificultad - Spearman(cot_steps, pasos usados), umbral 0.5:**
M0 = +0.677 · **M1 = +0.675** · M2 = +0.626.

**Pasos promedio por dificultad (M1, umbral 0.5):** escalan monótonamente con las operaciones
del problema - cot0 → 2.08, cot1 → 2.51, cot2 → 3.66, cot3 → 4.76, cot4 → 4.86, cot5 → 5.16,
cot6 → 5.67, cot7 → 5.78. Fáciles ~2 pasos, difíciles ~5–6: cómputo asignado por dificultad.

## Lectura (test)

1. **Exactitud esencialmente plana** (~39–40%) entre configuraciones y umbrales; M0 queda
   marginalmente más alto. No hay ranking de exactitud entre M0/M1/M2 en test - la elección es
   de **presupuesto de cómputo**, no de exactitud.

2. **La adaptividad de cómputo es el resultado robusto.** Frente al baseline de K fijo
   (39.50% @ 6.0 pasos), **M2 a umbral 0.5 da 39.9% @ 3.01 pasos**: paridad (+0.4 pp) a la
   **mitad del cómputo (−50% pasos)**. Frente a M0 a igual umbral (40.0% @ 4.46) es −33% pasos
   a −0.1 pp. Subir γ (M0→M1) y bajar α (M1→M2) mueven el punto de operación a la izquierda a
   exactitud de paridad.

3. **Seguimiento de dificultad fuerte y estable:** Spearman +0.675 (M1) - el modelo gasta ~2
   pasos en problemas fáciles y ~5–6 en los difíciles.

## Baseline de K fijo - exactitud por pasos de razonamiento del dataset

Fixed-K (K=6) gasta 6 pasos en **toda** entrada; su exactitud se desploma con la dificultad
(cot_steps = `cot.count('<<') - 1`), evidenciando que el presupuesto fijo no ayuda a los
problemas difíciles. `steps_used` es siempre 6 → Spearman indefinido (varianza nula), el
contraste con el +0.675 adaptativo. Reconstruido a `instance_results.json`; el accuracy
recomputado (39.50%) coincide exacto con el registrado (control de alineación).

| cot_steps (dataset) | n | exactitud (%) |
|--------------------:|--:|:-------------:|
| 0 |  65 | 44.6 |
| 1 | 357 | 60.5 |
| 2 | 364 | 46.1 |
| 3 | 290 | 25.9 |
| 4 | 138 | 17.4 |
| 5 |  57 |  5.3 |
| 6 |  21 |  4.8 |
| 7 |   9 | 22.2 |
| (sin cot) | 18 | 16.7 |

---
Artefactos crudos: `paper/results_test/<M0|M1|M2|baseline>/{thr<T>/}{summary.json,instance_results.json,eval.log,cot_steps_matrix.png,command.sh}`
(el baseline de K fijo no tiene barrido de umbral). También en `results/{01,07,08}-.../…-test/`.
Mapeo M0/M1/M2 ↔ exp/run: ver tabla de Nomenclatura arriba.
