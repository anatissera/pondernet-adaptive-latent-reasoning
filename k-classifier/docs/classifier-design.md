# Diseño del clasificador de pasos latentes (Option-A)

## Objetivo
Predecir, dada una consigna x, cuántos pasos latentes k conviene
usar antes de responder. Entrada: r(x), una representación de la
consigna. Salida: una estimación de k.

## Representación de entrada r(x)
- **Decisión:** encoder congelado all-MiniLM-L6-v2 (sentence-transformer,
  384 dims). No se fine-tunea; se precomputan los embeddings una vez.
- **Por qué:** barato (corre en CPU), embeddings de oración de buena
  calidad sin entrenamiento, independiente del backbone SIM-CoT (se
  puede computar sin los checkpoints de CODI/Coconut).
- **Alternativas descartadas por ahora:** features manuales (baseline
  futuro), hidden state del backbone (Opción 3, más fiel pero más cara
  y acoplada; queda como trabajo futuro).

## Formulación del problema
- **Decisión:** regresión sobre k, con threshold/redondeo posterior.
- **Por qué NO multiclase:** la distribución de k* está muy
  desbalanceada (en el sweep CODI n=100: 79 ejemplos con k*=1, 10 con
  k*=2, cola larga casi vacía). Un multiclase aprendería a predecir
  siempre "1" y acertaría ~79% sin aprender señal útil. Además k es
  ordinal, no categórico: k=2 está "entre" 1 y 3, estructura que el
  multiclase ignora y la regresión respeta.
- **A explorar:** cómo mapear la predicción continua a un k entero
  (redondeo simple vs. threshold calibrado), y qué función de pérdida
  (MSE vs. pérdidas que penalicen sub-estimar k, que es más costoso
  que sobre-estimar porque deja respuestas mal).

## Etiquetas
- k*(x) = menor k que alcanza el mejor score del ejemplo (de los sweeps).
- Se unen embeddings r(x) y etiquetas k* por el `id` del ejemplo.

## Pendiente de explorar / decidir
- Tratamiento del desbalance (¿reponderar? ¿transformar k?).
- Métrica de evaluación del clasificador (no solo error en k, sino
  accuracy final del sistema usando el k predicho vs. baselines de k fijo).
- Tamaño y arquitectura del MLP.

## Evidencia empírica (sweep CODI sobre train completo, n=7473, k_max=8)

### Accuracy por k
k=1: 0.408 | k=2: 0.427 | k=3: 0.479 | k=4: 0.579
k=5: 0.637 | k=6: 0.640 | k=7: 0.637 | k=8: 0.631

La accuracy crece fuerte hasta k≈5-6 (+23 puntos sobre k=1) y luego
satura: k=7 y k=8 no mejoran (incluso bajan levemente). Esto confirma
la premisa del proyecto —existe un presupuesto latente óptimo por
debajo del máximo— y sugiere que predecir k>6 no aporta.

### Distribución de k* (menor k que alcanza el mejor score)
k=1: 4939 (66%) | k=2: 1243 | k=3: 396 | k=4: 446
k=5: 319 | k=6: 75 | k=7: 32 | k=8: 23

Fuertemente desbalanceada hacia k=1 (66%), con cola muy fina en k altos.

### Implicancias de diseño
- MSE puro sesgaría el regresor hacia k≈1 (donde está la masa). Hay que
  tratar el desbalance: reponderación, transformación del target, o
  pérdida asimétrica.
- Sub-estimar k es más costoso que sobre-estimar: predecir k menor al
  necesario deja la respuesta mal; predecir de más solo gasta cómputo.
  La pérdida debería reflejar esta asimetría.
- Las clases k=6,7,8 (75/32/23 ejemplos) son casi inaprendibles por
  falta de datos. Dado que la accuracy satura en k≈5-6, considerar
  capear el rango predicho (p. ej. k∈{1..5}) en lugar de {1..8}.
- Outputs cambian con k en 97.4% de los casos → k tiene efecto real,
  el problema de aprender a asignarlo tiene sentido.

## Resultados v1 — clasificador clásico simple (pedido: empezar simple)

Primera versión deliberadamente simple (ML clásico de scikit-learn), antes de
cualquier MLP o transformación del target. Nota: el resto del doc razona el
problema como **regresión**; esta v1 lo ataca como **multiclase** (k*∈{1..8})
para tener un punto de partida barato y un baseline duro. El resultado negativo
de abajo, de hecho, refuerza la preocupación de desbalance ya anotada.

### Pipeline (3 scripts reproducibles, en `k-classifier/scripts/`)
1. `precompute_embeddings.py` — codifica el campo `input` del sweep con el
   encoder congelado all-MiniLM-L6-v2 (384 dims, CPU) y cachea (id, embedding)
   en `cache/embeddings_minilm_train_full.npz`. No recomputa si el cache cubre
   todos los ids (~2 min para n=7473 en CPU).
2. `build_dataset.py` — une embeddings (id→vector) con etiquetas k* del sweep
   (id→k_star) por `example_id`, y hace un split estratificado 80/20 interno.
   **El test de GSM8K queda reservado**: este split sale solo del train sweep.
3. `train_classifier.py` — entrena baseline mayoritario + regresión logística +
   random forest (ambos `class_weight='balanced'`) y reporta accuracy, F1-macro,
   matriz de confusión y comparación contra el baseline. Métricas crudas en
   `results/classifier_results.json`.

### Datos
- n=7473 ejemplos (sweep CODI train completo). Split: train n=5978, val n=1495.
- El split estratificado preserva la distribución de k* en ambas particiones
  (k=1 ≈ 66.1% en train y val).

### Métricas (validación interna, n=1495)

| Modelo                          | Accuracy | F1-macro | Δacc vs base | Δf1 vs base |
|---------------------------------|----------|----------|--------------|-------------|
| Baseline mayoritario (k=1)      | 0.6609   | 0.0995   | —            | —           |
| Regresión logística (balanced)  | 0.2100   | 0.1112   | −0.4508      | +0.0118     |
| Random forest (balanced)        | 0.6609   | 0.0995   | +0.0000      | +0.0000     |

Matrices de confusión (resumen):
- **Random forest**: toda la masa cae en la columna k=1 (predice k=1 para los
  1495 ejemplos de val). Idéntico al baseline pese a `class_weight='balanced'`.
- **Regresión logística**: reparte predicciones entre clases, pero con precisión
  ~0.01–0.09 en las minoritarias y precisión/recall de k=1 cayendo a 0.64/0.19.

### ¿El clasificador aprende algo más que "predecir k=1 siempre"?
**Prácticamente no.**
- El RF colapsa literalmente al baseline (misma accuracy y F1-macro).
- La LR balanceada sube el F1-macro apenas +0.012 sobre el baseline mientras
  hunde la accuracy a 0.21: está adivinando, no discriminando señal útil.

Conclusión: el embedding congelado MiniLM + clásico simple **no captura señal de
k\* por encima de la tasa base**. Resultado negativo informativo. Antes de pasar
al MLP conviene atacar la causa (la señal x→k* puede ser débil en el embedding
de oración, y/o el desbalance domina): probar representaciones más fieles
(hidden state del backbone, Opción 3), capear el rango a k∈{1..5} donde hay
datos, o re-encuadrar como regresión ordinal — no solo cambiar de modelo.