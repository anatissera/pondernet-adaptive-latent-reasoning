# Option-A: Predictor de pasos latentes

La Option-A propone entrenar un clasificador liviano que, dada una tarea o consigna, prediga cuántos pasos de razonamiento latente k debería usar el modelo principal antes de responder.

La idea final es:

text input de la tarea -> clasificador -> k estimado -> modelo razona con ese k -> respuesta 

Esta carpeta contiene la primera etapa experimental de esa opción. Todavía no entrenamos el clasificador. Por ahora estamos barriendo distintos valores de k para ver si cambiar la cantidad de razonamiento latente realmente afecta las respuestas y la accuracy.

---

## Qué estamos probando ahora

En esta etapa corremos el mismo modelo varias veces sobre los mismos ejemplos, cambiando k.

Por ejemplo:

text mismo problema -> k=1 -> respuesta mismo problema -> k=2 -> respuesta mismo problema -> k=3 -> respuesta ... 

Después guardamos las predicciones y medimos si cada k acertó o no. Esto sirve para saber si existe una señal útil antes de entrenar el clasificador.

Si todos los valores de k dieran siempre lo mismo, no tendría mucho sentido aprender a predecir k. En cambio, si distintos ejemplos mejoran con distintos valores de k, entonces la Option-A tiene más sentido.

---

## Qué significa k

k representa la cantidad de razonamiento latente que dejamos hacer al modelo.

En CODI, k es la cantidad de iteraciones latentes internas antes de generar la respuesta.

En Coconut, k es la cantidad de etapas latentes agregadas al prompt mediante tokens especiales.

---

## Estructura de archivos

text k-classifier/ ├── data/              # datasets en formato jsonl ├── models/            # checkpoints locales ignorados por git ├── results/           # resultados de los sweeps ├── scripts/           # scripts para correr experimentos ├── src/               # lógica principal de carga, inferencia y métricas ├── requirements.txt   # dependencias del entorno └── README.md 

Archivos principales:

text src/model_runner.py    # corre el modelo con un valor dado de k src/k_sweep.py         # barre k=1...k_max sobre un dataset src/metrics.py         # calcula exact match / accuracy src/model_loaders.py   # carga CODI y Coconut desde checkpoints locales 

Scripts principales:

text scripts/run_k_sweep.py        # corre el experimento principal scripts/smoke_model_load.py   # verifica que los modelos carguen y generen scripts/prepare_gsm8k.py      # convierte GSM8K al formato esperado scripts/download_models.py  # descarga los pesos locales necesarios 

---

## Formato del dataset

Cada ejemplo del dataset debe estar en una línea JSON:

json {"id": "ex_001", "input": "What is 2 + 2?", "gold": "4"} 

Donde:

- id: identificador del ejemplo;
- input: consigna o problema;
- gold: respuesta correcta.

---

## Recursos externos

Pesos de los modelos:

- [SIM-CoT GPT-2 Coconut](https://huggingface.co/internlm/SIM_COT-GPT2-Coconut/tree/main)
- [SIM-CoT GPT-2 CODI](https://huggingface.co/internlm/SIM_COT-GPT2-CODI/tree/main)
- [GPT-2 base](https://huggingface.co/openai-community/gpt2)

Dataset:

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)

Los loaders esperan encontrar los modelos descargados localmente en estas rutas:

- `k-classifier/models/gpt2`
- `k-classifier/models/SIM_COT-GPT2-Coconut`
- `k-classifier/models/SIM_COT-GPT2-CODI`

Para descargarlos:

bash python k-classifier/scripts/download_models.py

---

## Cómo correr

Activar el entorno:

bash source .venv-option-a/bin/activate 

Descargar los modelos:

bash python k-classifier/scripts/download_models.py

Correr un smoke test:

bash python k-classifier/scripts/smoke_model_load.py --backend codi python k-classifier/scripts/smoke_model_load.py --backend coconut 

Correr un sweep con CODI:

bash python k-classifier/scripts/run_k_sweep.py \   --backend codi \   --k-max 8 \   --n-examples 100 \   --data k-classifier/data/gsm8k_test_100.jsonl \   --output k-classifier/results/gsm8k_codi_k8_n100_results.jsonl \   --model-loader src.model_loaders:load_codi 

Correr un sweep con Coconut:

bash python k-classifier/scripts/run_k_sweep.py \   --backend coconut \   --k-max 6 \   --n-examples 100 \   --data k-classifier/data/gsm8k_test_100.jsonl \   --output k-classifier/results/gsm8k_coconut_k6_n100_results.jsonl \   --model-loader src.model_loaders:load_coconut \   --c-thought 2 

---

## Resultados que se guardan

Cada sweep genera un archivo .jsonl con una fila por ejemplo. Para cada ejemplo se guardan:

- la consigna;
- la respuesta correcta;
- la predicción para cada k;
- el score para cada k;
- el k_star.

k_star es el menor valor de k que logra el mejor resultado para un ejemplo, es decir, la mínima cantidad de razonamiento latente necesaria para alcanzar su mejor respuesta. Por ejemplo, si k=1 falla pero k=2 y k=3 aciertan, entonces k_star = 2.

---

## Experimento actual

Corrimos CODI sobre 100 ejemplos de GSM8K con k_max=8.

Resultado:

text Accuracy by k: k=1: 0.3100 k=2: 0.3000 k=3: 0.2900 k=4: 0.3500 k=5: 0.3300 k=6: 0.3600 k=7: 0.3700 k=8: 0.3700  k_star distribution: k=1: 79 k=2: 10 k=3: 4 k=4: 5 k=5: 0 k=6: 1 k=7: 1 k=8: 0  Outputs changed across k: 96.00% 

Interpretación breve:

- La accuracy absoluta es modesta, pero esta etapa no busca maximizar performance final.
- Lo importante es que cambiar k cambia las respuestas en el 96% de los ejemplos.
- La accuracy sube de 31% con k=1 a 37% con k=7/8.
- En 21 de 100 ejemplos, el mejor resultado no se obtiene con k=1.

Esto sugiere que la cantidad de razonamiento latente sí afecta el resultado, por lo que tiene sentido avanzar hacia el objetivo real de la Option-A: entrenar un clasificador que prediga k para cada tarea.

---

## Qué falta hacer

Todavía falta la parte principal de Option-A:

- construir labels a partir de los sweeps, por ejemplo usando k_star;
- entrenar un clasificador liviano que reciba el input y prediga k;
- comparar ese clasificador contra baselines de k fijo;
- medir no solo accuracy, sino también costo promedio de razonamiento.

---

## Adaptive k classifier

This module trains a multi-output classifier that predicts which latent reasoning budgets are likely to solve a prompt.

Each training example is:

- input: prompt
- label: binary vector over k values, e.g. `[0,0,1,1,1,0,0,0]`

The classifier outputs one logit per k and is trained with `BCEWithLogitsLoss`.

### Build dataset

```bash
python3 k-classifier/scripts/build_k_classifier_dataset.py \
  --input k-classifier/results/k_sweep_train_full_codi.jsonl \
  --output k-classifier/data/k_classifier_train.jsonl \
  --k-min 1 \
  --k-max 8
```

### Train classifier

```bash
python3 k-classifier/scripts/train_k_classifier.py \
  --data k-classifier/data/k_classifier_train.jsonl \
  --output-dir k-classifier/results/k_classifier_distilbert \
  --model-name distilbert-base-uncased \
  --epochs 3 \
  --batch-size 16 \
  --lr 2e-5 \
  --max-length 256 \
  --threshold 0.7 \
  --fallback-k 6
```

### Predict k

```bash
python3 k-classifier/scripts/predict_k.py \
  --checkpoint k-classifier/results/k_classifier_distilbert \
  --prompt "Natalia sold clips to 48 of her friends..."
```
