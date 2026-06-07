# AGENTS.md

## Qué es este proyecto
Investigación sobre **razonamiento latente adaptativo**. Objetivo final (Option-A):
entrenar un clasificador liviano que, dada una consigna, prediga cuántos pasos de
razonamiento latente `k` debe usar el modelo antes de responder
(input → clasificador → k → el modelo razona con ese k → respuesta).

Etapa actual: **barridos de k** (no hay clasificador todavía). Se corre el mismo
modelo sobre los mismos ejemplos variando k para comprobar que cambiar el
presupuesto latente cambia las respuestas/accuracy, y se calcula `k_star`
(el menor k que alcanza el mejor score por ejemplo).

## Estructura del repo
- `k-classifier/` — **el trabajo real**. Todo lo que se ejecuta vive acá.
  - `src/model_loaders.py` — reconstruye los wrappers CODI y Coconut y carga
    los checkpoints locales (los config.json de SIM-CoT vienen vacíos).
  - `src/model_runner.py` — `run_model` despacha por backend (coconut/codi).
  - `src/k_sweep.py` — barre k=1..k_max, calcula k_star, agrega métricas.
  - `src/metrics.py` — exact match numérico (último número de la salida).
  - `scripts/run_k_sweep.py` — CLI del experimento.
  - `scripts/smoke_model_load.py` — verifica carga + 1 generación (k=1).
  - `scripts/prepare_gsm8k.py` — baja GSM8K y lo pasa a {id,input,gold}.
  - `requirements.txt` — **dependencias reales** (no usar el pyproject.toml raíz).
- `SIM-CoT/` — repos vendored upstream (CODI, Coconut). **Solo se importan las
  clases de modelo de `SIM-CoT/Coconut/` (coconut.py, utils.py)**. El resto
  (scripts, test.py, train.py, args/*.yaml) tiene rutas hardcodeadas del cluster
  del autor original y NO se corre directo.
- `README.md` raíz, `main.py`, `pyproject.toml` (name=tpnlp): scaffolding de `uv`,
  ignorar para el experimento. Ojo: el pyproject pide `python >= 3.12` pero la VM
  corre 3.10 — el pyproject no representa el entorno real.

## Entorno de ejecución (VM)
- Corremos en una **VM de GCP con GPU Tesla T4 (15 GB)**, imagen Deep Learning con
  **PyTorch ya instalado a nivel sistema** (`torch 2.9.1+cu129`, CUDA 12.9).
- El binario de Python es **`python3`** (3.10.12). **NO existe `python`** — usar
  siempre `python3`.
- La VM es **Spot** (puede apagarse sola) y además **se apaga manualmente al
  terminar cada sesión** para no gastar crédito. No asumir que sigue viva entre
  sesiones; nada de estado en RAM debe darse por persistente.

## Entorno Python / dependencias
Fuente de verdad de deps: `k-classifier/requirements.txt`
(torch 2.5.1 / transformers 4.46.2 / peft 0.15.2 / accelerate 1.7.0 /
datasets 3.1.0 / safetensors 0.5.3 / numpy 2.1.3).

**Tensión de versiones (leer antes de instalar):**
- El sistema trae **torch 2.9.1+cu129** (build correcto para la T4), pero
  `requirements.txt` pide **torch 2.5.1**. **No degradar torch a ciegas:** un
  `pip install -r requirements.txt` arrastraría torch 2.5.1 y podría romper la
  build de CUDA que ya funciona en la GPU.
- A nivel sistema NO están instalados `transformers`, `peft` ni `safetensors`
  (verificado), así que igual hace falta instalar ese stack.

**Recomendación (avisar/confirmar antes de ejecutar):** crear un venv aislado
**`.venv-option-a`** con `--system-site-packages` para **reutilizar el torch 2.9.1
del sistema** e instalar el resto SIN reinstalar torch, p. ej.:

```bash
python3 -m venv --system-site-packages .venv-option-a
source .venv-option-a/bin/activate
pip install transformers==4.46.2 peft==0.15.2 accelerate==1.7.0 \
            datasets==3.1.0 safetensors==0.5.3 numpy==2.1.3 \
            tqdm PyYAML Pillow
```

- **Verificar si torch 2.5.1 es realmente necesario antes de pinearlo.** El código
  de Option-A usa APIs estables entre torch 2.5 y 2.9 (`AutoModelForCausalLM`,
  `resize_token_embeddings`, `past_key_values`, `safetensors.load_file`,
  `torch.load`, LoRA de peft). El riesgo real está en la versión de
  **transformers** (la 4.46.2 es la que el código de Coconut/CODI espera; las
  4.52+ deprecan la cache legacy que usa `coconut.py`), no en torch. Plan:
  correr el smoke test con el torch 2.9.1 del sistema; **solo** pinear torch 2.5.1
  si algo se rompe de verdad.
- **No** instalar `SIM-CoT/CODI/requirements.txt` encima (pide torch 2.7.1 /
  transformers 4.52.4 y rompería la compatibilidad). Option-A reimplementa el
  wrapper de CODI contra transformers 4.46.2, así que no necesita el código ni las
  deps de CODI. `SIM-CoT/Coconut/requirements.txt` sí está alineado (torch 2.5.1 /
  transformers 4.46.2).

## Significado de k (importante)
- **Coconut**: k = etapas latentes; se insertan `k * c_thought` tokens `<|latent|>`
  en el prompt (entre `<|start-latent|>` y `<|end-latent|>`) y se llama
  `model.generate`. `c_thought` por defecto 1 en la CLI, 2 en el smoke test /
  loader (env `OPTION_A_C_THOUGHT`).
- **CODI**: k = iteraciones latentes; bucle manual que reinyecta el último hidden
  state k veces (con proyección `prj`) antes de decodear la respuesta token a
  token. Sin tokens de texto especiales (usa bot/eot = vocab_size+1/+2).

Ambos backends entran por la misma `run_model(...)` y se diferencian por
`generation_config["backend"]`.

## Cómo correr
1. Datos: `python3 k-classifier/scripts/prepare_gsm8k.py` (requiere internet/HF).
2. Smoke: `python3 k-classifier/scripts/smoke_model_load.py --backend {codi|coconut}`.
3. Sweep:
   ```bash
   python3 k-classifier/scripts/run_k_sweep.py --backend codi --k-max 8 --n-examples 100 \
     --data k-classifier/data/gsm8k_test_100.jsonl \
     --output k-classifier/results/<nombre>.jsonl \
     --model-loader src.model_loaders:load_codi
   ```
   (para Coconut: `--model-loader src.model_loaders:load_coconut --c-thought 2`).

## Checkpoints (NO versionados — `k-classifier/models/` está vacío en git)
Hay que poblar a mano antes de correr nada:
- `k-classifier/models/gpt2/`                                          (GPT-2 base HF)
- `k-classifier/models/SIM_COT-GPT2-Coconut/checkpoint_28`             (Coconut)
- `k-classifier/models/SIM_COT-GPT2-CODI/model-00001-of-00001.safetensors` (CODI)

Sin estos, ambos loaders lanzan `ModelLoadError` al inicio.

## Datos
- Formato canónico JSONL: `{"id", "input", "gold"}`.
- Versionado: solo `k-classifier/data/examples.jsonl` (2 filas de prueba).
- `gsm8k_test_100.jsonl` se genera con prepare_gsm8k.py; no está en git.

## Git / colaboración
- Repo **compartido entre 5 personas**. Trabajamos en la rama **`option-a`**.
- **No pushear a `main`.** Los cambios van a `option-a` o a ramas derivadas de ella.
- Antes de empujar, sincronizar con la rama remota (varios autores tocan lo mismo).

## Gotchas conocidos
- **Presupuesto de cómputo limitado (~USD 50 total).** Se prioriza correr lo
  mínimo necesario; nada de barridos exploratorios de más.
- **Sweeps largos van en `tmux`** para sobrevivir desconexiones de Cloud Shell y
  el carácter Spot de la VM. No lanzar corridas largas en foreground de una sesión
  interactiva.
- `k-classifier/models/` vacío → ambos loaders fallan hasta poblar los checkpoints.
- `k-classifier/data/gsm8k_test_100.jsonl` no existe en disco; hay que generarlo.
- Símlink roto: `SIM-CoT/Coconut/ckpts/gsm_cot` → ruta de cluster (`/mnt/...`).
- `SIM-CoT/CODI/test.py` tiene un `pdb.set_trace()` activo (línea ~321) y rutas
  `/mnt/...` hardcodeadas; no correrlo directo.
- Los `results/*.jsonl|csv` versionados son stubs vacíos; los números del
  "Experimento actual" del README de Option-A no se reproducen desde lo commiteado.
- Helpers muertos en model_loaders.py
  (`_filter_unsupported_from_pretrained_kwargs`, `_load_module_from_path`) no están
  cableados.

## Convenciones
- Código y wrappers nuevos: mantener el estilo de `k-classifier/src` (type hints,
  funciones chicas, errores propios tipo `ModelLoadError` / `ModelRunnerError`).
- Texto de cara al usuario / docs del repo: español (como el README de Option-A).
</content>
</invoke>
