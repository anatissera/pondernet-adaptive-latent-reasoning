#!/usr/bin/env bash
# Bootstrap del entorno de Option-A en la VM (GCP T4, Deep Learning image).
#
# Recrea el venv `.venv-option-a` reutilizando el torch del sistema (no lo
# degrada) e instala el resto de requirements.txt. Idempotente: si el venv ya
# existe y torch importa, no lo recrea.
#
# Uso:
#   bash k-classifier/scripts/setup_session.sh            # solo venv + deps
#   bash k-classifier/scripts/setup_session.sh --with-models   # + baja checkpoints
#
# Por qué `--without-pip`: la VM no trae `python3-venv` (falta ensurepip), así
# que `python3 -m venv` normal falla. Con `--without-pip --system-site-packages`
# se reutiliza el pip del sistema sin necesidad de sudo.
set -euo pipefail

# Raíz del repo = dos niveles arriba de este script (scripts/ -> k-classifier/ -> repo).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPTION_A_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$OPTION_A_DIR")"
VENV_DIR="$REPO_ROOT/.venv-option-a"
REQ_FILE="$OPTION_A_DIR/requirements.txt"

WITH_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --with-models) WITH_MODELS=1 ;;
    *) echo "Argumento desconocido: $arg" >&2; exit 2 ;;
  esac
done

echo "==> Repo:  $REPO_ROOT"
echo "==> Venv:  $VENV_DIR"

# 1) Crear el venv si no existe (o si está roto y torch no importa).
if [ -x "$VENV_DIR/bin/python3" ] && "$VENV_DIR/bin/python3" -c "import torch" 2>/dev/null; then
  echo "==> Venv ya existe y torch importa; no se recrea."
else
  echo "==> Creando venv (--without-pip --system-site-packages)..."
  rm -rf "$VENV_DIR"
  python3 -m venv --system-site-packages --without-pip "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python3"

# 2) Instalar requirements.txt MENOS torch (no se degrada el torch del sistema).
echo "==> Instalando dependencias (sin tocar torch)..."
DEPS="$(grep -vE '^\s*(#|torch==|$)' "$REQ_FILE")"
# shellcheck disable=SC2086
"$PY" -m pip install $DEPS

# 3) Verificación.
echo "==> Verificando entorno:"
"$PY" - <<'PYEOF'
import torch, transformers, peft, accelerate, datasets
print(f"    torch        {torch.__version__} | CUDA: {torch.cuda.is_available()}")
print(f"    transformers {transformers.__version__}")
print(f"    peft         {peft.__version__}")
print(f"    accelerate   {accelerate.__version__}")
print(f"    datasets     {datasets.__version__}")
PYEOF

# 4) (Opcional) bajar los checkpoints de Hugging Face.
if [ "$WITH_MODELS" -eq 1 ]; then
  echo "==> Bajando checkpoints (download_models.py)..."
  "$PY" "$OPTION_A_DIR/scripts/download_models.py"
fi

echo
echo "==> Listo. Activá el entorno con:"
echo "      source .venv-option-a/bin/activate"
