#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$SCRIPT_DIR"
SIDECAR_DIR="$REPO_ROOT/AI_sidecar"
VENV_DIR="$SIDECAR_DIR/.venv"

log() {
  printf '[AI-SIDECAR] %s\n' "$1"
}

error_exit() {
  printf '[AI-SIDECAR][ERROR] %s\n' "$1" >&2
  exit 1
}

on_error() {
  local line="${1:-unknown}"
  error_exit "Script failed near line ${line}. Review the messages above and fix the reported issue."
}

trap 'on_error "$LINENO"' ERR

log "Validating repository layout..."
if [[ ! -d "$SIDECAR_DIR" ]]; then
  error_exit "Expected directory not found: AI_sidecar/. Place this script in openkore-ai-v3/ and run it again."
fi
if [[ ! -f "$SIDECAR_DIR/pyproject.toml" ]]; then
  error_exit "Expected file not found: AI_sidecar/pyproject.toml. Ensure script is run from the openkore-ai-v3 root."
fi

cd "$REPO_ROOT"

log "Locating Python 3.11+..."
PYTHON_BIN=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  error_exit "Python 3.11+ was not found. Install Python 3.11 or newer, then rerun this script."
fi

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
log "Using Python: $PYTHON_BIN ($PYTHON_VERSION)"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  log "Creating virtual environment at .venv/..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "Virtual environment already exists at .venv/."
fi

log "Installing/updating dependencies from AI_sidecar/pyproject.toml..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e "$SIDECAR_DIR"

if [[ ! -f "$SIDECAR_DIR/.env" ]]; then
  if [[ -f "$SIDECAR_DIR/.env.example" ]]; then
    log "Creating AI_sidecar/.env from AI_sidecar/.env.example..."
    cp "$SIDECAR_DIR/.env.example" "$SIDECAR_DIR/.env"
  else
    error_exit "AI_sidecar/.env.example not found, cannot create AI_sidecar/.env."
  fi
else
  log "Environment file already exists at AI_sidecar/.env."
fi

log "Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$SIDECAR_DIR"
log "Starting AI Sidecar in foreground. Press Ctrl+C to stop."

if command -v openkore-ai-sidecar >/dev/null 2>&1; then
  openkore-ai-sidecar
else
  log "CLI entrypoint not found; falling back to python -m ai_sidecar.app"
  python -m ai_sidecar.app
fi
