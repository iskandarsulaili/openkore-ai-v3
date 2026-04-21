#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
AI_SIDECAR_DIR="$ROOT_DIR/AI_sidecar"
CONTROL_DIR="$ROOT_DIR/control"
RUNTIME_DIR="$ROOT_DIR/.ai-sidecar-runtime"
BACKUP_DIR="$RUNTIME_DIR/backups/$(date +%Y%m%d-%H%M%S)"

SIDECAR_HOST="${SIDECAR_HOST:-127.0.0.1}"
SIDECAR_PORT="${SIDECAR_PORT:-18081}"

if [[ -t 1 ]]; then
  C_RESET='\033[0m'
  C_RED='\033[31m'
  C_GREEN='\033[32m'
  C_YELLOW='\033[33m'
  C_BLUE='\033[34m'
  C_CYAN='\033[36m'
else
  C_RESET=''
  C_RED=''
  C_GREEN=''
  C_YELLOW=''
  C_BLUE=''
  C_CYAN=''
fi

log_info() { printf "%b[INFO ]%b %s\n" "$C_BLUE" "$C_RESET" "$*"; }
log_warn() { printf "%b[WARN ]%b %s\n" "$C_YELLOW" "$C_RESET" "$*"; }
log_ok() { printf "%b[ OK  ]%b %s\n" "$C_GREEN" "$C_RESET" "$*"; }
log_err() { printf "%b[ERROR]%b %s\n" "$C_RED" "$C_RESET" "$*" >&2; }

on_error() {
  local exit_code=$?
  log_err "Setup failed (exit code: ${exit_code}). Check output above for the failing step."
  exit "$exit_code"
}
trap on_error ERR

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log_err "Required command not found: ${cmd}"
    return 1
  fi
}

backup_file() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  mkdir -p "$BACKUP_DIR"
  cp -f "$file" "$BACKUP_DIR/$(basename "$file").bak"
}

upsert_key_value_file() {
  local file="$1"
  local key="$2"
  local value="$3"
  if grep -Eq "^${key}[[:space:]]+" "$file"; then
    sed -i -E "s|^${key}[[:space:]].*$|${key} ${value}|" "$file"
  else
    printf '%s %s\n' "$key" "$value" >>"$file"
  fi
}

upsert_env_key() {
  local file="$1"
  local key="$2"
  local value="$3"
  if grep -Eq "^${key}=" "$file"; then
    sed -i -E "s|^${key}=.*$|${key}=${value}|" "$file"
  else
    printf '%s=%s\n' "$key" "$value" >>"$file"
  fi
}

ensure_plugin_in_sys() {
  local file="$1"
  local plugin="aiSidecarBridge"

  if ! grep -Eq '^loadPlugins[[:space:]]+' "$file"; then
    echo 'loadPlugins 2' >>"$file"
    log_info "Added missing loadPlugins setting to sys.txt"
  fi

  if grep -Eq '^loadPlugins[[:space:]]+0([[:space:]]|$)' "$file"; then
    sed -i -E 's/^loadPlugins[[:space:]]+0([[:space:]]|$)/loadPlugins 2\1/' "$file"
    log_warn "loadPlugins was 0; switched to 2 so aiSidecarBridge can load"
  fi

  if grep -Eq '^loadPlugins_list[[:space:]]+' "$file"; then
    local line
    line="$(grep -E '^loadPlugins_list[[:space:]]+' "$file" | head -n1)"
    if [[ "$line" != *"$plugin"* ]]; then
      local current="${line#loadPlugins_list }"
      current="${current%,}"
      if [[ -n "$current" ]]; then
        sed -i -E "s|^loadPlugins_list[[:space:]].*$|loadPlugins_list ${current},${plugin}|" "$file"
      else
        sed -i -E "s|^loadPlugins_list[[:space:]].*$|loadPlugins_list ${plugin}|" "$file"
      fi
      log_ok "Added ${plugin} into loadPlugins_list"
    else
      log_ok "${plugin} already present in loadPlugins_list"
    fi
  else
    echo "loadPlugins_list ${plugin}" >>"$file"
    log_ok "Added new loadPlugins_list with ${plugin}"
  fi
}

check_python_version() {
  local py_version
  py_version="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
if sys.version_info < (3, 11):
    raise SystemExit(42)
PY
)" || {
    log_err "python3 >= 3.11 is required"
    return 1
  }
  log_ok "Detected python3 ${py_version}"
}

log_info "Starting first-time AI sidecar + OpenKore setup"
log_info "Repository root: $ROOT_DIR"

require_cmd bash
require_cmd python3
require_cmd perl
require_cmd sed
require_cmd grep
require_cmd awk

if command -v curl >/dev/null 2>&1; then
  log_ok "curl detected (health checks will use curl)"
else
  log_warn "curl not found (fallback checks will use Python)"
fi

check_python_version

if [[ ! -d "$AI_SIDECAR_DIR" ]]; then
  log_err "AI_sidecar directory not found: $AI_SIDECAR_DIR"
  exit 1
fi
if [[ ! -d "$CONTROL_DIR" ]]; then
  log_err "control directory not found: $CONTROL_DIR"
  exit 1
fi

mkdir -p "$RUNTIME_DIR"/{logs,pids,backups}
log_ok "Ensured runtime directories under .ai-sidecar-runtime"

ENV_EXAMPLE="$AI_SIDECAR_DIR/.env.example"
ENV_FILE="$AI_SIDECAR_DIR/.env"
CONTROL_FILE="$CONTROL_DIR/ai_sidecar.txt"
POLICY_FILE="$CONTROL_DIR/ai_sidecar_policy.txt"
SYS_FILE="$CONTROL_DIR/sys.txt"
LAUNCHER_ENV_EXAMPLE="$ROOT_DIR/ai-sidecar-launcher.env.example"
LAUNCHER_ENV_FILE="$ROOT_DIR/ai-sidecar-launcher.env"

backup_file "$ENV_FILE"
backup_file "$CONTROL_FILE"
backup_file "$POLICY_FILE"
backup_file "$SYS_FILE"

if [[ -f "$ENV_FILE" ]]; then
  log_info "Using existing AI_sidecar/.env"
else
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  log_ok "Created AI_sidecar/.env from .env.example"
fi

log_info "Configuring sidecar environment defaults"
upsert_env_key "$ENV_FILE" "OPENKORE_AI_HOST" "$SIDECAR_HOST"
upsert_env_key "$ENV_FILE" "OPENKORE_AI_PORT" "$SIDECAR_PORT"
upsert_env_key "$ENV_FILE" "OPENKORE_AI_SQLITE_PATH" "AI_sidecar/data/sidecar.sqlite"
upsert_env_key "$ENV_FILE" "OPENKORE_AI_MEMORY_OPENMEMORY_PATH" "AI_sidecar/data/openmemory.sqlite"

if [[ ! -f "$CONTROL_FILE" ]]; then
  cp "$ROOT_DIR/ai-sidecar-control.template.txt" "$CONTROL_FILE"
  log_ok "Created control/ai_sidecar.txt from template"
fi

if [[ ! -f "$POLICY_FILE" ]]; then
  cp "$ROOT_DIR/ai-sidecar-policy.template.txt" "$POLICY_FILE"
  log_ok "Created control/ai_sidecar_policy.txt from template"
fi

log_info "Configuring OpenKore bridge control"
upsert_key_value_file "$CONTROL_FILE" "aiSidecar_enable" "1"
upsert_key_value_file "$CONTROL_FILE" "aiSidecar_baseUrl" "http://${SIDECAR_HOST}:${SIDECAR_PORT}"
upsert_key_value_file "$CONTROL_FILE" "aiSidecar_verbose" "1"

log_info "Ensuring aiSidecarBridge plugin is enabled in control/sys.txt"
ensure_plugin_in_sys "$SYS_FILE"

if [[ -f "$LAUNCHER_ENV_FILE" ]]; then
  log_info "Using existing ai-sidecar-launcher.env"
else
  cp "$LAUNCHER_ENV_EXAMPLE" "$LAUNCHER_ENV_FILE"
  log_ok "Created ai-sidecar-launcher.env from example template"
fi

log_info "Creating Python virtual environment and installing sidecar"
if [[ ! -d "$AI_SIDECAR_DIR/.venv" ]]; then
  python3 -m venv "$AI_SIDECAR_DIR/.venv"
  log_ok "Created AI_sidecar/.venv"
else
  log_info "Reusing existing AI_sidecar/.venv"
fi

# shellcheck disable=SC1091
source "$AI_SIDECAR_DIR/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$AI_SIDECAR_DIR"
python - <<'PY'
import ai_sidecar
print("ai_sidecar import check: OK")
PY

log_ok "Setup completed successfully"
log_info "Backups saved in: $BACKUP_DIR"
log_info "Next step: ./start-ai-openkore.sh"

