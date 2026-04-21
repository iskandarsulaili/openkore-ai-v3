#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
AI_SIDECAR_DIR="$ROOT_DIR/AI_sidecar"
RUNTIME_DIR="$ROOT_DIR/.ai-sidecar-runtime"
PID_DIR="$RUNTIME_DIR/pids"
LOG_DIR="$RUNTIME_DIR/logs"

LAUNCHER_ENV_FILE="$ROOT_DIR/ai-sidecar-launcher.env"
LAUNCHER_ENV_EXAMPLE="$ROOT_DIR/ai-sidecar-launcher.env.example"

if [[ -t 1 ]]; then
  C_RESET='\033[0m'
  C_RED='\033[31m'
  C_GREEN='\033[32m'
  C_YELLOW='\033[33m'
  C_BLUE='\033[34m'
else
  C_RESET=''
  C_RED=''
  C_GREEN=''
  C_YELLOW=''
  C_BLUE=''
fi

log_info() { printf "%b[INFO ]%b %s\n" "$C_BLUE" "$C_RESET" "$*"; }
log_ok() { printf "%b[ OK  ]%b %s\n" "$C_GREEN" "$C_RESET" "$*"; }
log_warn() { printf "%b[WARN ]%b %s\n" "$C_YELLOW" "$C_RESET" "$*"; }
log_err() { printf "%b[ERROR]%b %s\n" "$C_RED" "$C_RESET" "$*" >&2; }

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    log_err "Required command not found: $cmd"
    return 1
  }
}

resolve_path() {
  local input="$1"
  if [[ "$input" = /* ]]; then
    printf '%s\n' "$input"
  else
    printf '%s\n' "$ROOT_DIR/$input"
  fi
}

is_pid_running() {
  local pid="${1:-}"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

read_pid_file() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1
  local value
  value="$(tr -d '[:space:]' <"$pid_file")"
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$value"
}

wait_for_sidecar_health() {
  local url="$1"
  local timeout="$2"
  local deadline=$((SECONDS + timeout))

  while (( SECONDS < deadline )); do
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
        return 0
      fi
    else
      if python3 - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
url = sys.argv[1]
with urllib.request.urlopen(url, timeout=2):
    pass
PY
      then
        return 0
      fi
    fi
    sleep 1
  done
  return 1
}

if [[ -f "$LAUNCHER_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$LAUNCHER_ENV_FILE"; set +a
elif [[ -f "$LAUNCHER_ENV_EXAMPLE" ]]; then
  cp "$LAUNCHER_ENV_EXAMPLE" "$LAUNCHER_ENV_FILE"
  # shellcheck disable=SC1090
  set -a; source "$LAUNCHER_ENV_FILE"; set +a
  log_warn "Created ai-sidecar-launcher.env from template. Review it when needed."
fi

SIDECAR_HOST="${SIDECAR_HOST:-127.0.0.1}"
SIDECAR_PORT="${SIDECAR_PORT:-18081}"
SIDECAR_HEALTH_PATH="${SIDECAR_HEALTH_PATH:-/v1/health/live}"
SIDECAR_START_TIMEOUT_SECONDS="${SIDECAR_START_TIMEOUT_SECONDS:-45}"

SIDECAR_LOG_FILE="$(resolve_path "${SIDECAR_LOG_FILE:-.ai-sidecar-runtime/logs/sidecar.log}")"
OPENKORE_LOG_FILE="$(resolve_path "${OPENKORE_LOG_FILE:-.ai-sidecar-runtime/logs/openkore.log}")"

SIDECAR_PID_FILE="$PID_DIR/sidecar.pid"
OPENKORE_PID_FILE="$PID_DIR/openkore.pid"

require_cmd bash
require_cmd perl
require_cmd python3

[[ -d "$AI_SIDECAR_DIR" ]] || { log_err "Missing AI_sidecar directory"; exit 1; }
[[ -x "$AI_SIDECAR_DIR/.venv/bin/python" ]] || {
  log_err "Virtual environment missing at AI_sidecar/.venv. Run ./setup-ai-sidecar.sh first."
  exit 1
}

mkdir -p "$PID_DIR" "$LOG_DIR"
mkdir -p "$(dirname "$SIDECAR_LOG_FILE")" "$(dirname "$OPENKORE_LOG_FILE")"

if existing_pid="$(read_pid_file "$SIDECAR_PID_FILE" 2>/dev/null || true)"; then
  if is_pid_running "$existing_pid"; then
    log_info "Sidecar already running (pid: $existing_pid)"
  else
    rm -f "$SIDECAR_PID_FILE"
  fi
fi

if ! existing_pid="$(read_pid_file "$SIDECAR_PID_FILE" 2>/dev/null || true)" || ! is_pid_running "$existing_pid"; then
  log_info "Starting AI sidecar"
  nohup bash -lc "cd '$AI_SIDECAR_DIR' && source .venv/bin/activate && exec python -m ai_sidecar.app" >>"$SIDECAR_LOG_FILE" 2>&1 &
  sidecar_pid=$!
  echo "$sidecar_pid" >"$SIDECAR_PID_FILE"
  log_ok "Sidecar launched (pid: $sidecar_pid)"
else
  sidecar_pid="$existing_pid"
fi

health_url="http://${SIDECAR_HOST}:${SIDECAR_PORT}${SIDECAR_HEALTH_PATH}"
log_info "Waiting for sidecar health endpoint: $health_url"
if wait_for_sidecar_health "$health_url" "$SIDECAR_START_TIMEOUT_SECONDS"; then
  log_ok "Sidecar is healthy"
else
  log_err "Sidecar did not become healthy within ${SIDECAR_START_TIMEOUT_SECONDS}s"
  exit 1
fi

if existing_pid="$(read_pid_file "$OPENKORE_PID_FILE" 2>/dev/null || true)"; then
  if is_pid_running "$existing_pid"; then
    log_warn "OpenKore already running (pid: $existing_pid). Leaving it unchanged."
    log_info "Logs: $OPENKORE_LOG_FILE"
    exit 0
  else
    rm -f "$OPENKORE_PID_FILE"
  fi
fi

read -r -a OPENKORE_ARG_ARRAY <<< "${OPENKORE_ARGS:-}"
log_info "Starting OpenKore"
nohup perl "$ROOT_DIR/openkore.pl" "${OPENKORE_ARG_ARRAY[@]}" >>"$OPENKORE_LOG_FILE" 2>&1 &
openkore_pid=$!
echo "$openkore_pid" >"$OPENKORE_PID_FILE"

sleep 1
if is_pid_running "$openkore_pid"; then
  log_ok "OpenKore launched (pid: $openkore_pid)"
  log_info "Sidecar log:  $SIDECAR_LOG_FILE"
  log_info "OpenKore log: $OPENKORE_LOG_FILE"
else
  rm -f "$OPENKORE_PID_FILE"
  log_err "OpenKore exited immediately. Check log: $OPENKORE_LOG_FILE"
  exit 1
fi

