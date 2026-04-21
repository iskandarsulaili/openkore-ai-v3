#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
PID_DIR="$ROOT_DIR/.ai-sidecar-runtime/pids"

LAUNCHER_ENV_FILE="$ROOT_DIR/ai-sidecar-launcher.env"
LAUNCHER_ENV_EXAMPLE="$ROOT_DIR/ai-sidecar-launcher.env.example"

if [[ -f "$LAUNCHER_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$LAUNCHER_ENV_FILE"; set +a
elif [[ -f "$LAUNCHER_ENV_EXAMPLE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$LAUNCHER_ENV_EXAMPLE"; set +a
fi

SIDECAR_HOST="${SIDECAR_HOST:-127.0.0.1}"
SIDECAR_PORT="${SIDECAR_PORT:-18081}"
SIDECAR_HEALTH_PATH="${SIDECAR_HEALTH_PATH:-/v1/health/live}"

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

check_component() {
  local label="$1"
  local pid_file="$2"
  if pid="$(read_pid_file "$pid_file" 2>/dev/null)"; then
    if is_pid_running "$pid"; then
      log_ok "${label}: running (pid ${pid})"
    else
      log_warn "${label}: stale pid file (pid ${pid})"
    fi
  else
    log_warn "${label}: not running"
  fi
}

check_health() {
  local url="http://${SIDECAR_HOST}:${SIDECAR_PORT}${SIDECAR_HEALTH_PATH}"
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      log_ok "Sidecar health endpoint reachable: ${url}"
    else
      log_warn "Sidecar health endpoint not reachable: ${url}"
    fi
  else
    if python3 - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=2):
    pass
PY
    then
      log_ok "Sidecar health endpoint reachable: ${url}"
    else
      log_warn "Sidecar health endpoint not reachable: ${url}"
    fi
  fi
}

mkdir -p "$PID_DIR"

check_component "AI sidecar" "$PID_DIR/sidecar.pid"
check_component "OpenKore" "$PID_DIR/openkore.pid"
check_health

