#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
PID_DIR="$ROOT_DIR/.ai-sidecar-runtime/pids"

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

stop_by_pidfile() {
  local label="$1"
  local pid_file="$2"
  local grace_seconds="${3:-10}"

if ! pid="$(read_pid_file "$pid_file" 2>/dev/null)"; then
    [[ -f "$pid_file" ]] && rm -f "$pid_file"
    log_warn "${label}: no pid file"
    return 0
  fi

  if ! is_pid_running "$pid"; then
    rm -f "$pid_file"
    log_warn "${label}: stale pid ${pid}; cleaned pid file"
    return 0
  fi

  log_info "${label}: sending TERM to pid ${pid}"
  kill "$pid" 2>/dev/null || true

  local i
  for ((i=0; i<grace_seconds; i++)); do
    if ! is_pid_running "$pid"; then
      rm -f "$pid_file"
      log_ok "${label}: stopped gracefully"
      return 0
    fi
    sleep 1
  done

  if is_pid_running "$pid"; then
    log_warn "${label}: still running after ${grace_seconds}s; sending KILL"
    kill -9 "$pid" 2>/dev/null || true
  fi

  if is_pid_running "$pid"; then
    log_err "${label}: unable to stop pid ${pid}"
    return 1
  fi

  rm -f "$pid_file"
  log_ok "${label}: stopped forcefully"
}

mkdir -p "$PID_DIR"

OPENKORE_PID_FILE="$PID_DIR/openkore.pid"
SIDECAR_PID_FILE="$PID_DIR/sidecar.pid"

stop_by_pidfile "OpenKore" "$OPENKORE_PID_FILE" 10
stop_by_pidfile "AI sidecar" "$SIDECAR_PID_FILE" 10

log_ok "Stop workflow completed"

