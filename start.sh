#!/usr/bin/env bash
# ============================================================================
# openkore-ai-v3 — Start Script for Multi-User Multi-Char Production Stack
# ============================================================================
# Usage:
#   ./start.sh                 Start sidecar + all 3 bots
#   ./start.sh sidecar         Start sidecar only
#   ./start.sh bot <name>      Start one bot by profile name
#   ./start.sh stop            Kill all processes
#   ./start.sh status          Show status of all processes
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR_DIR="$SCRIPT_DIR/AI_sidecar"
SIDECAR_PORT=18081
SIDECAR_LOG="$SCRIPT_DIR/logs/sidecar.log"
BOT_LOGS="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/.openkore-pids"

# Bot profiles: master=server, each gets a separate control dir
declare -A BOT_MASTER BOT_USER BOT_PASS BOT_CHAR

BOT_NAMES=("kicapmasin2" "kicapmasin" "kicapmasin3")

# Profile configurations (no secrets in git — these are set locally)
# Edit this block with your credentials before first run
BOT_MASTER["kicapmasin2"]="Asgards Glory"
BOT_USER["kicapmasin2"]="kicapmasin2"
BOT_PASS["kicapmasin2"]="sedap888"
BOT_CHAR["kicapmasin2"]="0"

BOT_MASTER["kicapmasin"]="Asgards Glory"
BOT_USER["kicapmasin"]="kicapmasin"
BOT_PASS["kicapmasin"]="b0tTib0tTi"
BOT_CHAR["kicapmasin"]="0"

BOT_MASTER["kicapmasin3"]="Asgards Glory"
BOT_USER["kicapmasin3"]="kicapmasin3"
BOT_PASS["kicapmasin3"]="sedap888"
BOT_CHAR["kicapmasin3"]="0"

# ------------------------------------------------------------------
# Colors
# ------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_cleanup() {
    local exit_code=$?
    if [ -f "$PID_FILE" ]; then
        while IFS= read -r pid; do
            [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    info "Cleanup done"
    exit $exit_code
}

_save_pid() {
    echo "$1" >> "$PID_FILE"
}

_wait_for_sidecar() {
    local timeout=60
    local elapsed=0
    info "Waiting for sidecar at port $SIDECAR_PORT..."
    while [ $elapsed -lt $timeout ]; do
        if curl -sf "http://127.0.0.1:$SIDECAR_PORT/health/live" > /dev/null 2>&1; then
            ok "Sidecar is ready"
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    err "Sidecar did not become ready within ${timeout}s (check $SIDECAR_LOG)"
    return 1
}

_setup_env() {
    if [ ! -d "$SIDECAR_DIR/venv" ]; then
        err "Virtual environment not found at $SIDECAR_DIR/venv"
        err "Run: cd $SIDECAR_DIR && python3 -m venv venv && source venv/bin/activate && pip install -e ."
        exit 1
    fi
    mkdir -p "$SCRIPT_DIR/logs" "$SIDECAR_DIR/data"
}

# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

start_sidecar() {
    if curl -sf "http://127.0.0.1:$SIDECAR_PORT/health/live" > /dev/null 2>&1; then
        ok "Sidecar already running"
        return 0
    fi
    info "Starting sidecar..."
    cd "$SIDECAR_DIR"
    source venv/bin/activate
    nohup python -m ai_sidecar.app > "$SIDECAR_LOG" 2>&1 &
    local pid=$!
    _save_pid "$pid"
    deactivate
    cd "$SCRIPT_DIR"
    ok "Sidecar started (PID $pid)"
    _wait_for_sidecar
}

start_bot() {
    local name="$1"
    local profile_dir="$SCRIPT_DIR/profiles/$name"
    local log_file="$BOT_LOGS/$name.log"

    if [ ! -d "$profile_dir" ]; then
        err "Profile not found: $profile_dir"
        err "Run: mkdir -p $profile_dir/control && cp control/config.txt $profile_dir/control/"
        return 1
    fi

    # Ensure config has credentials
    local cfg="$profile_dir/control/config.txt"
    if [ -f "$cfg" ]; then
        local master="${BOT_MASTER[$name]:-}"
        local user="${BOT_USER[$name]:-}"
        local pass="${BOT_PASS[$name]:-}"
        local char="${BOT_CHAR[$name]:-}"

        if [ -n "$master" ]; then
            # Write credentials to config
            python3 << EOF
cfg = open("$cfg").readlines()
settings = {"master": "master $master", "server": "server 0", "username": "username $user", "password": "password $pass", "char": "char $char"}
new = []
for line in cfg:
    key = line.strip().split()[0] if line.strip() and not line.strip().startswith('#') else None
    new.append(settings.get(key, line.rstrip()) + '\n')
open("$cfg", 'w').writelines(new)
EOF
        fi
    fi

    info "Starting bot: $name -> $log_file"
    cd "$SCRIPT_DIR"
    nohup perl -I src openkore.pl --control="profiles/$name/control" > "$log_file" 2>&1 &
    local pid=$!
    _save_pid "$pid"
    ok "Bot $name started (PID $pid)"
    cd "$SCRIPT_DIR"
}

stop_all() {
    if [ ! -f "$PID_FILE" ]; then
        # Try pkill as fallback
        pkill -f "openkore.pl" 2>/dev/null || true
        pkill -f "ai_sidecar.app" 2>/dev/null || true
        ok "Stopped all processes (via pkill)"
        return 0
    fi
    info "Stopping all processes..."
    while IFS= read -r pid; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null && ok "Stopped PID $pid" || true
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    # Ensure no stragglers
    pkill -f "openkore.pl" 2>/dev/null || true
    pkill -f "ai_sidecar.app" 2>/dev/null || true
    ok "All processes stopped"
}

show_status() {
    echo ""
    echo "========================================="
    echo "  openkore-ai-v3 — System Status"
    echo "========================================="
    echo ""

    # Sidecar
    if curl -sf "http://127.0.0.1:$SIDECAR_PORT/health/live" > /dev/null 2>&1; then
        local ready=$(curl -sf "http://127.0.0.1:$SIDECAR_PORT/health/ready" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'PDCA={d.get(\"pdca_running\",\"?\")} Bots={d.get(\"bots_registered\",\"?\")} Gate={d.get(\"startup_gate_mode\",\"?\")}')
except: print('parse error')
" 2>/dev/null)
        ok "Sidecar    RUNNING  port=$SIDECAR_PORT  $ready"
    else
        err "Sidecar    STOPPED"
    fi

    # Bots
    for name in "${BOT_NAMES[@]}"; do
        local log="$BOT_LOGS/$name.log"
        if ps aux | grep -v grep | grep -q "openkore.pl.*$name"; then
            # Check if connected
            local connected=""
            if [ -f "$log" ]; then
                if tail -20 "$log" | grep -q "Connected to Map Server"; then
                    connected="${GREEN}[IN-GAME]${NC}"
                elif tail -20 "$log" | grep -q "Connecting to"; then
                    connected="${YELLOW}[CONNECTING]${NC}"
                else
                    connected="${YELLOW}[STARTING]${NC}"
                fi
            fi
            echo -e "  ${GREEN}RUNNING${NC}  Bot $name  $(ps aux | grep "openkore.pl.*$name" | grep -v grep | awk '{print "PID="$2}')  $connected"
        else
            echo -e "  ${RED}STOPPED${NC}  Bot $name"
        fi
    done

    echo ""
    echo "Logs: $SCRIPT_DIR/logs/"
    echo "PID file: $PID_FILE"
    echo ""
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

trap _cleanup SIGINT SIGTERM EXIT

_setup_env

case "${1:-all}" in
    sidecar)
        start_sidecar
        show_status
        ;;
    bot)
        if [ -z "${2:-}" ]; then
            err "Usage: $0 bot <name>"
            err "Available bots: ${BOT_NAMES[*]}"
            exit 1
        fi
        start_bot "$2"
        show_status
        ;;
    stop)
        stop_all
        ;;
    status)
        show_status
        ;;
    all|start)
        info "Starting full stack..."
        start_sidecar
        for name in "${BOT_NAMES[@]}"; do
            start_bot "$name"
            sleep 5  # Stagger bot startups to avoid connection storms
        done
        show_status
        ok "All systems started. Run '$0 status' to check, '$0 stop' to stop."
        ;;
    *)
        echo "Usage: $0 {all|sidecar|bot <name>|stop|status}"
        echo ""
        echo "  all              Start sidecar + all bots"
        echo "  sidecar          Start sidecar only"
        echo "  bot <name>       Start one bot (${BOT_NAMES[*]})"
        echo "  stop             Stop all processes"
        echo "  status           Show system status"
        exit 1
        ;;
esac
