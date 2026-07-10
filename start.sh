#!/usr/bin/env bash
# ============================================================================
# openkore-ai-v3 — Start Script for Multi-User Multi-Char Production Stack
# ============================================================================
# Usage:
#   ./start.sh                 Start sidecar + all 3 bots + tail all logs
#   ./start.sh sidecar         Start sidecar only
#   ./start.sh bot <name>      Start one bot by profile name
#   ./start.sh stop            Kill all processes
#   ./start.sh status          Show status of all processes
#   ./start.sh tail            Tail logs of running processes
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR_DIR="$SCRIPT_DIR/AI_sidecar"
SIDECAR_PORT=18081
SIDECAR_LOG="$SCRIPT_DIR/logs/sidecar.log"
BOT_LOGS="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/.openkore-pids"

# Bot profiles
declare -A BOT_MASTER BOT_USER BOT_PASS BOT_CHAR
BOT_NAMES=("kicapmasin2" "kicapmasin" "kicapmasin3")

BOT_MASTER["kicapmasin2"]="Asgards Glory"
BOT_USER["kicapmasin2"]="kicapmasin2"
BOT_CHAR["kicapmasin2"]="0"

BOT_MASTER["kicapmasin"]="Asgards Glory"
BOT_USER["kicapmasin"]="kicapmasin"
BOT_CHAR["kicapmasin"]="0"

BOT_MASTER["kicapmasin3"]="Asgards Glory"
BOT_USER["kicapmasin3"]="kicapmasin3"
BOT_CHAR["kicapmasin3"]="0"

# ------------------------------------------------------------------
# Colors
# ------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }
label() { echo -e "\n${BOLD}$1${NC}\n"; }

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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

_load_env() {
    local env_file="$SCRIPT_DIR/.env"
    if [ -f "$env_file" ]; then
        set -a
        source "$env_file"
        set +a
    fi
    for name in "${BOT_NAMES[@]}"; do
        local var_name="BOT_${name}_PASS"
        if [ -n "${!var_name:-}" ]; then
            BOT_PASS["$name"]="${!var_name}"
        elif [ -z "${BOT_PASS[$name]:-}" ]; then
            read -s -p "Enter password for $name (or set BOT_${name}_PASS in .env): " pw
            echo ""
            BOT_PASS["$name"]="$pw"
        fi
    done
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
    local profile_dir="$SCRIPT_DIR/.bot_profiles/$name"
    local log_file="$BOT_LOGS/$name.log"

    if [ ! -d "$profile_dir" ]; then
        err "Profile not found: $profile_dir"
        return 1
    fi

    # Write credentials to config
    local cfg="$profile_dir/control/config.txt"
    if [ -f "$cfg" ]; then
        local master="${BOT_MASTER[$name]:-}"
        local user="${BOT_USER[$name]:-}"
        local pass="${BOT_PASS[$name]:-}"
        local char="${BOT_CHAR[$name]:-}"
        if [ -n "$master" ]; then
            python3 << EOF
cfg_lines = open("$cfg").readlines()
settings = {"master": "master $master", "server": "server 0", "username": "username $user", "password": "password $pass", "char": "char $char"}
new = []
for line in cfg_lines:
    key = line.strip().split()[0] if line.strip() and not line.strip().startswith('#') else None
    new.append(settings.get(key, line.rstrip()) + '\n')
open("$cfg", 'w').writelines(new)
EOF
        fi
    fi

    info "Starting bot: $name"
    cd "$SCRIPT_DIR"
    nohup perl -I src openkore.pl --plugins=plugins --control=".bot_profiles/$name/control" > "$log_file" 2>&1 &
    local pid=$!
    _save_pid "$pid"
    ok "Bot $name started (PID $pid)"
    cd "$SCRIPT_DIR"
}

stop_all() {
    info "Stopping all processes..."
    # Kill by PID file first
    if [ -f "$PID_FILE" ]; then
        while IFS= read -r pid; do
            [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    # Force-kill any remaining processes
    pkill -9 -f "openkore.pl" 2>/dev/null || true
    pkill -9 -f "ai_sidecar.app" 2>/dev/null || true
    pkill -9 -f "llama-grammar-proxy" 2>/dev/null || true
    sleep 2
    # Verify all dead
    if pgrep -f "openkore.pl\|ai_sidecar.app" > /dev/null 2>&1; then
        warn "Some processes still running — forcing kill..."
        pkill -9 -f "openkore.pl" 2>/dev/null || true
        pkill -9 -f "ai_sidecar.app" 2>/dev/null || true
        sleep 1
    fi
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

    for name in "${BOT_NAMES[@]}"; do
        local log="$BOT_LOGS/$name.log"
        if ps aux | grep -v grep | grep -q "openkore.pl.*$name"; then
            local status="${YELLOW}[STARTING]${NC}"
            if [ -f "$log" ] && tail -20 "$log" 2>/dev/null | grep -q "Connected to Map Server"; then
                status="${GREEN}[IN-GAME]${NC}"
            fi
            local pid=$(ps aux | grep "openkore.pl.*$name" | grep -v grep | awk '{print $2}')
            echo -e "  ${GREEN}RUNNING${NC}  Bot $name  PID=$pid  $status"
        else
            echo -e "  ${RED}STOPPED${NC}  Bot $name"
        fi
    done
    echo ""
    echo "Logs: $BOT_LOGS/"
    echo ""
}

# ------------------------------------------------------------------
# Console Viewer — tails all logs with color-coded prefixed labels
# ------------------------------------------------------------------

_tail_all() {
    local log_files=()
    
    # Sidecar log
    if [ -f "$SIDECAR_LOG" ]; then
        log_files+=("$SIDECAR_LOG")
    fi
    
    # Bot logs
    for name in "${BOT_NAMES[@]}"; do
        local lf="$BOT_LOGS/$name.log"
        if [ -f "$lf" ]; then
            log_files+=("$lf")
        fi
    done

    if [ ${#log_files[@]} -eq 0 ]; then
        warn "No log files found yet — waiting for output..."
        sleep 3
        _tail_all
        return
    fi

    # Build a multi-tail command with labeled prefixes
    # Use different colors per source
    local colors=("${CYAN}" "${GREEN}" "${YELLOW}" "${MAGENTA}")
    local labels=("SIDECAR" "BOT:kicapmasin2" "BOT:kicapmasin" "BOT:kicapmasin3")
    
    local pid_list=""
    local temp_dir
    temp_dir=$(mktemp -d)
    
    info "Tailing all logs (Ctrl+C to stop)..."
    echo ""
    
    # Trap Ctrl+C to kill tails and clean up
    local tail_cleanup_called=0
    _tail_cleanup() {
        [ "$tail_cleanup_called" = "1" ] && return
        tail_cleanup_called=1
        echo ""
        info "Shutting down..."
        # Kill the tail processes
        for tp in $pid_list; do
            kill "$tp" 2>/dev/null || true
        done
        rm -rf "$temp_dir"
        # Stop all processes
        stop_all
        exit 0
    }
    trap _tail_cleanup SIGINT SIGTERM

    # Start a tail for each log, each writes to a named pipe with a label prefix
    for i in "${!log_files[@]}"; do
        local lf="${log_files[$i]}"
        local color="${colors[$i]:-${NC}}"
        local label="${labels[$i]:-LOG}"
        
        # Create a named pipe for this tail
        local fifo="$temp_dir/tail_$i"
        mkfifo "$fifo"
        
        # Tail the file and prefix each line with color + label
        (
            tail -n 0 -f "$lf" 2>/dev/null | while IFS= read -r line; do
                echo -e "${color}[${label}]${NC} ${line}"
            done
        ) > "$fifo" &
        pid_list="$pid_list $!"
        
        # Read from the fifo and display
        cat "$fifo" &
        pid_list="$pid_list $!"
    done

    # Wait for any child to exit (Ctrl+C triggers the trap)
    wait
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

_setup_env

case "${1:-all}" in
    sidecar)
        # Kill old sidecar before starting new
        pkill -f "ai_sidecar.app" 2>/dev/null || true
        sleep 1
        start_sidecar
        show_status
        _tail_all
        ;;
    bot)
        _load_env
        if [ -z "${2:-}" ]; then
            err "Usage: $0 bot <name>"
            err "Available bots: ${BOT_NAMES[*]}"
            exit 1
        fi
        # Kill any existing bot with this name
        pkill -f "openkore.pl.*$2" 2>/dev/null || true
        sleep 1
        start_bot "$2"
        show_status
        _tail_all
        ;;
    stop)
        stop_all
        ;;
    status)
        show_status
        ;;
    tail)
        _tail_all
        ;;
    all|start)
        _load_env
        # Kill any previously running processes to avoid "server recognizes last connection"
        stop_all
        sleep 2
        label "OPENKORE AI V3 — Starting Full Stack"
        echo -e "  Sidecar: port ${CYAN}$SIDECAR_PORT${NC}"
        echo -e "  Bots:    ${GREEN}${BOT_NAMES[*]}${NC}"
        echo ""

        start_sidecar
        for name in "${BOT_NAMES[@]}"; do
            start_bot "$name"
            sleep 3
        done

        show_status
        echo -e "${GREEN}All systems started. Streaming console output below...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop everything.${NC}"
        echo ""
        sleep 2
        _tail_all
        ;;
    *)
        echo "Usage: $0 {all|sidecar|bot <name>|stop|status|tail}"
        echo ""
        echo "  all              Start sidecar + all bots + tail logs"
        echo "  sidecar          Start sidecar only + tail"
        echo "  bot <name>       Start one bot + tail"
        echo "  stop             Stop all processes"
        echo "  status           Show system status"
        echo "  tail             Tail logs of running processes"
        exit 1
        ;;
esac
