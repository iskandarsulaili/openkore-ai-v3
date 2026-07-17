#!/usr/bin/env bash
# ============================================================================
# openkore-ai-v3 — Complete Kill Switch
# ============================================================================
# Usage:
#   ./stop.sh          Kill all openkore + sidecar processes
#   ./stop.sh -9       Force kill (SIGKILL only, no graceful)
#   ./stop.sh -v       Verbose — show what was killed
#   ./stop.sh -q       Quiet — no output on success
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.openkore-pids"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*" >&2; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" >&2; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

VERBOSE=false
QUIET=false
SIGNAL=""

for arg in "$@"; do
    case "$arg" in
        -9) SIGNAL="-9" ;;
        -v) VERBOSE=true ;;
        -q) QUIET=true ;;
    esac
done

# Patterns to kill — covers all openkore, sidecar, and related processes
# NOTE: llama-grammar-proxy is intentionally excluded — it's a shared systemd
# service used by Hermes Agent, not owned by this project.
KILL_PATTERNS=(
    "openkore\.pl"
    "ai_sidecar\.app"
    "python3.*ai_sidecar"
    "perl.*openkore"
)

_kill_all() {
    local sig="$1"
    local label="$2"
    local killed=0
    for pattern in "${KILL_PATTERNS[@]}"; do
        local pids
        pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            for pid in $pids; do
                if kill "$sig" "$pid" 2>/dev/null; then
                    killed=$((killed + 1))
                    $VERBOSE && info "Killed $label PID $pid ($pattern)"
                fi
            done
        fi
    done
    echo "$killed"
}

_kill_pidfile() {
    local sig="$1"
    local killed=0
    if [ -f "$PID_FILE" ]; then
        while IFS= read -r pid; do
            [ -z "$pid" ] && continue
            if kill "$sig" "$pid" 2>/dev/null; then
                killed=$((killed + 1))
                $VERBOSE && info "Killed PID $pid (from pidfile)"
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    echo "$killed"
}

_verify_dead() {
    local remaining
    remaining=$(pgrep -f "openkore\.pl|perl.*openkore" 2>/dev/null || true)
    if [ -n "$remaining" ]; then
        warn "Stubborn processes still alive: $(echo "$remaining" | tr '\n' ' ')"
        return 1
    fi
    return 0
}

# ── Main ────────────────────────────────────────────────────────────────────

total=0

# Phase 1: Kill from pidfile (graceful first unless -9)
if [ -z "$SIGNAL" ]; then
    count=$(_kill_pidfile "-15")
    total=$((total + count))
fi

# Phase 2: Graceful SIGTERM to all matching processes
if [ -z "$SIGNAL" ]; then
    count=$(_kill_all "-15" "graceful")
    total=$((total + count))
    sleep 2
fi

# Phase 3: Force SIGKILL to anything still alive
count=$(_kill_all "-9" "force")
total=$((total + count))
sleep 1

# Phase 4: Nuclear — SIGKILL any stragglers by PID
remaining=$(pgrep -f "openkore\.pl|perl.*openkore" 2>/dev/null || true)
if [ -n "$remaining" ]; then
    for pid in $remaining; do
        kill -9 "$pid" 2>/dev/null || true
        total=$((total + 1))
        $VERBOSE && warn "Nuclear SIGKILL PID $pid"
    done
    sleep 1
fi

# Phase 5: Clean up pidfile
rm -f "$PID_FILE"

# Phase 6: Verify
if _verify_dead; then
    $QUIET || ok "All processes stopped ($total killed)"
    # Rotate logs: keep last 5, compress old ones
    if [ -d "$SCRIPT_DIR/logs" ]; then
        # Remove logs older than 7 days
        find "$SCRIPT_DIR/logs" -name "*.log" -mtime +7 -delete 2>/dev/null || true
        # Compress logs older than 1 day
        find "$SCRIPT_DIR/logs" -name "*.log" -mtime +1 -not -name "*.gz" -exec gzip {} \; 2>/dev/null || true
        $QUIET || info "Log rotation: cleaned logs older than 7 days, compressed older than 1 day"
    fi
else
    warn "Some processes may still be lingering — check with: ps aux | grep -E 'openkore|ai_sidecar'"
    exit 1
fi
