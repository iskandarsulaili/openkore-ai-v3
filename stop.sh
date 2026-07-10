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

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

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

_kill_pidfile() {
    local killed=0
    if [ -f "$PID_FILE" ]; then
        while IFS= read -r pid; do
            [ -z "$pid" ] && continue
            if kill $SIGNAL "$pid" 2>/dev/null; then
                killed=$((killed + 1))
                $VERBOSE && info "Killed PID $pid (from pidfile)"
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    return $killed
}

_kill_pgrep() {
    local pattern="$1"
    local label="$2"
    local killed=0
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            if kill $SIGNAL "$pid" 2>/dev/null; then
                killed=$((killed + 1))
                $VERBOSE && info "Killed $label PID $pid"
            fi
        done
    fi
    return $killed
}

_verify_dead() {
    local remaining
    remaining=$(pgrep -f "openkore\.pl|ai_sidecar\.app|llama-grammar-proxy" 2>/dev/null || true)
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
    _kill_pidfile && total=$((total + $?))
fi

# Phase 2: Kill by process name
if [ -z "$SIGNAL" ]; then
    # Graceful first
    _kill_pgrep "openkore\.pl" "openkore" && total=$((total + $?))
    _kill_pgrep "ai_sidecar\.app" "sidecar" && total=$((total + $?))
    _kill_pgrep "llama-grammar-proxy" "grammar-proxy" && total=$((total + $?))
    sleep 1
fi

# Phase 3: Force kill anything remaining
_kill_pgrep "openkore\.pl" "openkore" && total=$((total + $?))
_kill_pgrep "ai_sidecar\.app" "sidecar" && total=$((total + $?))
_kill_pgrep "llama-grammar-proxy" "grammar-proxy" && total=$((total + $?))
sleep 1

# Phase 4: Nuclear option — SIGKILL any stragglers
remaining=$(pgrep -f "openkore\.pl|ai_sidecar\.app|llama-grammar-proxy" 2>/dev/null || true)
if [ -n "$remaining" ]; then
    for pid in $remaining; do
        kill -9 "$pid" 2>/dev/null || true
        total=$((total + 1))
        $VERBOSE && warn "Force-killed PID $pid (SIGKILL)"
    done
    sleep 1
fi

# Phase 5: Clean up pidfile
rm -f "$PID_FILE"

# Phase 6: Verify
if _verify_dead; then
    $QUIET || ok "All processes stopped ($total killed)"
else
    warn "Some processes may still be lingering — check with: ps aux | grep -E 'openkore|ai_sidecar'"
    exit 1
fi
