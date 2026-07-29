#!/bin/bash
# Watchdog Launcher — starts sidecar + bots + watchdog supervisor.
# This replaces start.sh for production use.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

WATCHDOG_LOG="$SCRIPT_DIR/logs/watchdog.log"
PID_FILE="$SCRIPT_DIR/.watchdog_pid"

mkdir -p "$SCRIPT_DIR/logs"

# First start the normal stack
echo "=== Starting full stack with watchdog ==="
"$SCRIPT_DIR/start.sh" 2>&1 | tail -20

echo ""
echo "=== Starting watchdog supervisor ==="

# Start watchdog daemon in background
nohup python3 -c "
import sys, os
sys.path.insert(0, '$SCRIPT_DIR/AI_sidecar')
os.chdir('$SCRIPT_DIR')
from ai_sidecar.runtime.watchdog import run_daemon
run_daemon()
" > "$WATCHDOG_LOG" 2>&1 &

WATCHDOG_PID=$!
echo $WATCHDOG_PID > "$PID_FILE"
echo "Watchdog started (PID $WATCHDOG_PID)"
echo "Log: $WATCHDOG_LOG"
echo ""
echo "Watchdog monitors bot processes every 15s and auto-restarts crashes."
echo "Circuit breaker: max 5 restarts per bot per hour."
echo ""
echo "To stop: kill $WATCHDOG_PID && ./stop.sh"
