#!/usr/bin/env bash
# Fully detached bot launcher — survives parent shell death
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="$1"
LOG_FILE="$SCRIPT_DIR/logs/${NAME}.log"

# Double-fork + setsid to fully detach from any parent
(
    setsid perl -I "$SCRIPT_DIR/src" "$SCRIPT_DIR/openkore.pl" \
        --plugins="$SCRIPT_DIR/plugins" \
        --control="$SCRIPT_DIR/.bot_profiles/${NAME}/control" \
        >> "$LOG_FILE" 2>&1 &
) &
disown
