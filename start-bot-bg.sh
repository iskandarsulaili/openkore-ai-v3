#!/bin/bash
# Start an OpenKore bot as a background process (no blocking tail)
# Usage: ./start-bot-bg.sh <profile_name>

cd /home/lot399/openkore-ai-v3
source .env 2>/dev/null

NAME="$1"
PROFILE_DIR=".bot_profiles/$NAME"
LOG_FILE="logs/$NAME.log"

if [ ! -d "$PROFILE_DIR" ]; then
    echo "ERROR: Profile not found: $PROFILE_DIR"
    exit 1
fi

# Kill existing instance
pkill -f "openkore.pl.*$NAME" 2>/dev/null
sleep 1

# Launch in background, redirect output to log
perl -I src openkore.pl --plugins=plugins --control="$PROFILE_DIR/control" >> "$LOG_FILE" 2>&1 &
PID=$!

echo "$PID" > ".pids/$NAME.pid"

echo "Bot $NAME started (PID $PID), logging to $LOG_FILE"
