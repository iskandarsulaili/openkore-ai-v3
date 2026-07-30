#!/bin/bash
# Start the watchdog daemon in background with nohup, detached from this shell
cd /home/lot399/openkore-ai-v3
nohup python3 -c "
import sys
sys.path.insert(0, 'AI_sidecar')
from ai_sidecar.runtime.watchdog import run_daemon
run_daemon()
" > /tmp/watchdog_startup.log 2>&1 &
WPID=$!
echo $WPID > /home/lot399/openkore-ai-v3/.watchdog_pid
echo "Watchdog started with PID $WPID"
