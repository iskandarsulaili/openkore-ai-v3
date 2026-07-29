#!/bin/bash
cd /home/lot399/openkore-ai-v3
exec nohup python3 -c "import sys; sys.path.insert(0, 'AI_sidecar'); from ai_sidecar.runtime.watchdog import run_daemon; run_daemon()" > /dev/null 2>&1 &
echo $! > /home/lot399/openkore-ai-v3/.watchdog_pid
