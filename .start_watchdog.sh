#!/bin/bash
# Launcher for the openkore-ai-v3 watchdog daemon.
# Fully detaches the daemon (new session, nohup, stdio redirected) so it
# survives the launching session (cron/hermes) and reparents to init.
cd /home/lot399/openkore-ai-v3 || exit 1
setsid nohup python3 -c "import sys; sys.path.insert(0, 'AI_sidecar'); from ai_sidecar.runtime.watchdog import run_daemon; run_daemon()" >> /home/lot399/openkore-ai-v3/.watchdog.log 2>&1 < /dev/null &
echo $! > /home/lot399/openkore-ai-v3/.watchdog_pid
