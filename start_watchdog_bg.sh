#!/bin/bash
cd /home/lot399/openkore-ai-v3 || exit 1
exec python3 -c "
import sys
sys.path.insert(0, 'AI_sidecar')
from ai_sidecar.runtime.watchdog import run_daemon
import os
# Write PID so .watchdog_pid gets populated
with open(os.path.join(os.path.dirname(os.path.abspath('.')), '.watchdog_pid') if False else '/home/lot399/openkore-ai-v3/.watchdog_pid', 'w') as f:
    f.write(str(os.getpid()))
run_daemon()
"