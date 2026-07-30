#!/usr/bin/env python3
"""Launcher for the openkore watchdog daemon."""
import os
import sys
import signal

PID_FILE = "/home/lot399/openkore-ai-v3/.watchdog_pid"

# Write PID immediately before starting the daemon
with open(PID_FILE, "w") as f:
    f.write(str(os.getpid()))

sys.path.insert(0, "/home/lot399/openkore-ai-v3/AI_sidecar")
from ai_sidecar.runtime.watchdog import run_daemon

# Ignore SIGHUP so we survive terminal close
signal.signal(signal.SIGHUP, signal.SIG_IGN)

run_daemon()