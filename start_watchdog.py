#!/usr/bin/env python3
"""Launcher for the watchdog supervisor daemon."""
import sys
import os

# Ensure proper working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'AI_sidecar')

from ai_sidecar.runtime.watchdog import run_daemon
run_daemon()
