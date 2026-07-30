#!/usr/bin/env python3
"""Start the watchdog supervisor as a truly detached daemon process."""
import os
import sys
import time

def daemonize():
    """Double-fork to detach from the parent process."""
    # First fork
    pid = os.fork()
    if pid > 0:
        # Parent exits so the child is orphaned
        os._exit(0)
    # Child continues
    os.setsid()
    os.umask(0)
    # Second fork
    pid = os.fork()
    if pid > 0:
        os._exit(0)
    # Now fully detached from parent

if __name__ == "__main__":
    daemonize()
    
    # Redirect stdin/stdout/stderr to /dev/null
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    
    # Change to project root
    os.chdir("/home/lot399/openkore-ai-v3")
    sys.path.insert(0, "AI_sidecar")
    
    # Write PID file
    pid = os.getpid()
    with open(".watchdog_pid", "w") as f:
        f.write(str(pid))
    
    # Import and run
    from ai_sidecar.runtime.watchdog import run_daemon
    run_daemon()
