#!/bin/bash
cd "$(dirname "$0")"
set -a
source .env
source AI_sidecar/.env
set +a
python3 -m uvicorn ai_sidecar.app:app --host 127.0.0.1 --port 18081 --app-dir AI_sidecar &
echo $! > /tmp/sidecar.pid
echo "Sidecar starting with PID $(cat /tmp/sidecar.pid)"
