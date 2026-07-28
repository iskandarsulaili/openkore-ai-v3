#!/bin/bash
cd "$(dirname "$0")"
rm -f /tmp/sidecar.log
set -a
source .env 2>/dev/null
source AI_sidecar/.env 2>/dev/null
set +a
export OPENKORE_AI_COST_MODE=max
export OPENKORE_AI_LLM_MAX_CALLS_PER_HOUR=300
python3 -m uvicorn ai_sidecar.app:app --host 127.0.0.1 --port 18081 --app-dir AI_sidecar > /tmp/sidecar.log 2>&1 &
echo "Sidecar PID: $!"
