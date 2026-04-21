#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/stop-ai-openkore.sh"
sleep 1
"$SCRIPT_DIR/start-ai-openkore.sh"

