#!/bin/bash
# Phase 2 — 1-hour monitoring script
# 5 checkpoints: T+0, T+15, T+30, T+45, T+60
LOG_DIR="$(cd "$(dirname "$0")/logs" && pwd)"
REPORT="$LOG_DIR/phase2_report.txt"

log_cp() {
  local label="$1"
  local ts=$(date '+%Y-%m-%d %H:%M:%S')
  {
    echo ""
    echo "========================================"
    echo "CHECKPOINT $label — $ts"
    echo "========================================"
    echo ""
    
    echo "== BOT PROCESSES =="
    ps aux | grep -E "perl.*kicap" | grep -v grep | awk '{print $2, $11, $12}' | head -5
    echo "Total: $(ps aux | grep -E 'perl.*kicap' | grep -v grep | wc -l)"
    echo ""
    
    echo "== BOT CPU/MEMORY =="
    for b in kicapmasin kicapmasin2 kicapmasin3; do
      pid=$(ps aux | grep "openkore.pl.*$b" | grep -v grep | awk '{print $2}')
      if [ -n "$pid" ]; then
        ps -p $pid -o pid,%cpu,%mem,rss,etime --no-headers 2>/dev/null || echo "$b: no ps data"
      else
        echo "$b: NOT RUNNING"
      fi
    done
    echo ""
    
    echo "== SIDECAR COUNTERS =="
    curl -s http://127.0.0.1:18081/health/ready 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
c=d.get('counters',{})
print(f'Mode: {d.get(\"runtime_mode\")}')
print(f'PDCA: {d.get(\"pdca_running\")}')
print(f'Bots registered: {d.get(\"bots_registered\")}')
print(f'Reflex T/E: {c.get(\"reflex_triggers_total\",0)}/{c.get(\"reflex_actions_emitted\",0)}')
print(f'Snapshots: {c.get(\"snapshots_ingested\",0)}')
print(f'Actions queued/ack: {c.get(\"actions_queued\",0)}/{c.get(\"actions_acknowledged\",0)}')
" 2>/dev/null || echo "Sidecar not reachable"
    echo ""
    
    echo "== SIDECAR ISSUES =="
    echo "reflex_emit_failed: $(grep -c 'reflex_emit_chain_all_targets_failed' $LOG_DIR/sidecar.log 2>/dev/null || echo 0)"
    echo "validation_failed: $(grep -c 'http_request_validation_failed' $LOG_DIR/sidecar.log 2>/dev/null || echo 0)"
    echo "latency_exceeded: $(grep -c 'latency_budget_exceeded' $LOG_DIR/sidecar.log 2>/dev/null || echo 0)"
    echo ""
    
    echo "== LLM CALLS =="
    echo "200 OK: $(grep -c '200 OK.*chat' $LOG_DIR/sidecar.log 2>/dev/null || echo 0)"
    echo "502/errors: $(grep -c '502\|llm_request_failed\|structured_parse' $LOG_DIR/sidecar.log 2>/dev/null || echo 0)"
    echo "Planner runs: $(grep -c 'provider_route_attempt' $LOG_DIR/sidecar.log 2>/dev/null || echo 0)"
    echo ""
    
    echo "== BOT LOG SUMMARY =="
    for b in kicapmasin kicapmasin2 kicapmasin3; do
      f="$LOG_DIR/$b.log"
      if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        uninit=$(grep -c 'uninitialized' "$f" 2>/dev/null || echo 0)
        emrg=$(grep -c 'emergency' "$f" 2>/dev/null || echo 0)
        ai_toggle=$(grep -c 'AI set to' "$f" 2>/dev/null || echo 0)
        echo "$b: ${lines}L uninit=$uninit emrg=$emrg ai_toggle=$ai_toggle"
        echo "  Last 3: $(tail -3 "$f" | tr '\n' '; ')"
        echo ""
      fi
    done
    
    echo "== LOG SIZES =="
    ls -lh $LOG_DIR/*.log 2>/dev/null | awk '{print $5, $NF}' | head -5
    
  } >> "$REPORT"
  echo "Checkpoint $label logged"
}

# T+0
log_cp "0 — INITIAL"

# T+15
sleep 900
log_cp "1 — T+15min"

# T+30
sleep 900
log_cp "2 — T+30min"

# T+45
sleep 900
log_cp "3 — T+45min"

# T+60
sleep 900
log_cp "4 — T+60min (FINAL)"

echo "PHASE 2 COMPLETE" >> "$REPORT"
echo "MONITOR DONE"
