#!/bin/bash
# Phase 2 — 1-hour monitoring script (4 checkpoints at 0, 15, 30, 45, 60 min)
# Logs to monitor_report.txt

MONITOR_DIR="/home/lot399/openkore-ai-v3"
REPORT="$MONITOR_DIR/logs/monitor_report.txt"

echo "PHASE 2 — 1-HOUR MONITORING REPORT" > "$REPORT"
echo "Started: $(date)" >> "$REPORT"
echo "========================================" >> "$REPORT"

do_checkpoint() {
  local label="$1"
  local timestamp=$(date)
  
  {
    echo ""
    echo "--- CHECKPOINT $label ($timestamp) ---"
    echo ""
    
    echo "== Process Status =="
    ps aux | grep -E "openkore|perl.*kicap" | grep -v grep | awk '{print $2, $11, $12, $13}' 2>/dev/null
    echo ""
    
    echo "== CPU/Memory per bot =="
    for b in kicapmasin kicapmasin2 kicapmasin3; do
      pid=$(ps aux | grep "openkore.pl.*$b" | grep -v grep | awk '{print $2}')
      if [ -n "$pid" ]; then
        ps -p $pid -o pid,%cpu,%mem,rss,vsz,etime --no-headers 2>/dev/null
      else
        echo "$b: DEAD/CRASHED"
      fi
    done
    echo ""
    
    echo "== Bot console logs (last 15 lines each) =="
    for b in kicapmasin kicapmasin2 kicapmasin3; do
      echo "--- $b ---"
      f="$MONITOR_DIR/logs/$b.log"
      if [ -f "$f" ]; then
        tail -15 "$f"
        echo ""
        echo "Unique errors in $b:"
        grep -oi 'error\|warning\|uninitialized\|not found\|emergency\|exception\|crash\|fatal' "$f" 2>/dev/null | sort | uniq -c | sort -rn || echo "(none)"
      fi
      echo ""
    done
    
    echo "== Sidecar health counters =="
    curl -s http://127.0.0.1:18081/health/ready 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
c=d.get('counters',{})
print(f'Bots registered: {d.get(\"bots_registered\")}')
print(f'Snapshots ingested: {c.get(\"snapshots_ingested\",0)}')
print(f'Actions acknowledged: {c.get(\"actions_acknowledged\",0)}')
print(f'Reflex triggers total: {c.get(\"reflex_triggers_total\",0)}')
print(f'Reflex actions emitted: {c.get(\"reflex_actions_emitted\",0)}')
print(f'LLM calls: {c.get(\"llm_calls_total\",0)}')
print(f'Planner runs: {c.get(\"planner_runs_total\",0)}')
print(f'Runtime mode: {d.get(\"runtime_mode\")}')
print(f'PDCA running: {d.get(\"pdca_running\")}')
print(f'Startup gate open: {d.get(\"startup_gate_open\")}')
print(f'Autonomy policy: {json.dumps(d.get(\"autonomy_policy\",{}), indent=1)}')
" 2>/dev/null
    
    echo ""
    echo "== Sidecar log tail =="
    tail -10 "$MONITOR_DIR/logs/sidecar.log" 2>/dev/null
    
    echo ""
    echo "== Log sizes =="
    ls -lh "$MONITOR_DIR/logs/"*.log 2>/dev/null | awk '{print $5, $NF}'
    
    echo ""
    echo "========================================"
  } >> "$REPORT"
}

# Checkpoint 0 (T+0 — already running)
do_checkpoint "0 — T+0 (initial)"

# Checkpoint 1 (T+15)
sleep 900
do_checkpoint "1 — T+15"

# Checkpoint 2 (T+30)
sleep 900
do_checkpoint "2 — T+30"

# Checkpoint 3 (T+45)
sleep 900
do_checkpoint "3 — T+45"

# Checkpoint 4 (T+60 — final)
sleep 900
do_checkpoint "4 — T+60 (FINAL)"

# Final summary
{
  echo ""
  echo "========================================"
  echo "FINAL SUMMARY"
  echo "========================================"
  echo "Monitor completed at: $(date)"
  
  echo ""
  echo "== Final error counts =="
  for b in kicapmasin kicapmasin2 kicapmasin3; do
    f="$MONITOR_DIR/logs/$b.log"
    [ -f "$f" ] && echo "$b: $(wc -l < "$f") lines, $(grep -ci 'error\|warning\|uninitialized' "$f" 2>/dev/null || echo 0) issues"
  done
  echo "sidecar: $(wc -l < "$MONITOR_DIR/logs/sidecar.log") lines, $(grep -ci 'error\|exception\|traceback\|warning' "$MONITOR_DIR/logs/sidecar.log" 2>/dev/null || echo 0) issues"
  
  echo ""
  echo "== Final sidecar counters =="
  curl -s http://127.0.0.1:18081/health/ready 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
c=d.get('counters',{})
for k,v in sorted(c.items()):
    print(f'{k}: {v}')
print(f'Runtime mode: {d.get(\"runtime_mode\")}')
print(f'PDCA running: {d.get(\"pdca_running\")}')
" 2>/dev/null
  
  echo ""
  echo "== All bot processes =="
  ps aux | grep -E "openkore|perl.*kicap" | grep -v grep
} >> "$REPORT"

echo "Monitor complete. Report at $REPORT" >> "$REPORT"
echo "REPORT GENERATED"
