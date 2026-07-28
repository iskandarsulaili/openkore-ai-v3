#!/bin/bash
LOGFILE="$(cd "$(dirname "$0")/logs" && pwd)/8h_checkpoints.log"
echo "8-hour endurance test started: $(date -u)" > "$LOGFILE"
echo "Bots: kicapmasin, kicapmasin2, kicapmasin3" >> "$LOGFILE"
echo "Sidecar: http://127.0.0.1:18081" >> "$LOGFILE"
echo "Intervals: 10 minutes x 48 cycles" >> "$LOGFILE"
echo "" >> "$LOGFILE"

cd "$(dirname "$0")"

for i in $(seq 1 48); do
    sleep 600
    echo "=== T+$((i*10))min ===" >> "$LOGFILE"
    echo "Checkpoint $i/48 at $(date -u)" >> "$LOGFILE"
    python3 logs/char_progress.py >> "$LOGFILE" 2>&1
    echo "--- raw ---" >> "$LOGFILE"
    for f in logs/kicapmasin.log logs/kicapmasin2.log logs/kicapmasin3.log; do
        if [ -f "$f" ]; then
            k=$(grep -c 'Target Monster .* died' "$f")
            e=$(grep -c 'You have gained [1-9]' "$f")
            echo "$f: K:$k Exp:$e" >> "$LOGFILE"
        fi
    done
    echo "" >> "$LOGFILE"
done

echo "=== FINAL ===" >> "$LOGFILE"
echo "8-hour test complete at $(date -u)" >> "$LOGFILE"
echo "Sidecar status:" >> "$LOGFILE"
curl -sf http://127.0.0.1:18081/health/live && echo " LIVE" || echo " DOWN" >> "$LOGFILE"
echo "Bot status:" >> "$LOGFILE"
for f in logs/kicapmasin.log logs/kicapmasin2.log logs/kicapmasin3.log; do
    if [ -f "$f" ]; then
        k=$(grep -c 'Target Monster .* died' "$f")
        e=$(grep -c 'You have gained [1-9]' "$f")
        l=$(grep -c 'reached level\|level up\|You have increased' "$f")
        echo "$f: K:$k Exp:$e Lvl_ups:$l" >> "$LOGFILE"
    fi
done
cat "$LOGFILE"
