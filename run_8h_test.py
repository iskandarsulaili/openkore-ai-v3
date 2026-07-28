#!/usr/bin/env python3
import subprocess, time, re, json, sys
from pathlib import Path

LOG = Path(__file__).resolve().parent / "logs"
RESULTS = []
OUT = LOG / "8h_checkpoints.log"

def checkpoint(msg):
    ts = time.strftime("%H:%M:%S UTC", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line)
    RESULTS.append(line)

checkpoint("8-hour test started")
checkpoint("Bots: kicapmasin, kicapmasin2, kicapmasin3")
checkpoint("Note: Dual-login prevents all 3 on same account — 1 active, 2 reconnect on timeout")
checkpoint("")

LOOP = Path("/dev/shm/ro_8h_alive")  # alive marker
LOOP.write_text("1")

for i in range(1, 49):
    for _ in range(60):
        time.sleep(10)
        if not LOOP.exists() or LOOP.read_text().strip() != "1":
            checkpoint(f"Stopped at cycle {i}/48")
            _write_results()
            sys.exit(0)
    
    checkpoint(f"=== T+{i*10}min (cycle {i}/48) ===")
    
    # Sidecar health
    r = subprocess.run(["curl", "-sf", "http://127.0.0.1:18081/health/live"],
                       capture_output=True, text=True, timeout=10)
    checkpoint(f"Sidecar: {'LIVE' if r.returncode == 0 else 'DOWN'}")
    
    # Bot logs
    for bot in ["kicapmasin", "kicapmasin2", "kicapmasin3"]:
        lf = LOG / f"{bot}.log"
        if not lf.exists():
            checkpoint(f"  {bot}: no log")
            continue
        b = lf.read_text(errors="replace")
        kills = len(re.findall(r'Target Monster \S+ \(\d+\) died', b))
        exp = sum(1 for m in re.finditer(r'You have gained (\d+)/', b) if int(m.group(1)) > 0)
        items = b.count("Item added to inventory")
        maps = len([l for l in b.split("\n") if "Map Change:" in l])
        
        # Current location
        map_lines = [l for l in b.split("\n") if "Map Change:" in l]
        loc = map_lines[-1].strip()[:60] if map_lines else "none"
        
        checkpoint(f"  {bot}: K:{kills} Exp:{exp} Items:{items} Maps:{maps} @ {loc}")

OUT.write_text("\n".join(RESULTS))
print(f"\nResults written to {OUT}")
