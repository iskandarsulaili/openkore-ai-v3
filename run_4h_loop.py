#!/usr/bin/env python3
"""
4-hour Pro RO Player test loop — v2
Every 10 min: checkpoint, fix issues, restart stuck bots, iterate.
Hardcoded-check: scans for hardcoded map/item/NPC names.
"""
import subprocess, time, re, json, sys, os
from pathlib import Path

BASE = Path("/home/lot399/openkore-ai-v3")
LOG = BASE / "logs"
OUT = LOG / "4h_loop.log"
ALIVE = Path("/dev/shm/ro_4h_alive")
BOTS = ["kicapmasin", "kicapmasin2", "kicapmasin3"]
SIDECAR_URL = "http://127.0.0.1:18081"

ALIVE.write_text("1")
ITERATIONS = 24  # 10 min * 24 = 4 hours

# Known hardcoded patterns to scan for
HARDCODED_PATTERNS = [
    (r'prt_fild0[0-9]', 'hardcoded map name'),
    (r'prontera', 'hardcoded map name'),
    (r'moran|aldebaran|geffen|payon|izlude', 'hardcoded map name'),
    (r'"Red Potion"|"White Potion"|"Fly Wing"', 'hardcoded item name'),
    (r'npc_steps\s*=>\s*\[', 'hardcoded NPC steps'),
    (r'Tool Dealer|Kafra|Portal Girl', 'hardcoded NPC name'),
    (r'lockMap\s+prt_', 'hardcoded lockMap in config'),
]

def log(msg):
    ts = time.strftime("%H:%M:%S UTC", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line)
    with open(OUT, "a") as f:
        f.write(line + "\n")

def run(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=BASE)
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1
    except Exception as e:
        return "", str(e), -1

def sidecar_health():
    out, err, rc = run(f"curl -sf {SIDECAR_URL}/v1/health/live")
    if rc != 0:
        return False
    out2, _, rc2 = run(f"curl -sf {SIDECAR_URL}/v1/health/ready")
    if rc2 != 0:
        return False
    try:
        d = json.loads(out2)
        return d.get("bots_registered", 0)
    except:
        return 0

def bot_log_summary(name):
    f = LOG / f"{name}.log"
    if not f.exists():
        return "NO_LOG", "NO_LOG", ""
    # Kills in LAST 50 lines (recent activity)
    out, _, _ = run(f"tail -50 {f} | grep -c 'Target Monster .* died'")
    recent_kills = int(out.strip() or "0")
    # Total kills ever
    out2, _, _ = run(f"grep -c 'Target Monster .* died' {f}")
    total_kills = out2.strip() or "0"
    # Recent exp
    out3, _, _ = run(f"tail -50 {f} | grep -c 'You have gained [1-9]'")
    recent_exp = int(out3.strip() or "0")
    # Recent errors (last 50 lines)
    out4, _, _ = run(f"tail -50 {f} | grep -c 'reconnect\\|Error\\|server still recognizes'")
    recent_errs = int(out4.strip() or "0")
    # Total errors
    out5, _, _ = run(f"grep -c 'server still recognizes' {f}")
    total_reconn = out5.strip() or "0"
    # Last map line
    out6, _, _ = run(f"grep 'Map Change' {f} | tail -1")
    map_info = out6.strip()[:50] if out6.strip() else "?"
    # Last log line with timestamp
    out7, _, _ = run(f"tail -1 {f}")
    last_line = out7.strip()[:60] if out7.strip() else "?"

    summary = f"recent_kills={recent_kills} total_kills={total_kills} recent_exp={recent_exp} recent_errs={recent_errs} map={map_info}"
    status = "ACTIVE" if recent_kills > 0 or recent_exp > 0 else ("CONNECTED" if recent_errs == 0 else "STUCK")
    if recent_errs > 3:
        status = "RECONNECT_LOOP"
    total_r = int(total_reconn or "0")
    return f"status={status} {summary} total_reconn={total_r}", status, last_line

def find_bot_pids():
    """Find PIDs by scanning /proc for perl processes with openkore.pl"""
    pids = {}
    try:
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                cmdline = (p / "cmdline").read_text(errors="replace").replace("\0", " ")
            except:
                continue
            if "openkore.pl" in cmdline:
                for b in BOTS:
                    if b in cmdline:
                        pids[b] = p.name
    except:
        pass
    return pids

def restart_bot(name):
    log(f"RESTARTING {name}")
    run(f"pkill -f 'openkore\.pl.*{name}' 2>/dev/null")
    time.sleep(3)
    cmd = f"cd {BASE} && source .env 2>/dev/null && perl -I src openkore.pl --plugins=plugins --control=\".bot_profiles/{name}/control\" >> logs/{name}.log 2>&1 &"
    subprocess.Popen(cmd, shell=True, cwd=BASE,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"{name} restart issued (PID ?, check next cycle)")
    time.sleep(3)

def restart_sidecar():
    log("RESTARTING sidecar")
    run("pkill -f 'uvicorn.*ai_sidecar' 2>/dev/null")
    time.sleep(3)
    cmd = f"cd {BASE}/AI_sidecar && source venv/bin/activate && python3 -m uvicorn ai_sidecar.app:app --host 127.0.0.1 --port 18081 > {BASE}/logs/sidecar.log 2>&1 &"
    subprocess.Popen(cmd, shell=True, cwd=BASE,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log("sidecar restart issued")
    time.sleep(5)

def scan_hardcoded():
    """Scan critical files for hardcoded values"""
    findings = []
    files_to_check = [
        BASE / "AI_sidecar" / "ai_sidecar" / "autonomy" / "pdca_loop.py",
        BASE / "AI_sidecar" / "ai_sidecar" / "crewai" / "agents" / "pro_ro_player_agent.py",
        BASE / "AI_sidecar" / "ai_sidecar" / "crewai" / "tasks.py",
        BASE / "AI_sidecar" / "ai_sidecar" / "api" / "routers" / "discovery.py",
    ]
    for pattern, desc in HARDCODED_PATTERNS:
        for f in files_to_check:
            if f.exists():
                try:
                    content = f.read_text()
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for m in matches:
                        line_num = content[:m.start()].count("\n") + 1
                        findings.append(f"{f.name}:{line_num} - {desc}: '{m.group()}'")
                except:
                    pass
    return findings

# ===== MAIN LOOP =====
log("=" * 60)
log("4-HOUR PRO RO PLAYER TEST LOOP v2 STARTED")
log(f"Iterations: {ITERATIONS} x 10min = 4 hours")
log(f"Bots: {', '.join(BOTS)}")
log(f"Sidecar: {SIDECAR_URL}")
log("=" * 60)

for i in range(1, ITERATIONS + 1):
    if not ALIVE.exists() or ALIVE.read_text().strip() != "1":
        log(f"STOP signal received at iteration {i}/{ITERATIONS}")
        break

    elapsed = i * 10
    log(f"\n--- CHECKPOINT {i}/{ITERATIONS} (T+{elapsed}min) ---")

    # 1. Sidecar health
    registered = sidecar_health()
    if registered is False:
        log("SIDECAR DOWN — restarting")
        restart_sidecar()
        time.sleep(5)
        registered = sidecar_health()
        if registered is False:
            log("SIDECAR STILL DOWN — will retry next cycle")
    else:
        log(f"Sidecar: OK, bots_registered={registered}")

    # 2. Char progress report
    out, _, _ = run(f"python3 logs/char_progress.py 2>/dev/null")
    if out.strip():
        for line in out.strip().split("\n")[:8]:
            log(f"  {line.strip()}")
    else:
        log("  char_progress: no output")

    # 3. Per-bot check via /proc
    pids = find_bot_pids()
    log(f"  Found PIDs via /proc: {pids}")
    for bot in BOTS:
        summary, status, last = bot_log_summary(bot)
        if bot in pids:
            log(f"  {bot}: RUNNING (PID {pids[bot]}) — {summary}")
        else:
            log(f"  {bot}: NOT RUNNING — {summary}")
            log(f"    last: {last}")
            restart_bot(bot)
            continue

        if status == "RECONNECT_LOOP":
            log(f"  {bot}: RECONNECT LOOP — killing and restarting")
            restart_bot(bot)

    # 4. Hardcoded value scan (every 2 checkpoints = 20min)
    if i % 2 == 0:
        hardcoded = scan_hardcoded()
        if hardcoded:
            log(f"  HARDCODED VALUES FOUND ({len(hardcoded)}):")
            for h in hardcoded[:10]:
                log(f"    {h}")
        else:
            log(f"  Hardcoded scan: clean")

    # 5. Sidecar errors
    out, _, _ = run(f"grep -c 'ERROR\\|CRITICAL\\|Traceback' logs/sidecar.log 2>/dev/null")
    sidecar_errs = int(out.strip() or "0")
    if sidecar_errs > 0:
        out2, _, _ = run(f"grep 'ERROR\\|CRITICAL\\|Traceback' logs/sidecar.log 2>/dev/null | tail -3")
        log(f"  Sidecar errors ({sidecar_errs}): {out2.strip()[:200]}")

    # 6. LLM provider status
    out, _, _ = run(f"grep -c 'provider_fail\\|ProvFail' logs/sidecar.log 2>/dev/null")
    prov_fails = int(out.strip() or "0")
    if prov_fails > 0:
        log(f"  LLM provider failures in log: {prov_fails}")
        out2, _, _ = run(f"grep 'provider_fail\\|ProvFail' logs/sidecar.log 2>/dev/null | tail -2")
        log(f"    last: {out2.strip()[:150]}")

    log(f"  --- checkpoint {i} complete ---")

    # Wait for next cycle
    if i < ITERATIONS:
        log(f"  Next checkpoint in 10 min...")
        for _ in range(60):
            time.sleep(10)
            if not ALIVE.exists() or ALIVE.read_text().strip() != "1":
                log("STOP signal received during wait")
                break

log("=" * 60)
log("4-HOUR TEST LOOP COMPLETE")
log(f"Completed {ITERATIONS} checkpoints over 4 hours")
log("=" * 60)
