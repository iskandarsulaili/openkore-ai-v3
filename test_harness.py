#!/usr/bin/env python3
"""OpenKore AI Test Harness — offline verification."""
import os, sys, re, ast
os.chdir("/home/lot399/openkore-ai-v3")

PASS, FAIL = 0, 0
def test(name, ok, detail=""):
    global PASS, FAIL
    if ok: PASS += 1; print(f"  PASS: {name}")
    else: FAIL += 1; print(f"  FAIL: {name}  {'-- ' + detail if detail else ''}")

with open("AI_sidecar/ai_sidecar/autonomy/heuristic_service.py") as f:
    hs = f.read()
with open("plugins/aiSidecarBridge/aiSidecarBridge.pl") as f:
    br = f.read()
with open("RULE.md") as f:
    rules = f.read()

# Config audit
print("--- Config Audit ---")
for k, v in {'sitAuto_hp_lower':'20','sitAuto_hp_upper':'50','attackAuto':'3',
             'teleportAuto_hp':'10','sellAuto_maxWeight':'70'}.items():
    test(f"Config audit sets {k}={v}", f'\"{k}\", \"{v}\"' in hs or f"\"{k}\", \"{v}\"" in hs or f"\"{k}\",\'{v}\'" in hs)
test("No teleportAuto_hp=0 in hunting", 'teleportAuto_hp", "0", "hunting"' not in hs)
test("No aiSidecar_sitAutoHp=0", 'aiSidecar_sitAutoHp", "0"' not in hs)

# Bridge reflexes
print("\n--- Bridge Reflexes ---")
em = br[br.find("EMERGENCY"):]
test("Emergency: no sitting check", '$char->{sitting}' not in em[:200])
test("Emergency: AI::dequeue", 'AI::dequeue' in em[:1000])
test("Emergency: no config overrides", '$::config{' not in em[:1000])
test("Emergency: walks to Kafra (290,224)", '290 224' in em[:1000])

fs = br[br.find("force_stand]")-100:]
active_fs = [l for l in fs.split('\n') if '$::config{' in l and not l.strip().startswith('#')]
test("Force stand: no config overrides", len(active_fs) == 0)

# Bridge config violations
violations = []
for m in re.finditer(r'\$::config\{([^}]*)\}\s*=', br):
    ls = br.rfind('\n', 0, m.start())+1
    lt = br[ls:br.find('\n', m.start())].strip()
    if lt.startswith('#'): continue
    ctx = br[max(0,m.start()-100):m.end()+100]
    if any(x in ctx for x in ['_sidecar_set', '$set_val', '$orig_key', 'lockMap', 'route_randomWalk']): continue
    violations.append(m.group(1))
test(f"Zero bridge config violations", len(violations)==0, f"{len(violations)} left: {violations[:5]}")

# Attack block
print("\n--- Attack Block ---")
ab = br[br.find("ATTACK BLOCK"):br.find("ATTACK BLOCK")+800] if "ATTACK BLOCK" in br else ""
test("Attack block exists", bool(ab))
test("Checks attack_auto", 'attack_auto' in ab)
test("Blocks when <=0 (Thief Bug=-1)", '<= 0' in ab)

# Pipeline flexibility
print("\n--- Pipeline Flexibility ---")
test("Step 1: level>=10 skip", 'base_level >= 10' in hs and 'skip' in hs[hs.find('base_level >= 10'):hs.find('base_level >= 10')+100])
test("Step 0: already on map skip", 'already on prt_fild05' in hs)
test("Step 3: level-skip potions", '_skip_potions' in hs)

# RULE.md
print("\n--- RULE.md Compliance ---")
test("Section 1: Bridge is LIMITED", "Bridge is LIMITED" in rules)
test("Section 2: lockMap Consistency", "lockMap consistency" in rules.lower())
test("Section 3: Config audit authority", "Heuristic Config Audit" in rules)
test("Section 10: Single Routing Authority", "Single Routing Authority" in rules)

print(f"\n{'='*60}")
print(f"Results: {PASS} PASS, {FAIL} FAIL, {PASS+FAIL} TOTAL")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
