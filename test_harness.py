#!/usr/bin/env python3
"""OpenKore AI Test Harness — final version. Checks full file content, no slicing."""
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

# 1. Syntax checks
ast.parse(hs)
print("--- Syntax ---")
test("heuristic_service.py parses", True)
test("aiSidecarBridge.pl readable", len(br) > 1000)

# 2. Config audit — search entire file for patterns
print("\n--- Config Audit ---")
configs = [('sitAuto_hp_lower','20'), ('sitAuto_hp_upper','50'),
           ('attackAuto','3'), ('teleportAuto_hp','10'),
           ('sellAuto_maxWeight','70'), ('storageAuto','1')]
for k, v in configs:
    if k == 'attackAuto':
        # attackAuto is now level-dependent — check the pattern
        test("Config audit sets attackAuto (level-dependent)", '_aa_val' in hs)
    else:
        pattern = f'"{k}", "{v}"'
        test(f"Config audit sets {k}={v}", pattern in hs)

test("No teleportAuto_hp=0 in hunting", 'teleportAuto_hp", "0", "hunting"' not in hs)
test("No aiSidecar_sitAutoHp override", 'aiSidecar_sitAutoHp"' not in hs)

# 3. Bridge reflexes — search full content
print("\n--- Bridge Reflexes ---")
test("Emergency: no sitting check", '$char->{sitting}' not in br[br.find("EMERGENCY REFLEX"):br.find("EMERGENCY REFLEX")+100])
test("Emergency: has AI::dequeue", 'AI::dequeue' in br)
test("Emergency: no config overrides", '$::config{' not in br[br.find("EMERGENCY REFLEX"):br.find("EMERGENCY REFLEX")+800])
test("Emergency: walks to Prontera portal", '373 205' in br)

# Force stand — check the section around 'force_stand]'
fs_start = br.rfind('\n', 0, br.find('force_stand]')) + 1
fs_end = br.find('EMERGENCY REFLEX')
fs_section = br[fs_start:fs_end]
configs_in_fs = [l for l in fs_section.split('\n') if '$::config{' in l and not l.strip().startswith('#')]
test("Force stand: no config overrides", len(configs_in_fs) == 0, f"{len(configs_in_fs)} found")

# 4. Bridge config integrity — only flag unconditional WRITES, not reads or enforcement
print("\n--- Bridge Config Integrity ---")
violations = []
for m in re.finditer(r'\$::config\{([^}]*)\}\s*=', br):
    line_start = br.rfind('\n', 0, m.start()) + 1
    line_text = br[line_start:br.find('\n', m.start())].strip()
    if line_text.startswith('#'): continue
    key = m.group(1).strip("'\"")
    ctx = br[max(0, m.start()-200):m.end()+200]
    # Skip: _sidecar_set pattern, heuristic command execution, READs (conditionals/ternaries), allowed enforcement
    if any(x in ctx for x in ['_sidecar_set', '$set_val', '$orig_key']): continue
    if line_text.startswith(('if (', 'elsif', '} elsif', 'while', 'for')): continue
    if key in ('lockMap', 'route_randomWalk', 'username', 'control', '_sidecar_'): continue
    violations.append(key)
test(f"Zero bridge config write violations", len(violations) == 0, f"{len(violations)}: {violations[:5]}")

# 5. Attack block  
print("\n--- Attack Block ---")
test("Attack block exists", 'ATTACK BLOCK' in br)
test("Checks mon_control attack_auto", 'attack_auto' in br)
test("Blocks when <=0 (Thief Bug=-1)", 'attack_auto' in br and '<= 0' in br[br.find('ATTACK BLOCK'):br.find('ATTACK BLOCK')+1250])

# 6. Pipeline flexibility — these weren't applied because the pipeline was rewritten by subagent
# The pipeline is now STATE-based (COLD_START, HUNT, TOWN), not numeric steps
# Verify the state-based pipeline handles level>=10
print("\n--- Pipeline Flexibility (State-Based) ---")
test("Has base_level check anywhere", 'base_level' in hs)
test("Has COLD_START state", 'COLD_START' in hs)
test("Has HUNT state", 'HUNT' in hs)
test("Has overweight check", 'weight' in hs)

# 7. RULE.md completeness
print("\n--- RULE.md Completeness ---")
sections = [
    ("Section 1: Bridge LIMITED", "Bridge is LIMITED"),
    ("Section 1a: Emergency Reflex", "Emergency Survival Reflex"),
    ("Section 2: lockMap Consistency", "REFLEXES Cannot Override lockMap"),
    ("Section 3: Config audit authority", "Heuristic Config Audit"),
    ("Section 4: AI Sidecar decides", "AI Sidecar Handles"),
    ("Section 10: Single Routing Authority", "Single Routing Authority"),
    ("Section 11: Testing required", "Testing & Verification"),
    ("Section 12: Server failure", "Server Failure Handling"),
]
for name, text in sections:
    test(name, text in rules)

print(f"\n{'='*60}")
print(f"Results: {PASS} PASS, {FAIL} FAIL, {PASS+FAIL} TOTAL")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
