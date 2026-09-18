#!/usr/bin/env python3
"""Verify EVERY row of tables/job_change_locations.txt against the LIVE server.

RENEWAL-AWARE: this server loads npc/re/scripts_main.conf (Renewal), which
imports npc/scripts_*.conf and npc/re/scripts_*.conf. Any coordinate that
matches only a file OUTSIDE that import chain (e.g. npc/pre-re/**) is a DEAD
coordinate even though an NPC textually exists there — the earlier revision of
this audit was fooled by exactly that (Crusader matched npc/pre-re/jobs/2-2/
crusader.txt while the loaded chain uses prt_cas 251 75).

Reports per row: OK | NO_NPC | DEAD_FILE (unloaded) | UNREACHABLE.
"""
import os, re, struct

ROOT = "/home/lot399/rathena-AI-world"
TABLE = "/home/lot399/openkore-ai-v3/tables/job_change_locations.txt"
FIELDS = "/home/lot399/openkore-ai-v3/fields"

SPAWN = re.compile(r"^([a-z_0-9@]+),(\d+),(\d+)[,\t]", re.I)


def loaded_npc_files():
    """Follow the conf import chain from npc/re/scripts_main.conf and collect
    every npc: <path> that is actually loaded."""
    loaded, seen = set(), set()
    stack = [os.path.join(ROOT, "npc/re/scripts_main.conf")]
    while stack:
        conf = stack.pop()
        if conf in seen or not os.path.exists(conf):
            continue
        seen.add(conf)
        base = os.path.dirname(conf)
        # conf files reference paths relative to the server root
        for line in open(conf, encoding="latin-1", errors="ignore"):
            s = line.strip()
            if s.startswith("//") or not s:
                continue
            m = re.match(r"^import:\s*(.+?)\s*$", s)
            if m:
                tgt = m.group(1)
                cand = tgt if os.path.isabs(tgt) else os.path.join(ROOT, tgt)
                if os.path.exists(cand):
                    stack.append(cand)
                else:
                    cand2 = os.path.join(base, os.path.basename(tgt))
                    if os.path.exists(cand2):
                        stack.append(cand2)
                continue
            m = re.match(r"^npc:\s*(.+?)\s*$", s)
            if m:
                p = m.group(1)
                cand = p if os.path.isabs(p) else os.path.join(ROOT, p)
                if os.path.exists(cand):
                    loaded.add(os.path.realpath(cand))
    return loaded


LOADED = loaded_npc_files()
print(f"loaded npc files (renewal chain): {len(LOADED)}")


def is_loaded(path):
    return os.path.realpath(path) in LOADED


# index (map,x,y) -> [files] restricted to loaded files
spawns = {}
for path in LOADED:
    if not path.endswith(".txt") or ".bak" in path:
        continue
    try:
        with open(path, encoding="latin-1", errors="ignore") as fh:
            for line in fh:
                if line.startswith("//"):
                    continue
                m = SPAWN.match(line)
                if m:
                    spawns.setdefault((m.group(1).lower(), int(m.group(2)), int(m.group(3))), []).append(path)
    except OSError:
        pass
print(f"loaded spawn coordinates indexed: {len(spawns)}\n")


def reachable(m, x, y):
    p = os.path.join(FIELDS, f"{m}.dist")
    if not os.path.exists(p):
        return None
    raw = open(p, "rb").read()
    w, h = struct.unpack("<HH", raw[4:8])
    d = raw[8:]
    if not (0 <= x < w and 0 <= y < h):
        return False
    return d[y * w + x] != 0xFF


rows = [l for l in open(TABLE, encoding="utf-8").read().splitlines()
        if l.strip() and not l.strip().startswith("#")]
print(f"{'job':14s} {'map':12s} {'x':>4s} {'y':>4s}  {'reach':6s} status")
bad = []
for r in rows:
    parts = [p.strip() for p in r.split("|")]
    if len(parts) < 3:
        continue
    job, m = parts[0], parts[1].lower()
    xy = parts[2].split()
    if len(xy) < 2:
        continue
    try:
        x, y = int(xy[0]), int(xy[1])
    except ValueError:
        continue
    hit = spawns.get((m, x, y))
    reach = reachable(m, x, y)
    rtxt = {True: "YES", False: "NO", None: "NODATA"}[reach]
    if hit:
        status = "OK"
    elif reach is False:
        status = "UNREACHABLE"
        bad.append((job, m, x, y, status))
    else:
        status = "NO_NPC_AT_COORD"
        bad.append((job, m, x, y, status))
    print(f"{job:14s} {m:12s} {x:4d} {y:4d}  {rtxt:6s} {status}")

print()
if bad:
    print("rows needing correction:")
    for b in bad:
        print("  ", b)
else:
    print("all rows verified against the live renewal chain")
