#!/usr/bin/env python3
"""
rAthena Knowledge Ingestion Pipeline
=====================================
Parses rAthena YAML database files and populates the AI sidecar's
ExperienceDatabase / knowledge base with game server knowledge.

This gives the AI a complete game encyclopedia so it can make
informed strategic decisions without hardcoded config.

Usage:
    python rathena_ingest.py [--rathena-path ./data/rAthena] [--sidecar-url http://127.0.0.1:18081]
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("rathena_ingest")

# Try to import yaml parser
try:
    import yaml
except ImportError:
    logger.error("PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


def parse_yaml_file(path: Path) -> list[dict] | dict | None:
    """Parse a YAML file safely."""
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        logger.warning("Failed to parse %s: %s", path, e)
        return None


def extract_monster_knowledge(mob_db: list[dict]) -> list[dict]:
    """Extract monster knowledge entries from mob_db.yml."""
    entries = []
    body = mob_db
    # The YAML has a Header section then Body section
    if isinstance(mob_db, dict):
        body = mob_db.get("Body", mob_db)

    if not isinstance(body, list):
        logger.warning("mob_db body is not a list, got %s", type(body))
        return entries

    for mob in body:
        if not isinstance(mob, dict):
            continue
        name = mob.get("AegisName") or mob.get("Name") or ""
        if not name:
            continue

        level = mob.get("Level", 1) or 1
        hp = mob.get("Hp", 1) or 1
        base_exp = mob.get("BaseExp", 0) or 0
        job_exp = mob.get("JobExp", 0) or 0
        race = mob.get("Race", "Formless")
        element = mob.get("Element", "Neutral")
        element_level = mob.get("ElementLevel", 1) or 1
        size = mob.get("Size", "Small")
        attack = mob.get("Attack", 0) or 0
        defense = mob.get("Defense", 0) or 0

        # Extract drops
        drops = []
        for drop in (mob.get("Drops") or []):
            if isinstance(drop, dict):
                drops.append({
                    "item": str(drop.get("Item", "")),
                    "rate": drop.get("Rate", 0) or 0,
                })

        entry = {
            "type": "monster",
            "name": name,
            "display_name": mob.get("Name", name),
            "level": level,
            "hp": hp,
            "base_exp": base_exp,
            "job_exp": job_exp,
            "race": race,
            "element": element,
            "element_level": element_level,
            "size": size,
            "attack": attack,
            "defense": defense,
            "drops": drops[:10],  # Top 10 drops
            "mvp_drops": len(mob.get("MvpDrops") or []) > 0,
        }
        entries.append(entry)
    return entries


def extract_map_knowledge(map_drops: list[dict] | dict) -> dict[str, dict]:
    """Extract map knowledge — which monsters spawn where."""
    maps = {}
    body = map_drops
    if isinstance(map_drops, dict):
        body = map_drops.get("Body", map_drops)
    if not isinstance(body, list):
        return maps

    for entry in body:
        if not isinstance(entry, dict):
            continue
        map_name = entry.get("Map", "")
        if not map_name:
            continue
        maps[map_name] = {
            "global_drops": entry.get("GlobalDrops") or [],
            "specific_drops": entry.get("SpecificDrops") or [],
        }
    return maps


def extract_job_exp_knowledge(job_exp: list[dict] | dict) -> dict[str, list]:
    """Extract job EXP tables — XP needed per level per job."""
    jobs = {}
    body = job_exp
    if isinstance(job_exp, dict):
        body = job_exp.get("Body", job_exp)
    if not isinstance(body, list):
        return jobs

    for entry in body:
        if not isinstance(entry, dict):
            continue
        job = entry.get("Class", "")
        if not job:
            continue
        exp_table = []
        for level_entry in (entry.get("Levels") or []):
            if isinstance(level_entry, dict):
                exp_table.append({
                    "level": level_entry.get("Level", 1),
                    "base_exp": level_entry.get("BaseExp", 0),
                    "job_exp": level_entry.get("JobExp", 0),
                })
        jobs[job] = exp_table
    return jobs


def extract_level_penalty(penalty: list[dict] | dict) -> list[dict]:
    """Extract level penalty ranges — what level difference is optimal."""
    body = penalty
    if isinstance(penalty, dict):
        body = penalty.get("Body", penalty)
    if not isinstance(body, list):
        return []
    return body


def build_grind_recommendations(
    monsters: list[dict],
    maps: dict[str, dict],
) -> list[dict]:
    """Build hunting ground recommendations by level range."""
    recs = []

    # Group monsters by level range
    by_level: dict[str, list[dict]] = {}
    for mob in monsters:
        level = mob["level"]
        if level <= 10:
            key = "1-10"
        elif level <= 20:
            key = "11-20"
        elif level <= 35:
            key = "21-35"
        elif level <= 50:
            key = "36-50"
        elif level <= 70:
            key = "51-70"
        elif level <= 90:
            key = "71-90"
        else:
            key = "91+"
        by_level.setdefault(key, []).append(mob)

    for level_range, mobs in sorted(by_level.items()):
        # Best monsters: high base_exp, low HP, low defense
        scored = sorted(
            mobs,
            key=lambda m: (m["base_exp"] / max(m["hp"], 1)) if m["hp"] > 0 else 0,
            reverse=True,
        )
        top = scored[:10]
        recs.append({
            "level_range": level_range,
            "recommended_monsters": [
                {
                    "name": m["name"],
                    "display_name": m["display_name"],
                    "level": m["level"],
                    "base_exp": m["base_exp"],
                    "job_exp": m["job_exp"],
                    "hp": m["hp"],
                    "exp_per_hp": round(m["base_exp"] / max(m["hp"], 1), 4) if m["hp"] > 0 else 0,
                    "element": m["element"],
                    "race": m["race"],
                    "drops": m["drops"],
                }
                for m in top
            ],
        })
    return recs


def send_to_sidecar(url: str, knowledge: dict) -> bool:
    """Send knowledge to the AI sidecar via API."""
    try:
        payload = json.dumps(knowledge).encode("utf-8")
        req = urllib.request.Request(
            f"{url}/v2/ingest/knowledge",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            logger.info("Sidecar response: %s", result.get("status", "unknown"))
            return True
    except urllib.error.HTTPError as e:
        logger.warning("Sidecar HTTP error: %s (endpoint may not exist yet)", e.code)
        return False
    except Exception as e:
        logger.warning("Sidecar error: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Ingest rAthena knowledge into AI sidecar")
    parser.add_argument("--rathena-path", default=str(Path(__file__).resolve().parent.parent / "data" / "rAthena"),
                        help="Path to rAthena git clone")
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:18081",
                        help="AI sidecar base URL")
    parser.add_argument("--output", default="",
                        help="Output JSON file path (instead of sending to sidecar)")
    args = parser.parse_args()

    rathena = Path(args.rathena_path)
    db_re = rathena / "db" / "re"

    logger.info("Loading rAthena databases from %s...", db_re)

    # Load monster database
    mob_data = parse_yaml_file(db_re / "mob_db.yml")
    monsters = extract_monster_knowledge(mob_data or [])
    logger.info("Loaded %d monsters", len(monsters))

    # Load map drops
    map_data = parse_yaml_file(db_re / "map_drops.yml")
    maps = extract_map_knowledge(map_data or {})
    logger.info("Loaded %d maps with drops", len(maps))

    # Load job EXP tables
    job_exp_data = parse_yaml_file(db_re / "job_exp.yml")
    job_exp = extract_job_exp_knowledge(job_exp_data or {})
    logger.info("Loaded %d job EXP tables", len(job_exp))

    # Load level penalty
    penalty_data = parse_yaml_file(db_re / "level_penalty.yml")
    penalty = extract_level_penalty(penalty_data or {})
    logger.info("Loaded level penalty table")

    # Build grind recommendations
    recommendations = build_grind_recommendations(monsters, maps)
    logger.info("Built %d level range recommendations", len(recommendations))

    # Build the knowledge package
    knowledge = {
        "version": 1,
        "ingested_at": time.time(),
        "source": "rathena",
        "stats": {
            "monsters": len(monsters),
            "maps": len(maps),
            "job_exp_tables": len(job_exp),
            "level_ranges": len(recommendations),
        },
        "monsters": monsters[:100],  # Top 100 for LLM context
        "grind_recommendations": recommendations,
        "level_penalty": penalty,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(knowledge, f, indent=2)
        logger.info("Saved knowledge to %s (%d KB)", args.output, os.path.getsize(args.output) // 1024)
    else:
        logger.info("Sending knowledge to sidecar at %s...", args.sidecar_url)
        ok = send_to_sidecar(args.sidecar_url, knowledge)
        if ok:
            logger.info("Knowledge ingested successfully!")
        else:
            logger.info("Sidecar endpoint not available. Saved to knowledge.json instead.")
            with open("knowledge.json", "w") as f:
                json.dump(knowledge, f, indent=2)
            logger.info("Saved to knowledge.json (%d KB)", len(json.dumps(knowledge)) // 1024)

    # Print summary
    print()
    print("=" * 60)
    print("rAthena Knowledge Ingestion Summary")
    print("=" * 60)
    print(f"  Monsters loaded:   {len(monsters)}")
    print(f"  Maps with drops:   {len(maps)}")
    print(f"  Job EXP tables:    {len(job_exp)}")
    print(f"  Level ranges:      {len(recommendations)}")
    print()
    print("  Grind Recommendations:")
    for rec in recommendations[:5]:
        print(f"    Level {rec['level_range']}:")
        for m in rec["recommended_monsters"][:3]:
            print(f"      - {m['display_name']} (Lv{m['level']}, "
                  f"EXP: {m['base_exp']}, HP: {m['hp']}, "
                  f"EXP/HP: {m['exp_per_hp']})")
    print()


if __name__ == "__main__":
    main()