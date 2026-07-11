#!/usr/bin/env python3
"""
rAthena Knowledge Extractor
===========================
Extracts all game data from rAthena YAML/TXT files and saves as structured JSON
in the openkore-ai-v3 repo. This makes the AI self-contained — no runtime
dependency on the rAthena clone.

Run once after cloning rAthena, or re-run when rAthena updates.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("extract")

try:
    import yaml
except ImportError:
    logger.error("PyYAML required. Run: pip install pyyaml")
    sys.exit(1)


def extract_mob_db(path: Path) -> list[dict]:
    """Extract monster database — 120K lines of monster stats, drops, EXP."""
    data = _parse_yaml(path)
    body = data.get("Body", data) if isinstance(data, dict) else data
    if not isinstance(body, list):
        logger.warning("mob_db body is not a list")
        return []

    monsters = []
    for mob in body:
        if not isinstance(mob, dict):
            continue
        name = mob.get("AegisName") or mob.get("Name") or ""
        if not name:
            continue

        entry = {
            "id": mob.get("Id", 0),
            "name": name,
            "display_name": mob.get("Name", name),
            "level": mob.get("Level", 1) or 1,
            "hp": mob.get("Hp", 1) or 1,
            "sp": mob.get("Sp", 0) or 0,
            "base_exp": mob.get("BaseExp", 0) or 0,
            "job_exp": mob.get("JobExp", 0) or 0,
            "mvp_exp": mob.get("MvpExp", 0) or 0,
            "attack": mob.get("Attack", 0) or 0,
            "attack2": mob.get("Attack2", 0) or 0,
            "defense": mob.get("Defense", 0) or 0,
            "magic_defense": mob.get("MagicDefense", 0) or 0,
            "str": mob.get("Str", 1) or 1,
            "agi": mob.get("Agi", 1) or 1,
            "vit": mob.get("Vit", 1) or 1,
            "int": mob.get("Int", 1) or 1,
            "dex": mob.get("Dex", 1) or 1,
            "luk": mob.get("Luk", 1) or 1,
            "attack_range": mob.get("AttackRange", 0) or 0,
            "size": mob.get("Size", "Medium"),
            "race": mob.get("Race", "Formless"),
            "element": mob.get("Element", "Neutral"),
            "element_level": mob.get("ElementLevel", 1) or 1,
            "walk_speed": mob.get("WalkSpeed", 200) or 200,
            "mode": dict(mob.get("Modes", {}) or {}),
            "drops": [
                {
                    "item": d.get("Item", ""),
                    "rate": d.get("Rate", 0) or 0,
                    "steal_protected": bool(d.get("StealProtected", False)),
                }
                for d in (mob.get("Drops") or []) if isinstance(d, dict)
            ][:10],  # Top 10 drops
            "mvp_drops": [
                {
                    "item": d.get("Item", ""),
                    "rate": d.get("Rate", 0) or 0,
                }
                for d in (mob.get("MvpDrops") or []) if isinstance(d, dict)
            ],
            "is_mvp": bool(mob.get("MvpDrops")),
        }
        monsters.append(entry)

    logger.info("Extracted %d monsters", len(monsters))
    return monsters


def extract_mob_skills(path: Path) -> dict[str, list[dict]]:
    """Extract monster skills — 15K lines of skill definitions."""
    if not path.exists():
        logger.warning("mob_skill_db.txt not found at %s", path)
        return {}

    mob_skills: dict[str, list[dict]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue

            mob_id = parts[0].strip()
            skill_name = parts[1].strip()
            state = parts[2].strip()
            skill_lv = parts[3].strip()
            rate = int(parts[4].strip() or 0)
            cast_time = int(parts[5].strip() or 0)
            delay = int(parts[6].strip() or 0)
            cancelable = parts[7].strip()
            target = parts[8].strip() if len(parts) > 8 else ""
            condition = parts[10].strip() if len(parts) > 10 else ""
            condition_val = parts[11].strip() if len(parts) > 11 else ""

            if mob_id not in mob_skills:
                mob_skills[mob_id] = []
            mob_skills[mob_id].append({
                "skill": skill_name,
                "level": skill_lv,
                "state": state,
                "rate": rate,
                "cast_time_ms": cast_time,
                "delay_ms": delay,
                "cancelable": cancelable,
                "target": target,
                "condition": condition,
                "condition_value": condition_val,
            })

    logger.info("Extracted skills for %d monster IDs", len(mob_skills))
    return mob_skills


def extract_map_drops(path: Path) -> dict[str, dict]:
    """Extract map-specific drops."""
    data = _parse_yaml(path)
    body = data.get("Body", data) if isinstance(data, dict) else data
    if not isinstance(body, list):
        return {}

    maps = {}
    for entry in body:
        if not isinstance(entry, dict):
            continue
        map_name = entry.get("Map", "")
        if not map_name:
            continue
        maps[map_name] = {
            "global_drops": [
                {"item": d.get("Item", ""), "rate": d.get("Rate", 0)}
                for d in (entry.get("GlobalDrops") or []) if isinstance(d, dict)
            ],
            "specific_drops": [
                {
                    "monster": s.get("Monster", ""),
                    "drops": [
                        {"item": d.get("Item", ""), "rate": d.get("Rate", 0)}
                        for d in (s.get("Drops") or []) if isinstance(d, dict)
                    ],
                }
                for s in (entry.get("SpecificDrops") or []) if isinstance(s, dict)
            ],
        }
    logger.info("Extracted %d maps with drops", len(maps))
    return maps


def extract_level_penalty(path: Path) -> list[dict]:
    """Extract level penalty brackets."""
    data = _parse_yaml(path)
    body = data.get("Body", data) if isinstance(data, dict) else data
    if not isinstance(body, list):
        return []
    logger.info("Extracted %d level penalty brackets", len(body))
    return body


def extract_job_exp(path: Path) -> list[dict]:
    """Extract job EXP tables."""
    data = _parse_yaml(path)
    body = data.get("Body", data) if isinstance(data, dict) else data
    if not isinstance(body, list):
        return []
    logger.info("Extracted %d job EXP tables", len(body))
    return body


def extract_simple_yaml(path: Path, name: str) -> list[dict] | dict | None:
    """Extract a simple YAML file (extracts Body if present)."""
    data = _parse_yaml(path)
    if data is None:
        return None
    body = data.get("Body", data) if isinstance(data, dict) else data
    if isinstance(body, list):
        logger.info("Extracted %s: %d entries", name, len(body))
    elif isinstance(body, dict):
        logger.info("Extracted %s: %d keys", name, len(body))
    return body


def extract_text_file(path: Path) -> list[str] | None:
    """Extract a text file as lines."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            lines = [line.rstrip() for line in f if line.strip() and not line.startswith("//")]
        logger.info("Extracted %s: %d lines", path.name, len(lines))
        return lines
    except Exception as e:
        logger.warning("Failed to extract %s: %s", path.name, e)
        return None


def _parse_yaml(path: Path) -> dict | list | None:
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", path, e)
        return None


def main():
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    output_dir = repo_root / "knowledge"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to find rAthena
    rathena_candidates = [
        Path("/home/lot399/rathena"),
        repo_root.parent / "rathena",
        Path(os.environ.get("RATHENA_PATH", "/home/lot399/rathena")),
    ]
    rathena_path = None
    for p in rathena_candidates:
        if (p / "db" / "re" / "mob_db.yml").exists():
            rathena_path = p
            break

    if not rathena_path:
        logger.error("rAthena not found. Set RATHENA_PATH env var or clone to ~/rathena")
        sys.exit(1)

    db_re = rathena_path / "db" / "re"
    logger.info("Extracting from %s → %s", db_re, output_dir)

    # Extract all databases
    knowledge = {
        "version": 3,
        "extracted_at": time.time(),
        "source": str(rathena_path),
        "stats": {},
        "monsters": extract_mob_db(db_re / "mob_db.yml"),
        "mob_skills": extract_mob_skills(db_re / "mob_skill_db.txt"),
        "map_drops": extract_map_drops(db_re / "map_drops.yml"),
        "level_penalty": extract_level_penalty(db_re / "level_penalty.yml"),
        "job_exp": extract_job_exp(db_re / "job_exp.yml"),
        "refine": extract_simple_yaml(db_re / "refine.yml", "refine"),
        "attr_fix": extract_simple_yaml(db_re / "attr_fix.yml", "attr_fix"),
        "skill_tree": extract_simple_yaml(db_re / "skill_tree.yml", "skill_tree"),
        "guild_skill_tree": extract_simple_yaml(db_re / "guild_skill_tree.yml", "guild_skill_tree"),
        "job_stats": extract_simple_yaml(db_re / "job_stats.yml", "job_stats"),
        "job_aspd": extract_simple_yaml(db_re / "job_aspd.yml", "job_aspd"),
        "job_basepoints": extract_simple_yaml(db_re / "job_basepoints.yml", "job_basepoints"),
        "pet_db": extract_simple_yaml(db_re / "pet_db.yml", "pet_db"),
        "homunculus_db": extract_simple_yaml(db_re / "homunculus_db.yml", "homunculus_db"),
        "mercenary_db": extract_simple_yaml(db_re / "mercenary_db.yml", "mercenary_db"),
        "quest_db": extract_simple_yaml(db_re / "quest_db.yml", "quest_db"),
        "enchantgrade": extract_simple_yaml(db_re / "enchantgrade.yml", "enchantgrade"),
        "elemental_db": extract_simple_yaml(db_re / "elemental_db.yml", "elemental_db"),
        "item_combos": extract_simple_yaml(db_re / "item_combos.yml", "item_combos"),
        "item_enchant": extract_simple_yaml(db_re / "item_enchant.yml", "item_enchant"),
        "item_reform": extract_simple_yaml(db_re / "item_reform.yml", "item_reform"),
        "mob_summon": extract_simple_yaml(db_re / "mob_summon.yml", "mob_summon"),
        "magicmushroom_db": extract_simple_yaml(db_re / "magicmushroom_db.yml", "magicmushroom_db"),
        "produce_db": extract_text_file(db_re / "produce_db.txt"),
        "exp_guild": extract_simple_yaml(db_re / "exp_guild.yml", "exp_guild"),
        "exp_homun": extract_simple_yaml(db_re / "exp_homun.yml", "exp_homun"),
    }

    knowledge["stats"] = {
        "monsters": len(knowledge["monsters"]),
        "mob_skills": len(knowledge["mob_skills"]),
        "maps": len(knowledge["map_drops"]),
        "level_penalty": len(knowledge["level_penalty"]),
        "job_exp_tables": len(knowledge["job_exp"]),
    }

    # Save as JSON
    output_path = output_dir / "knowledge.json"
    with open(output_path, "w") as f:
        json.dump(knowledge, f, indent=2)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Saved knowledge to %s (%.1f MB)", output_path, size_mb)

    # Print summary
    print()
    print("=" * 60)
    print("rAthena Knowledge Extraction Complete")
    print("=" * 60)
    for key, val in knowledge["stats"].items():
        print(f"  {key}: {val}")
    print(f"  total_size: {size_mb:.1f} MB")
    print(f"  saved_to: {output_path}")
    print()


if __name__ == "__main__":
    main()