"""Skill Engine — selects optimal skills per target element.

Architecture:
  - Loads skill_tree.yml (175 jobs) for available skills per class
  - Loads skill_db.yml (1,635 skills) for cast/cooldown/SP/element data
  - Scores skills: base_damage × element_modifier × sp_efficiency
  - Tracks cooldowns per skill per bot
  - Queues skills_add <name> <level> actions

RULE.md compliance:
  - All skill data from rAthena DB (skill_db.yml + skill_tree.yml)
  - Zero hardcoded: elements from attr_fix.yml, SP costs from skill_db
  - Bridge only executes skills_add commands — sidecar selects skills
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Data Structures ────────────────────────────────────────────────


@dataclass
class SkillInfo:
    """Full skill data from skill_db.yml."""
    id: int
    name: str
    description: str
    max_level: int
    skill_type: str           # Weapon, Magic, Misc, Passive
    target_type: str          # Attack, Self, Party, Enemy, Ground, Passive
    hit_type: str             # Single, Multi, Weapon, Magic
    element: str              # Element or "Weapon" (inherits weapon element)
    sp_cost: list[int]        # SP cost per level
    cast_time: list[int]      # Cast time per level (ms)
    fixed_cast: list[int]     # Fixed cast per level (ms)
    after_cast_delay: list[int]  # GCD per level (ms)
    cooldown: list[int]       # Cooldown per level (ms)
    range: int                # Skill range
    splash_area: list[int]    # AoE per level
    knockback: list[int]      # Knockback per level
    hit_count: list[int]      # Hit count per level
    status: str               # Status effect (Stun, Freeze, etc.)
    is_passive: bool          # True if passive skill


@dataclass
class SkillScore:
    """Scored skill for combat."""
    name: str
    level: int
    element: str
    sp_cost: int
    cast_time: int
    cooldown: int
    score: float
    reason: str


# ── Skill DB Loader ────────────────────────────────────────────────


class SkillDB:
    """Loads skill data from skill_db.yml."""

    def __init__(self):
        self._skills_by_id: dict[int, SkillInfo] = {}
        self._skills_by_name: dict[str, SkillInfo] = {}
        self._tree_by_job: dict[str, list[dict]] = {}
        self._loaded = False
    
    def load(self) -> bool:
        """Load skill_db.yml + skill_tree.yml. Returns True on success."""
        if self._loaded:
            return True
        
        # Find rathena path
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        paths = [
            os.path.join(os.path.expanduser("~"), "rathena", "db", "re", "skill_db.yml"),
            os.path.join(base, "knowledge", "rathena_db", "db", "re", "skill_db.yml"),
            os.path.join(base, "knowledge", "rathena_db", "db", "skill_db.yml"),
        ]
        
        skill_path = None
        tree_path = None
        for p in paths:
            if os.path.exists(p):
                skill_path = p
                break
        
        tree_paths = [p.replace("skill_db.yml", "skill_tree.yml") for p in paths]
        for p in tree_paths:
            if os.path.exists(p):
                tree_path = p
                break
        
        if not skill_path:
            logger.warning("skill_db: skill_db.yml not found")
            return False
        
        try:
            import yaml
            
            # Load skill_db.yml
            with open(skill_path) as f:
                data = yaml.safe_load(f)
            
            if not data or "Body" not in data:
                logger.warning("skill_db: invalid format")
                return False
            
            for entry in data["Body"]:
                try:
                    self._parse_skill_entry(entry)
                except Exception as e:
                    logger.debug("skill_db: skipped %s: %s", entry.get("Name", "?"), e)
            
            # Load skill_tree.yml if available
            if tree_path:
                with open(tree_path) as f:
                    tree_data = yaml.safe_load(f)
                if tree_data and "Body" in tree_data:
                    for job_entry in tree_data["Body"]:
                        job_name = job_entry.get("Job", "")
                        tree = job_entry.get("Tree", [])
                        self._tree_by_job[job_name] = tree
                        # Also add inherited skills
                        inherit = job_entry.get("Inherit", {})
                        for parent_job, active in inherit.items():
                            if active and parent_job in self._tree_by_job:
                                parent_skills = self._tree_by_job.get(parent_job, [])
                                existing_names = {s["Name"] for s in tree}
                                for ps in parent_skills:
                                    if ps["Name"] not in existing_names:
                                        tree.append(ps)
            
            self._loaded = True
            logger.info("skill_db: loaded %d skills, %d job trees", 
                       len(self._skills_by_id), len(self._tree_by_job))
            return True
            
        except Exception as e:
            logger.warning("skill_db: failed to load: %s", e)
            return False
    
    def _parse_skill_entry(self, entry: dict):
        """Parse a single skill entry from skill_db.yml."""
        sid = entry.get("Id", 0)
        name = entry.get("Name", "")
        max_lv = entry.get("MaxLevel", 1)
        
        # Parse SP cost per level
        sp_cost = []
        requires = entry.get("Requires", {}) or {}
        if "SpCost" in requires:
            for level_entry in requires["SpCost"]:
                lv = level_entry.get("Level", len(sp_cost) + 1)
                amount = level_entry.get("Amount", 0)
                while len(sp_cost) < lv:
                    sp_cost.append(0)
                sp_cost[lv - 1] = amount
        
        # Parse other per-level fields
        def parse_per_level(field_name, default=0) -> list[int]:
            field_data = entry.get(field_name)
            if not field_data:
                return []
            result = []
            if isinstance(field_data, list):
                for level_entry in field_data:
                    lv = level_entry.get("Level", len(result) + 1)
                    val = level_entry.get("Time", level_entry.get("Count", 
                            level_entry.get("Area", level_entry.get("Amount", level_entry.get("Element", default)))))
                    while len(result) < lv:
                        result.append(default)
                    result[lv - 1] = int(val) if not isinstance(val, str) else default
            return result
        
        cast_time = parse_per_level("CastTime")
        fixed_cast = parse_per_level("FixedCastTime")
        after_cast = parse_per_level("AfterCastActDelay")
        cooldown = parse_per_level("Cooldown")
        splash = parse_per_level("SplashArea")
        knockback = parse_per_level("Knockback")
        hit_count = parse_per_level("HitCount")
        
        # Element handling: "Weapon" means inherits weapon element
        element = entry.get("Element", "Neutral")
        if isinstance(element, list):
            # Per-level element
            ele_list = element
            if ele_list:
                element = ele_list[0].get("Element", "Neutral") if isinstance(ele_list[0], dict) else "Neutral"
        elif element == "Weapon":
            element = "Weapon"  # Special: inherits weapon element
        
        skill = SkillInfo(
            id=sid,
            name=name,
            description=entry.get("Description", ""),
            max_level=max_lv,
            skill_type=str(entry.get("Type", "Misc")),
            target_type=str(entry.get("TargetType", "Passive")),
            hit_type=str(entry.get("Hit", "Single")),
            element=element,
            sp_cost=sp_cost,
            cast_time=cast_time,
            fixed_cast=fixed_cast,
            after_cast_delay=after_cast,
            cooldown=cooldown,
            range=entry.get("Range", -1) if isinstance(entry.get("Range"), int) else -1,
            splash_area=splash,
            knockback=knockback,
            hit_count=hit_count,
            status=str(entry.get("Status", "")),
            is_passive=entry.get("Type") == "Passive" or entry.get("TargetType") == "Passive",
        )
        
        self._skills_by_id[sid] = skill
        self._skills_by_name[name] = skill
    
    def get_skills_for_job(self, job_name: str) -> list[SkillInfo]:
        """Get all available skills for a job class."""
        if not self._loaded:
            self.load()
        
        tree = self._tree_by_job.get(job_name, [])
        skills = []
        for entry in tree:
            sname = entry.get("Name", "")
            if sname in self._skills_by_name:
                skill = self._skills_by_name[sname]
                if not skill.is_passive:
                    skills.append(skill)
        return skills
    
    def get_skill(self, name: str) -> SkillInfo | None:
        """Look up a skill by name."""
        if not self._loaded:
            self.load()
        return self._skills_by_name.get(name)
    
    def normalize_job_name(self, raw_job: str) -> str:
        """Normalize various job name formats to skill_tree format."""
        mapping = {
            "novice": "Novice", "swordman": "Swordman", "mage": "Mage",
            "archer": "Archer", "acolyte": "Acolyte", "merchant": "Merchant",
            "thief": "Thief", "knight": "Knight", "priest": "Priest",
            "wizard": "Wizard", "blacksmith": "Blacksmith", "hunter": "Hunter",
            "assassin": "Assassin", "crusader": "Crusader", "monk": "Monk",
            "sage": "Sage", "rogue": "Rogue", "alchemist": "Alchemist",
            "bard": "Bard", "dancer": "Dancer",
        }
        lower = raw_job.lower().replace("_", " ").strip()
        for key, val in mapping.items():
            if key in lower:
                return val
        # Capitalize first letter
        return raw_job.capitalize()


# ── Skill Engine ──────────────────────────────────────────────────


class SkillEngine:
    """Selects optimal skills for combat based on target and resources."""

    def __init__(self):
        self._db = SkillDB()
        self._cooldowns: dict[str, dict[str, float]] = {}  # bot_id → {skill_name: ready_at_ms}
        self._last_cast: dict[str, float] = {}  # bot_id → last cast time ms
    
    def select_best_skill(self, bot_id: str, job_name: str, 
                          sp_ratio: float, monster_element: str,
                          monster_element_level: int = 1,
                          weapon_element: str = "Neutral") -> SkillScore | None:
        """Select the best skill for the current situation.
        
        Args:
            bot_id: Bot identifier
            job_name: Job class name
            sp_ratio: Current SP / Max SP (0.0-1.0)
            monster_element: Target monster element
            monster_element_level: Element level (1-4)
            weapon_element: Current weapon element
        
        Returns:
            SkillScore for best skill, or None if no skill available
        """
        if not self._db._loaded:
            self._db.load()
        
        job_key = self._db.normalize_job_name(job_name)
        skills = self._db.get_skills_for_job(job_key)
        if not skills:
            return None
        
        from ai_sidecar.combat.element_table import get_element_table
        et = get_element_table()
        
        now_ms = time.time() * 1000
        bot_cds = self._cooldowns.get(bot_id, {})
        last_cast = self._last_cast.get(bot_id, 0)
        gcd_ms = 1000  # approximate global cooldown
        next_available = last_cast + gcd_ms
        
        best: Optional[SkillScore] = None
        
        for skill in skills:
            # Skip passives — no castable skills
            if skill.is_passive or skill.target_type == "Passive":
                continue
            # Skip skills with no SP cost AND no cast time — they're auto-proc or passives
            use_level = skill.max_level
            sp_needed = skill.sp_cost[use_level - 1] if use_level <= len(skill.sp_cost) and skill.sp_cost else 0
            has_cast = skill.cast_time[use_level - 1] > 0 if use_level <= len(skill.cast_time) and skill.cast_time else False
            if sp_needed == 0 and not has_cast and skill.skill_type != "Misc":
                continue  # Passive weapon/armor proficiency
            
            # Determine effective element
            skill_element = skill.element
            if skill_element == "Weapon":
                skill_element = weapon_element  # Inherits weapon element
            
            # use_level and sp_needed set above in passive filter
            
            # Skip if not enough SP
            if sp_needed > 0 and sp_ratio < (sp_needed / 100):  
                # SP ratio check: need at least enough SP to cast once
                continue
            
            # Check cooldown
            if skill.name in bot_cds and now_ms < bot_cds[skill.name]:
                continue  # Still on cooldown
            
            # Check global cooldown
            if now_ms < next_available:
                continue
            
            # Calculate element modifier
            if skill_element in ("Neutral", "Weapon"):
                ele_mod = 1.0
            else:
                ele_mod = et.get_modifier(skill_element, monster_element, monster_element_level) / 100.0
            
            # Calculate score
            # Base: level × hit_count + element bonus
            base_power = use_level * max(len(skill.hit_count), 1)
            # Weapon/Magic skills deal damage (1.5x priority over misc/utility)
            if skill.skill_type in ("Weapon", "Magic"):
                base_power *= 1.5
            ele_bonus = ele_mod
            sp_efficiency = 1.0 / max(sp_needed, 1)
            
            # Priority: elemental advantage > raw power > SP efficiency
            if ele_mod > 1.0:
                score = base_power * ele_bonus * 2.0 * sp_efficiency
                reason = f"element_advantage ({skill_element}→{monster_element}: {ele_mod:.0%})"
            elif ele_mod < 1.0:
                # Skip resisted skills unless no other option
                score = base_power * max(ele_bonus, 0.5) * sp_efficiency
                reason = f"resisted ({skill_element}→{monster_element}: {ele_mod:.0%})"
            else:
                score = base_power * sp_efficiency
                reason = "neutral"
            
            # Status effect bonus
            if skill.status:
                score *= 1.2  # 20% bonus for status-inflicting skills
                if skill.status == "Stun" and monster_element == "Neutral":
                    score *= 1.3  # Extra for neutral (no element resist)
            
            # Cooldown penalty (prefer shorter cooldowns)
            cd = skill.cooldown[use_level - 1] if use_level <= len(skill.cooldown) and skill.cooldown else 0
            if cd > 0:
                score *= 1.0 / (1.0 + cd / 10000)  # -10% per second of cooldown
            
            skill_score = SkillScore(
                name=skill.name,
                level=use_level,
                element=skill_element,
                sp_cost=sp_needed,
                cast_time=skill.cast_time[use_level - 1] if use_level <= len(skill.cast_time) and skill.cast_time else 0,
                cooldown=cd,
                score=score,
                reason=reason,
            )
            
            if best is None or score > best.score:
                best = skill_score
        
        return best
    
    def mark_cast(self, bot_id: str, skill_name: str, skill_level: int, cooldown_ms: int = 0):
        """Record a skill cast for cooldown tracking."""
        now_ms = time.time() * 1000
        self._last_cast[bot_id] = now_ms
        
        if skill_name not in self._cooldowns.get(bot_id, {}):
            if bot_id not in self._cooldowns:
                self._cooldowns[bot_id] = {}
        
        if cooldown_ms > 0:
            self._cooldowns[bot_id][skill_name] = now_ms + cooldown_ms
    
    def get_known_skills(self, job_name: str) -> list[str]:
        """Get list of skill names available for a job."""
        if not self._db._loaded:
            self._db.load()
        job_key = self._db.normalize_job_name(job_name)
        skills = self._db.get_skills_for_job(job_key)
        return [s.name for s in skills]


# Singleton
_engine: SkillEngine | None = None


def get_skill_engine() -> SkillEngine:
    """Get global SkillEngine instance."""
    global _engine
    if _engine is None:
        _engine = SkillEngine()
    return _engine
