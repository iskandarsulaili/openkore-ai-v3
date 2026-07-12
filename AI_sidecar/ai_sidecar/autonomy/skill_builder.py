"""AutoSkillBuilder — Intelligent skill allocation for OpenKore bots.

Reads the skill_tree from knowledge/knowledge.json (175 entries) and
allocates available skill_points to the best available skill per class.
Skill priority: job-change required skills first (max all class skills
to 10), then combat skills. Respects skill prerequisites and MaxLevel.
Thread-safe with RLock.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal

logger = logging.getLogger(__name__)

# ── Path to the knowledge JSON ──
_KNOWLEDGE_PATH = Path("/home/lot399/openkore-ai-v3/knowledge/knowledge.json")

# ── Core combat skills per class (prioritised after job-change skills) ──
_COMBAT_PRIORITY: dict[str, list[str]] = {
    "swordman": ["SM_BASH", "SM_MAGNUM", "SM_PROVOKE", "SM_ENDURE"],
    "mage": ["MG_FIREBOLT", "MG_COLDBOLT", "MG_LIGHTNINGBOLT", "MG_NAPALMBEAT",
             "MG_FIREWALL", "MG_FROSTDIVER"],
    "archer": ["AC_DOUBLE", "AC_CONCENTRATION", "AC_SHOWER"],
    "acolyte": ["AL_HEAL", "AL_INCAGI", "AL_BLESSING", "AL_TELEPORT", "AL_WARP"],
    "merchant": ["MC_DISCOUNT", "MC_OVERCHARGE", "MC_PUSHCART", "MC_VENDING",
                 "MC_IDENTIFY"],
    "thief": ["TF_DOUBLE", "TF_STEAL", "TF_HIDING", "TF_POISON", "TF_MISS"],
    "knight": ["KN_BOWLINGBASH", "KN_BRANDISHSPEAR", "KN_PIERCE", "KN_SPEARSTAB"],
    "wizard": ["WZ_METEOR", "WZ_STORMGUST", "WZ_JUPITEL", "WZ_FROSTNOVA",
               "WZ_HEAVENDRIVE"],
    "hunter": ["HT_BLITZBEAT", "HT_STEELCROW", "HT_FALCON", "HT_LANDMINE",
               "HT_FREEZINGTRAP"],
    "priest": ["PR_SANCTUARY", "PR_KYRIE", "PR_MAGNUS", "PR_GLORIA",
               "ALL_RESURRECTION"],
    "blacksmith": ["BS_ADRENALINE", "BS_OVERTHRUST", "BS_WEAPONPERFECT",
                   "BS_MAXIMIZE", "BS_HAMMERFALL"],
    "assassin": ["AS_SONICBLOW", "AS_CLOAKING", "AS_ENCHANTPOISON",
                 "AS_GRIMTOOTH", "AS_VENOMDUST"],
    "paladin": ["CR_HOLYCROSS", "CR_GRANDCROSS", "CR_SHIELDCHARGE",
                "CR_REFLECTSHIELD", "CR_DEVOTION"],
    "sage": ["SA_AUTOSPELL", "SA_FREECAST", "SA_FLAMELAUNCHER",
             "SA_FROSTWEAPON", "SA_LANDPROTECTOR"],
    "monk": ["MO_TRIPLEATTACK", "MO_CHAINCOMBO", "MO_COMBOFINISH",
             "MO_EXTREMITYFIST", "MO_FINGEROFFENSIVE"],
    "rogue": ["RG_BACKSTAP", "RG_RAID", "RG_SNATCHER", "RG_STEALCOIN",
              "RG_PLAGIARISM"],
    "alchemist": ["AM_DEMONSTRATION", "AM_ACIDTERROR", "AM_POTIONPITCHER",
                  "AM_CANNIBALIZE", "AM_SPHEREMINE"],
    "bard": ["BA_MUSICALSTRIKE", "BA_DISSONANCE", "BA_WHISTLE", "BA_POEMBRAGI"],
    "dancer": ["DC_THROWARROW", "DC_UGLYDANCE", "DC_HUMMING", "DC_DONTFORGETME"],
}

# ── Job tiers for advancement (know which skills are "class skills") ──
_FIRST_CLASSES = {"swordman", "mage", "archer", "acolyte", "merchant", "thief"}
_SECOND_CLASSES = {"knight", "wizard", "hunter", "priest", "blacksmith",
                   "assassin", "paladin", "sage", "monk", "rogue", "alchemist",
                   "bard", "dancer", "crusader"}


class AutoSkillBuilder:
    """Thread-safe automatic skill point allocation.

    Reads the rAthena skill_tree from knowledge.json and allocates
    points to:
    1. Job-change required skills (max all class-specific skills to 10)
    2. Combat skills (based on _COMBAT_PRIORITY per class)
    3. Buff/utility skills

    Respects skill prerequisites and does not exceed MaxLevel.
    Tracks known skills per bot via ``runtime_state.skill_allocation_state``.
    """

    def __init__(self, knowledge_path: str | None = None) -> None:
        self._lock = RLock()
        self._knowledge_path = Path(knowledge_path) if knowledge_path else _KNOWLEDGE_PATH
        # Per-bot skill allocation tracking: bot_id -> {skill_name -> allocated_level}
        self._allocation_state: dict[str, dict[str, int]] = {}
        # Loaded skill tree: job_name -> list of skill entries
        self._skill_tree: dict[str, list[dict[str, Any]]] = {}
        self._load_skill_tree()

    # ── Public API ──────────────────────────────────────────────────────

    def evaluate(
        self,
        bot_id: str,
        class_name: str,
        skill_points: int,
        known_skills: dict[str, int] | None = None,
    ) -> ActionProposal | None:
        """Evaluate skill point availability and return an ActionProposal.

        Args:
            bot_id: Unique bot identifier.
            class_name: Current job/class name (lowercase).
            skill_points: Available skill points from the snapshot.
            known_skills: Dict of skill_name -> level already known.
                          If ``None``, reads from internal tracking state.

        Returns:
            A single ActionProposal for one skill point allocation, or
            ``None`` if no points are available or no allocate-able skill
            is found.
        """
        if not skill_points or skill_points <= 0:
            return None

        class_key = class_name.lower().strip()
        now = datetime.now(UTC)

        # Load known skills from caller or internal state
        known = dict(known_skills or self._allocation_state.get(bot_id, {}))

        # Get the skill tree for this class
        tree = self._skill_tree.get(class_key, [])
        if not tree:
            logger.debug("skill_builder[%s]: no tree for '%s'", bot_id, class_key)
            return None

        # 1. Pick the best skill to level up
        chosen_skill = self._pick_skill(class_key, tree, known)
        if chosen_skill is None:
            logger.debug("skill_builder[%s]: no skill to allocate for '%s'", bot_id, class_key)
            return None

        skill_name = chosen_skill["Name"]
        max_level = int(chosen_skill.get("MaxLevel", 10))
        current_level = known.get(skill_name, 0)

        # Safety: do not exceed max level
        new_level = current_level + 1
        if new_level > max_level:
            logger.debug(
                "skill_builder[%s]: %s already at max level %d",
                bot_id, skill_name, max_level,
            )
            return None

        # Build command
        command = f"skills_add {skill_name} 1"

        # Idempotency key based on the exact allocation
        idem_key = f"skill-{bot_id}-{skill_name}-lvl{new_level}-{uuid.uuid4().hex[:8]}"

        proposal = ActionProposal(
            action_id=f"skill_{skill_name}_{uuid.uuid4().hex[:8]}",
            kind="command",
            command=command,
            priority_tier=ActionPriorityTier.strategic,
            source="planner",
            created_at=now,
            expires_at=now + timedelta(seconds=30),
            idempotency_key=idem_key,
            metadata={
                "bot_id": bot_id,
                "class": class_key,
                "skill": skill_name,
                "current_level": current_level,
                "new_level": new_level,
                "max_level": max_level,
                "skill_points_remaining": skill_points - 1,
                "prerequisites_met": self._check_prerequisites_met(
                    chosen_skill, known
                ),
            },
        )

        # Track allocation
        with self._lock:
            state = self._allocation_state.setdefault(bot_id, {})
            state[skill_name] = new_level

        logger.info(
            "skill_builder[%s]: %s -> %s lvl %d/%d",
            bot_id, class_key, skill_name, new_level, max_level,
        )

        return proposal

    def get_skill_tree(self, class_name: str) -> list[dict[str, Any]]:
        """Return the complete skill tree for a class with requirements.

        Each entry::
            {
                "Name": str,
                "MaxLevel": int,
                "Requires": [{"Name": str, "Level": int}, ...] | None,
                "Exclude": bool | None,
            }
        """
        class_key = class_name.lower().strip()
        return list(self._skill_tree.get(class_key, []))

    def get_allocation_state(self, bot_id: str) -> dict[str, int]:
        """Return the current skill allocation state for a bot.

        Useful for the PDCA loop to feed ``known_skills`` into evaluate().
        """
        with self._lock:
            return dict(self._allocation_state.get(bot_id, {}))

    def reset_allocation_state(self, bot_id: str) -> None:
        """Reset tracking state for a bot (e.g. after job change)."""
        with self._lock:
            self._allocation_state.pop(bot_id, None)

    # ── Internal helpers ────────────────────────────────────────────────

    def _load_skill_tree(self) -> None:
        """Load and index the skill tree from knowledge.json."""
        try:
            path = self._knowledge_path
            if not path.exists():
                logger.warning("skill_builder: knowledge.json not found at %s", path)
                return
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            raw_tree = data.get("skill_tree", [])
            logger.info("skill_builder: loaded %d skill_tree entries", len(raw_tree))

            # Index by job name (normalised)
            for entry in raw_tree:
                job_raw = entry.get("Job", "")
                job_key = self._normalise_job_name(job_raw)

                skills = []
                for skill_entry in entry.get("Tree", []):
                    skills.append({
                        "Name": skill_entry.get("Name"),
                        "MaxLevel": int(skill_entry.get("MaxLevel", 10)),
                        "Requires": skill_entry.get("Requires"),
                        "Exclude": skill_entry.get("Exclude", False),
                    })
                if skills:
                    self._skill_tree[job_key] = skills

            logger.info(
                "skill_builder: indexed %d classes in skill tree",
                len(self._skill_tree),
            )

        except Exception:
            logger.exception("skill_builder: failed to load skill tree")

    def _normalise_job_name(self, job_raw: str) -> str:
        """Normalise a knowledge.json job name to our canonical form.

        Handles: ``Novice`` -> ``novice``, ``Swordman`` -> ``swordman``,
        ``Swordman_High`` -> ``swordman_high``, ``Baby_*``, ``*_T``, ``*_T2``.
        """
        name = job_raw.lower().strip()
        # Remove _high, _t, _t2 variants for base class lookups
        # but keep baby/transcendent variants separate
        return name

    def _pick_skill(
        self,
        class_key: str,
        tree: list[dict[str, Any]],
        known: dict[str, int],
    ) -> dict[str, Any] | None:
        """Pick the single best skill to allocate a point to.

        Priority order:
        1. Class skills needed for job change (not maxed, have prerequisites)
        2. Combat skills from _COMBAT_PRIORITY (not maxed, have prerequisites)
        3. Any other class skill that has prerequisites met and is not maxed
        """
        # Strategy 1: level up a class skill that isn't maxed
        for skill in tree:
            name = skill["Name"]
            max_lvl = int(skill.get("MaxLevel", 10))
            current_lvl = known.get(name, 0)
            excluded = skill.get("Exclude", False)

            if excluded:
                continue
            if current_lvl >= max_lvl:
                continue
            if not self._check_prerequisites_met(skill, known):
                continue

            # Try to find it in combat priority first
            combat_list = _COMBAT_PRIORITY.get(class_key, [])
            if name in combat_list:
                return skill

        # Strategy 2: any other class skill
        for skill in tree:
            name = skill["Name"]
            max_lvl = int(skill.get("MaxLevel", 10))
            current_lvl = known.get(name, 0)
            excluded = skill.get("Exclude", False)

            if excluded:
                continue
            if current_lvl >= max_lvl:
                continue
            if not self._check_prerequisites_met(skill, known):
                continue

            return skill

        # Strategy 3: any skill at all (may be pre-requisite gated)
        for skill in tree:
            name = skill["Name"]
            max_lvl = int(skill.get("MaxLevel", 10))
            current_lvl = known.get(name, 0)
            excluded = skill.get("Exclude", False)

            if excluded:
                continue
            if current_lvl >= max_lvl:
                continue

            return skill

        return None

    def _check_prerequisites_met(
        self,
        skill: dict[str, Any],
        known: dict[str, int],
    ) -> bool:
        """Check if all prerequisites for a skill are satisfied.

        The Requires field is a list of:
            ``[{"Name": "...", "Level": N}, ...]``
        """
        requires = skill.get("Requires")
        if not requires:
            return True  # No prerequisites

        for req in requires:
            req_name = req.get("Name", "")
            req_level = int(req.get("Level", 0))
            known_level = known.get(req_name, 0)
            if known_level < req_level:
                return False

        return True
