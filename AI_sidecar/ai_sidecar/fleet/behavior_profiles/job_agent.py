"""JobAgent — auto job change, stat allocation, skill allocation, job level tracking."""

from __future__ import annotations

from typing import Any

from ai_sidecar.fleet.behavior_profiles import BehaviorProfile

_STAT_DEFAULTS = {
    "swordsman": {"str": 9, "agi": 1, "vit": 9, "int": 1, "dex": 5, "luk": 1},
    "mage": {"str": 1, "agi": 1, "vit": 5, "int": 9, "dex": 7, "luk": 1},
    "archer": {"str": 1, "agi": 9, "vit": 5, "int": 1, "dex": 9, "luk": 1},
    "acolyte": {"str": 5, "agi": 5, "vit": 5, "int": 9, "dex": 5, "luk": 1},
    "merchant": {"str": 9, "agi": 1, "vit": 9, "int": 5, "dex": 5, "luk": 1},
    "thief": {"str": 5, "agi": 9, "vit": 5, "int": 1, "dex": 9, "luk": 1},
}

_BUILD_TEMPLATES = {
    "knight_agi": {"str": 60, "agi": 80, "vit": 40, "int": 1, "dex": 40, "luk": 1},
    "knight_vit": {"str": 80, "agi": 30, "vit": 70, "int": 1, "dex": 40, "luk": 1},
    "wizard_int": {"str": 1, "agi": 40, "vit": 30, "int": 99, "dex": 60, "luk": 1},
    "hunter_dex": {"str": 1, "agi": 60, "vit": 30, "int": 1, "dex": 99, "luk": 30},
    "priest_full": {"str": 1, "agi": 30, "vit": 40, "int": 99, "dex": 60, "luk": 1},
    "assassin_str": {"str": 70, "agi": 80, "vit": 30, "int": 1, "dex": 40, "luk": 1},
    "blacksmith_str": {"str": 80, "agi": 40, "vit": 50, "int": 1, "dex": 40, "luk": 1},
}

_JOB_CHANGE_NPCS = {
    "novice": {"swordsman": ("prontera", 163, 42), "mage": ("geffen", 60, 140),
               "archer": ("payon", 186, 230), "acolyte": ("prontera", 195, 230),
               "merchant": ("alberta", 48, 152), "thief": ("morocc", 144, 306)},
    "swordsman": {"knight": ("prontera", 170, 30), "crusader": ("payon", 100, 200)},
    "mage": {"wizard": ("geffen", 50, 130), "sage": ("geffen", 70, 150)},
    "archer": {"hunter": ("payon", 150, 240), "bard": ("payon", 165, 255)},
    "acolyte": {"priest": ("prontera", 195, 230), "monk": ("morocc", 120, 280)},
    "merchant": {"blacksmith": ("geffen", 80, 160), "alchemist": ("aldebaran", 100, 100)},
    "thief": {"assassin": ("morocc", 120, 280), "rogue": ("alberta", 60, 160)},
}

_SKILL_PRIORITY = {
    "swordsman": ["bash", "magnum_break", "provoke", "endure"],
    "mage": ["fire_bolt", "cold_bolt", "lightning_bolt", "soul_strike", "fire_wall"],
    "archer": ["double_strafe", "improve_concentration", "vulture_eye", "owl_eye"],
    "acolyte": ["heal", "increase_agi", "blessing", "teleport"],
    "merchant": ["discount", "overcharge", "identify", "pushcart", "vending"],
    "thief": ["double_attack", "steal", "hiding", "envenom"],
    "knight": ["bowling_bash", "brandish_spear", "cavalier_mastery", "provoke"],
    "wizard": ["fire_bolt", "frost_diver", "lightning_bolt", "fire_wall", "storm_gust"],
    "hunter": ["double_strafe", "blitz_beat", "improve_concentration"],
    "priest": ["heal", "increase_agi", "blessing", "resurrection", "gloria"],
    "blacksmith": ["weapon_perfection", "over_thrust", "adrenaline_rush", "discount"],
    "assassin": ["sonic_blow", "cloaking", "enchant_deadly_poison", "grimtooth"],
}


class JobAgent(BehaviorProfile):
    """Handles RO job progression — change, stats, skills, and level tracking."""

    def stat_allocation(self, current_class: str, current_stats: dict[str, int],
                        base_level: int, build_name: str = "") -> list[str]:
        template = _BUILD_TEMPLATES.get(build_name) or _STAT_DEFAULTS.get(current_class, _STAT_DEFAULTS["swordsman"])
        to_add = []
        for stat, target in template.items():
            have = current_stats.get(stat, 1)
            remaining = target - have
            if remaining > 0:
                to_add.extend([f"stat_add {stat}"] * min(remaining, 5))
        return to_add[:10]

    def skill_allocation(self, current_class: str,
                         current_skills: dict[str, int]) -> list[str]:
        priority_skills = _SKILL_PRIORITY.get(current_class, [])
        to_learn = []
        for skill in priority_skills:
            current_lvl = current_skills.get(skill, 0)
            if current_lvl < 10:
                to_learn.append(f"skill_learn {skill} 1")
        return to_learn[:5]

    def job_change(self, current_job: str, target_job: str, base_level: int,
                   job_level: int) -> dict[str, Any]:
        routes = _JOB_CHANGE_NPCS.get(current_job, {})
        data = routes.get(target_job)
        if not data:
            return {"action": "no_route", "current": current_job, "target": target_job,
                    "known_routes": list(routes.keys())}
        map_name, x, y = data
        if current_job == "novice" and job_level < 10:
            return {"action": "train_job_level", "current_jlvl": job_level,
                    "needed": 10 - job_level, "target_job": target_job}
        if current_job != "novice" and (base_level < 40 or job_level < 40):
            return {"action": "train_for_change", "needed_base": max(0, 40 - base_level),
                    "needed_job": max(0, 40 - job_level)}
        return {"action": "execute_job_change", "current": current_job,
                "target": target_job, "npc_map": map_name, "npc_coords": (x, y),
                "steps": ["go_to_map", "talk_npc", "complete_change"]}

    def recommend_build(self, target_job: str, playstyle: str = "") -> dict[str, Any]:
        known = {k: v for k, v in _BUILD_TEMPLATES.items() if k.startswith(target_job.lower())}
        if not known:
            stat_base = _STAT_DEFAULTS.get(target_job, _STAT_DEFAULTS["swordsman"])
            return {"build_name": f"{target_job}_balanced", "stats": stat_base,
                    "note": "using default"}
        build_name = f"{target_job}_{playstyle}" if playstyle and f"{target_job}_{playstyle}" in known else list(known.keys())[0]
        return {"build_name": build_name, "stats": known[build_name],
                "recommended_skills": _SKILL_PRIORITY.get(target_job, [])}

    def job_level_tracker(self, current_jlvl: int, max_jlvl: int,
                          current_class: str, xp_rate: float) -> dict[str, Any]:
        pct = current_jlvl / max_jlvl if max_jlvl > 0 else 0
        if current_jlvl >= max_jlvl:
            return {"action": "max_job_level", "class": current_class,
                    "suggestion": "change_class_or_transcend"}
        if pct > 0.5:
            return {"action": "nearing_job_cap", "progress": f"{pct:.0%}",
                    "remaining": max_jlvl - current_jlvl}
        return {"action": "leveling_job", "progress": f"{pct:.0%}",
                "xp_rate": xp_rate}

    def record_outcome(self, action: str, success: bool, jlvl_gained: float = 0.0) -> None:
        self._record_experience("job", action, success, reward=jlvl_gained,
                                job_level_gained=jlvl_gained)
