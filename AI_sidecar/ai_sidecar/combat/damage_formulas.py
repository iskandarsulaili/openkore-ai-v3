"""Damage formulas and cooldown tracking for RO combat."""
import datetime


class SkillCooldownTracker:
    """Track per-skill cooldowns and prevent re-use before expiry."""
    def __init__(self):
        self._last_used: dict[str, datetime.datetime] = {}
        self._cooldowns: dict[str, float] = {}
    
    def set_cooldown(self, skill_name: str, cooldown_seconds: float) -> None:
        self._cooldowns[skill_name] = cooldown_seconds
    
    def record_use(self, skill_name: str) -> None:
        self._last_used[skill_name] = datetime.datetime.now(datetime.timezone.utc)
    
    def is_available(self, skill_name: str) -> bool:
        if skill_name not in self._last_used:
            return True
        cd = self._cooldowns.get(skill_name, 0.0)
        if cd <= 0:
            return True
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - self._last_used[skill_name]).total_seconds()
        return elapsed >= cd
    
    def seconds_until_available(self, skill_name: str) -> float:
        if skill_name not in self._last_used:
            return 0.0
        cd = self._cooldowns.get(skill_name, 0.0)
        if cd <= 0:
            return 0.0
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - self._last_used[skill_name]).total_seconds()
        return max(0.0, cd - elapsed)


def calculate_damage_after_def(raw_damage: float, monster_def: int = 0, monster_mdef: int = 0, is_physical: bool = True) -> float:
    """RO damage formula: DEF/(DEF+100). Returns actual damage after DEF/MDEF."""
    if is_physical:
        if monster_def <= 0:
            return raw_damage
        reduction = monster_def / (monster_def + 100)
        return max(1, int(raw_damage * (1.0 - reduction)))
    else:
        if monster_mdef <= 0:
            return raw_damage
        reduction = monster_mdef / (monster_mdef + 100)
        return max(1, int(raw_damage * (1.0 - reduction)))
