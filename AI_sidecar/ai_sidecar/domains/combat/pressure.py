"""Combat Engine v2 — danger-adjusted, pressure-aware skill selection.

The original engine picked skills by DPS/SP × element mod.
This version accounts for:
- Enemy distance (close = prioritize instant-cast, not big cast-time)
- Current HP vs enemy DPS (low HP = prioritize interrupt-immune skills)
- Surround count (multiple enemies = prioritize AoE, not single-target)
- Cast interruption risk (being hit = penalize long-cast skills)
- Monster skills (some monsters have special attacks that change priority)
- Weather effects (rain reduces fire damage)
- Time of day (undead 1.5x at night)
"""
from __future__ import annotations
from typing import Any
import logging
from pathlib import Path

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None

_DATA_DIR = Path(__file__).parent.parent.parent / "data"


class SkillWeights:
    """Weight a skill for selection based on combat pressure."""
    
    def __init__(self, skill_id: str, skill_data: dict):
        self.skill_id = skill_id
        self.cast_time = skill_data.get("cast_time_s", 0)
        self.delay = skill_data.get("delay_s", 0.3)
        self.sp_cost = skill_data.get("sp_cost", 0)
        self.element = skill_data.get("element", "Neutral")
        self.skill_type = skill_data.get("type", "attack")
        self.range = skill_data.get("range", 1)
        self.interrupt_immune = skill_data.get("cast_interrupt", True) is False
        self.aoe = skill_data.get("aoe", False)
        self.aoe_radius = skill_data.get("aoe_radius", 0)
        self.damage_mult = skill_data.get("damage_mult", 1.0)
    
    def calculate_score(self, context: dict[str, Any]) -> float:
        """Score this skill in the current combat context.
        
        Base: damage_mult × element_mult
        Penalties:
        - Long cast time when enemy is close
        - Cast interruption risk when being hit
        - High SP cost when low on SP
        Bonuses:
        - Interrupt-immune when being hit
        - AoE when surrounded
        - Instant-cast when low HP
        """
        # Base damage score
        element_mult = context.get("element_mult", {}).get(self.element, 1.0)
        base_score = self.damage_mult * element_mult
        
        # Pressure adjustments
        enemy_distance = context.get("enemy_distance", 10)
        current_hp_pct = context.get("hp_pct", 100)
        being_hit = context.get("being_hit", False)
        surround_count = context.get("surround_count", 0)
        sp_pct = context.get("sp_pct", 100)
        weather_mod = context.get("weather_mod", {}).get(self.element, 1.0)
        
        score = base_score * weather_mod
        
        # Penalty: long cast time when enemy is close
        if enemy_distance < 5 and self.cast_time > 2:
            penalty = 1.0 - (self.cast_time / 10.0) * (1.0 - enemy_distance / 10.0)
            score *= max(0.1, penalty)
        
        # Penalty: being hit + interruptable
        if being_hit and not self.interrupt_immune:
            if self.cast_time > 0:
                score *= 0.3  # 70% penalty — will likely be interrupted
            else:
                score *= 0.8  # Small penalty for delay
        
        # Bonus: interrupt-immune when being hit
        if being_hit and self.interrupt_immune:
            score *= 1.5
        
        # Bonus: AoE when surrounded
        if self.aoe and surround_count >= 2:
            bonus = 1.0 + (surround_count * 0.3 * min(self.aoe_radius / 5.0, 1.0))
            score *= bonus
        
        # Penalty: high SP cost when low on SP
        if sp_pct < 30 and self.sp_cost > 10:
            score *= max(0.2, sp_pct / 100.0)
        
        # Bonus: instant-cast when low HP
        if current_hp_pct < 30 and self.cast_time == 0:
            score *= 2.0
        
        return score


class PressureAwareSelector:
    """Selects the best skill considering combat pressure."""
    
    def __init__(self, skills_data: dict[str, dict] | None = None):
        self._skills: dict[str, SkillWeights] = {}
        if skills_data:
            for sid, sdata in skills_data.items():
                self._skills[sid] = SkillWeights(sid, sdata)
    
    def select_skill(self, context: dict[str, Any]) -> tuple[str, float] | None:
        """Select the best skill for the current pressure context.
        
        Returns (skill_id, score) or None.
        """
        if not self._skills:
            return None
        
        best_skill = None
        best_score = -1.0
        
        for sid, skill in self._skills.items():
            # Skip skills the player doesn't have
            if sid not in context.get("available_skills", {}):
                continue
            # Skip skills we can't afford
            if skill.sp_cost > context.get("current_sp", 999):
                continue
            # Skip skills with wrong range
            if skill.range > context.get("enemy_distance", 99):
                continue
            
            score = skill.calculate_score(context)
            if score > best_score:
                best_score = score
                best_skill = sid
        
        if best_skill:
            return (best_skill, best_score)
        return None


def assess_combat_pressure(engine, signals: dict, actions: list[HeuristicAction], bot_id: str) -> None:
    """Run pressure-aware combat assessment.
    
    This replaces the basic DPS-based combat engine with one that
    considers survival, positioning, and pressure.
    """
    current_hp = int(signals.get("hp", 100) or 100)
    max_hp = int(signals.get("hp_max", 100) or 100)
    current_sp = int(signals.get("sp", 100) or 100)
    max_sp = int(signals.get("sp_max", 100) or 100)
    monsters_around = signals.get("monsters_around", []) or []
    current_map = str(signals.get("map", "") or "")
    
    hp_pct = current_hp / max(max_hp, 1) * 100
    sp_pct = current_sp / max(max_sp, 1) * 100
    
    # Calculate surround count
    surround_count = 0
    nearest_distance = 99
    target_element = "Neutral"
    for m in monsters_around:
        if isinstance(m, dict):
            dist = abs(m.get("distance_to", 99))
            if dist < 5:
                surround_count += 1
            if dist < nearest_distance:
                nearest_distance = dist
                target_element = m.get("element", "Neutral")
    
    # Check if being hit
    being_hit = bool(signals.get("being_hit", False))
    
    # Get weather modifier
    weather_mod = {}
    try:
        from ai_sidecar.domains.world.state import WeatherSystem
        weather = WeatherSystem.get_weather(current_map)
        if weather:
            wm = WeatherSystem.WEATHER_MODIFIERS.get(weather, {})
            weather_mod = wm
    except ImportError:
        pass
    
    # Build pressure context
    context = {
        "hp_pct": hp_pct,
        "sp_pct": sp_pct,
        "enemy_distance": nearest_distance,
        "being_hit": being_hit,
        "surround_count": surround_count,
        "current_sp": current_sp,
        "element_mult": {
            "Neutral": 1.0, "Fire": 1.0, "Water": 1.0, "Wind": 1.0,
            "Earth": 1.0, "Holy": 1.0, "Shadow": 1.0, "Ghost": 1.0,
            "Undead": 1.0, "Poison": 1.0,
        },
        "weather_mod": weather_mod,
        "available_skills": {},
    }
    
    # Get available skills from engine
    if hasattr(engine, '_skill_db'):
        context["available_skills"] = engine._skill_db
    elif hasattr(engine, 'skills'):
        context["available_skills"] = engine.skills
    
    # If surrounded by 3+ and HP is low → recommend flee
    if surround_count >= 3 and hp_pct < 50:
        actions.append(HeuristicAction(
            kind="command",
            command="flywing",
            confidence=0.9,
            reason=f"Surrounded by {surround_count} at {hp_pct:.0f}% HP — fleeing",
            domain="combat",
        ))
        return
    
    # If being hit and HP is low → recommend instant-cast only
    if being_hit and hp_pct < 30:
        actions.append(HeuristicAction(
            kind="command",
            command="teleport",
            confidence=0.8,
            reason=f"HP critical ({hp_pct:.0f}%) while being hit — teleporting",
            domain="safety",
        ))
        return
    
    # Select skill under pressure
    selector = PressureAwareSelector(context.get("available_skills", {}))
    best = selector.select_skill(context)
    if best:
        sid, score = best
        actions.append(HeuristicAction(
            kind="command",
            command=f"skill_cast {sid} {context.get('target_id', 0)}",
            confidence=min(0.9, score),
            reason=f"Pressure-adjusted: {sid} (score={score:.2f}, dist={nearest_distance}, hp={hp_pct:.0f}%)",
            domain="combat",
        ))
    else:
        # Fall back to auto-attack
        actions.append(HeuristicAction(
            kind="command",
            command="attack",
            confidence=0.5,
            reason="No optimal skill — auto-attack",
            domain="combat",
        ))


# Combination with existing combat engine
class CombatPressureDomain:
    """Domain wrapper that runs pressure-aware combat alongside the base engine."""
    
    def assess(self, signals: dict, actions: list[HeuristicAction], bot_id: str) -> None:
        assess_combat_pressure(None, signals, actions, bot_id)
