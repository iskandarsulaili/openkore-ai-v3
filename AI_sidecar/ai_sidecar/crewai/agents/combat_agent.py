from __future__ import annotations

from typing import Any
from .base_agent import BehaviorProfile



class CombatProfile(BehaviorProfile):
    """Aggro management, attack priority, flee threshold."""

    agent_id = "combat"
    role = "Combat Specialist"
    goal = "Eliminate threats efficiently while minimizing damage taken"
    backstory = (
        "Trained in the heat of countless battles, this agent reads the "
        "battlefield instantly — prioritizing dangerous mobs, managing aggro, "
        "and knowing exactly when to stand ground or flee."
    )

    def can_handle(self, signals: dict[str, Any]) -> float:
        monsters = signals.get("monsters_around", [])
        aggro = signals.get("aggro_list", [])
        if not monsters and not aggro:
            return 0.0
        score = 0.3 * min(len(aggro), 5) / 5.0
        score += 0.4 * min(len(monsters), 10) / 10.0
        hp_pct = signals.get("hp", 1) / max(signals.get("hp_max", 1), 1)
        if hp_pct < 0.3:
            score += 0.3  # danger — combat decisions critical
        return min(score, 1.0)

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        monsters = signals.get("monsters_around", [])
        aggro = signals.get("aggro_list", [])
        hp_pct = signals.get("hp", 1) / max(signals.get("hp_max", 1), 1)

        if hp_pct < 0.2 and aggro:
            return {"kind": "flee", "command": "ai manual_flee", "confidence": 0.95, "reason": "HP critical with aggro"}

        if aggro:
            # target the most dangerous monster (lowest HP% mob = highest threat to us)
            target = min(aggro, key=lambda m: m.get("hp_pct", 1)) if aggro else None
            if target:
                return {"kind": "attack", "command": f"attack {target.get('name', 'monster')}", "confidence": 0.85, "reason": f"Engaging aggro target {target.get('name', '?')}"}

        if monsters:
            target = monsters[0]
            return {"kind": "attack", "command": f"attack {target.get('name', 'monster')}", "confidence": 0.6, "reason": "Clearing nearby monsters"}

        return None
