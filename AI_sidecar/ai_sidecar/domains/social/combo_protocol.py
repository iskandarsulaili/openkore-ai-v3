"""Combo Handshake Protocol — signal→confirm→cast→verify for party combos.

On a real server with 200ms latency, combos fail because:
- Target moves between cast start and finish
- Caster thinks receiver is in range but isn't
- Skill "lands" on server but client doesn't see it for 500ms

This protocol adds a handshake layer to party combos:
1. Signal: Caster signals intention ("ready to Aspersio your weapon")
2. Confirm: Receiver confirms ("ready, standing still")
3. Cast: Caster executes
4. Verify: Receiver confirms it landed ("Aspersio received")
5. Resume: Both resume normal operations

Timeout: if no confirm within 3 seconds, abort combo.
"""
from __future__ import annotations
from typing import Any
import logging
from datetime import datetime, timedelta
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

HANDSHAKE_TIMEOUT = timedelta(seconds=5)
VERIFY_TIMEOUT = timedelta(seconds=3)


class ComboHandshakeState:
    """Tracks the state of a combo handshake between party members."""
    
    IDLE = "idle"
    SIGNAL_SENT = "signal_sent"
    CONFIRMED = "confirmed"
    CASTING = "casting"
    VERIFIED = "verified"
    FAILED = "failed"
    TIMEOUT = "timeout"
    
    def __init__(self):
        self.state = self.IDLE
        self.skill_id: str = ""
        self.caster: str = ""
        self.target: str = ""
        self.receiver: str = ""  # The party member receiving the buff
        self.started_at: datetime | None = None
        self.completed_count: int = 0
    
    def start(self, skill_id: str, caster: str, receiver: str) -> None:
        self.state = self.SIGNAL_SENT
        self.skill_id = skill_id
        self.caster = caster
        self.receiver = receiver
        self.started_at = datetime.now()
    
    def confirm(self) -> None:
        self.state = self.CONFIRMED
    
    def cast(self) -> None:
        self.state = self.CASTING
    
    def verify(self) -> None:
        self.state = self.VERIFIED
        self.completed_count += 1
        self.reset()
    
    def fail(self) -> None:
        self.state = self.FAILED
        self.reset()
    
    def is_timed_out(self) -> bool:
        if self.started_at and (datetime.now() - self.started_at) > HANDSHAKE_TIMEOUT:
            self.state = self.TIMEOUT
            self.reset()
            return True
        return False
    
    def reset(self) -> None:
        """Reset to idle but keep completed_count."""
        self.state = self.IDLE
        self.skill_id = ""
        self.caster = ""
        self.receiver = ""
        self.target = ""
        self.started_at = None


# Known combos and their requirements
KNOWLEDGE_COMBOS = {
    "aspersio_combo": {
        "caster_skill": "AL_ASPERSIO",
        "caster_job": "priest",
        "effect": "holy_weapon",
        "description": "Priest buffs weapon with holy element",
        "range": 5,
        "latency_buffer": 0.5,
    },
    "storm_gust_bowling_bash": {
        "caster_skill": "WZ_STORMGUST",
        "pusher_skill": "KN_BOWLINGBASH",
        "effect": "aoe_trap",
        "description": "Knight pushes mobs into Wizard's Storm Gust AoE",
        "range": 5,
        "latency_buffer": 1.0,
    },
    "gloria_crit": {
        "caster_skill": "PR_GLORIA",
        "caster_job": "priest",
        "effect": "crit_buff",
        "description": "Priest casts Gloria for +20% crit on party",
        "range": 9,
        "latency_buffer": 0.3,
    },
    "magnificat_sp": {
        "caster_skill": "PR_MAGNIFICAT",
        "caster_job": "priest",
        "effect": "sp_regen",
        "description": "Priest casts Magnificat for SP regen",
        "range": 9,
        "latency_buffer": 0.3,
    },
    "lex_aeterna_soul_strike": {
        "caster_skill": "PR_LEXAETERNA",
        "followup_skill": "MG_SOULSTRIKE",
        "effect": "double_magic_damage",
        "description": "Lex Aeterna marks target, Soul Strike deals 2x damage",
        "range": 5,
        "latency_buffer": 1.5,
    },
    "frost_combo": {
        "caster_skill": "MG_FROSTDIVER",
        "followup_skill": "MG_COLDBOLT",
        "effect": "elemental_synergy",
        "description": "Freeze target with Frost Diver, then Cold Bolt deals bonus damage",
        "range": 7,
        "latency_buffer": 0.5,
    },
}


class ComboHandshakeProtocol:
    """Manages the handshake protocol for party combos with latency compensation."""
    
    def __init__(self):
        self._active_combos: dict[str, ComboHandshakeState] = {}  # bot_id -> state
    
    def suggest_combo(self, party_members: list[dict[str, Any]]) -> dict | None:
        """Suggest a combo based on available party members."""
        jobs = [m.get("job", "").lower() for m in party_members if isinstance(m, dict)]
        for combo_name, combo_info in KNOWLEDGE_COMBOS.items():
            caster_job = combo_info.get("caster_job", "")
            if caster_job and any(caster_job in j for j in jobs):
                return {"name": combo_name, **combo_info}
        return None
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        """Run combo handshake protocol."""
        party = signals.get("party", {}) or {}
        party_members = party.get("members", []) if isinstance(party, dict) else []
        
        if len(party_members) < 2:
            return
        
        # Check for timeouts on active combos
        if bot_id in self._active_combos:
            combo = self._active_combos[bot_id]
            if combo.is_timed_out():
                actions.append(HeuristicAction(
                    kind="log",
                    command=f"combo_timeout {combo.skill_id}",
                    confidence=0.5,
                    reason=f"Combo handshake timed out for {combo.skill_id}",
                    domain="party",
                ))
                del self._active_combos[bot_id]
        
        # Suggest combos
        suggested = self.suggest_combo(party_members)
        if suggested:
            combo_name = suggested["name"]
            actions.append(HeuristicAction(
                kind="command",
                command=f"combo_suggest {combo_name}",
                confidence=0.6,
                reason=f"Party combo available: {suggested['description']}",
                domain="party",
            ))
            
            # If this bot is the caster, signal readiness
            my_job = str(signals.get("job", "") or "").lower()
            caster_job = suggested.get("caster_job", "")
            if caster_job and caster_job in my_job:
                combo = ComboHandshakeState()
                skill_id = suggested.get("caster_skill", "")
                combo.start(skill_id, bot_id, "party_lead")
                self._active_combos[bot_id] = combo
                
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"combo_signal {skill_id}",
                    confidence=0.8,
                    reason=f"Signaling combo readiness: {suggested['description']}",
                    domain="party",
                ))
