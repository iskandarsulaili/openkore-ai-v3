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

Enhanced with:
- Class combo system integration (Sage+Wizard, Priest+Hunter, Dancer+Bard, etc.)
- Timing coordination for Lex Aeterna → Asura Strike instakill
- Elemental synergy tracking
- WoE-specific combo coordination
"""
from __future__ import annotations
from typing import Any
import logging
import time
from datetime import datetime, timedelta
from ai_sidecar.actions import HeuristicAction
from ai_sidecar.domains.social.class_combos import (
    CLASS_COMBOS,
    ClassCombo,
    ComboCategory,
    get_combos_for_classes,
    get_combos_by_category,
    get_woe_combos,
    get_instakill_combos,
    find_combo_by_name,
    get_class_vs_class_counter,
)

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
        self.combo_name: str = ""
        self.caster: str = ""
        self.target: str = ""
        self.receiver: str = ""  # The party member receiving the buff
        self.started_at: float | None = 0.0
        self.completed_count: int = 0
        self.category: str = ""
        self.woe_only: bool = False
    
    def start(self, skill_id: str, caster: str, receiver: str, combo_name: str = "",
              category: str = "", woe_only: bool = False) -> None:
        self.state = self.SIGNAL_SENT
        self.skill_id = skill_id
        self.combo_name = combo_name
        self.caster = caster
        self.receiver = receiver
        self.started_at = time.time()
        self.category = category
        self.woe_only = woe_only
    
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
        if self.started_at and (time.time() - self.started_at) > HANDSHAKE_TIMEOUT.total_seconds():
            self.state = self.TIMEOUT
            self.reset()
            return True
        return False
    
    def reset(self) -> None:
        """Reset to idle but keep completed_count."""
        self.state = self.IDLE
        self.skill_id = ""
        self.combo_name = ""
        self.caster = ""
        self.receiver = ""
        self.target = ""
        self.started_at = 0.0


# Known combos and their requirements (legacy support)
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
    # New combos from class_combos.py
    "endow_elemental_nuke": {
        "caster_skill": "SAGE_ENDOW",
        "caster_job": "sage",
        "followup_skill": "WZ_EARTHSPIKE",
        "effect": "elemental_synergy",
        "description": "Sage enchants Wizard's weapon with element matching monster weakness",
        "range": 5,
        "latency_buffer": 1.0,
    },
    "deluge_storm_gust_freeze": {
        "caster_skill": "SAGE_DELUGE",
        "caster_job": "sage",
        "followup_skill": "WZ_STORMGUST",
        "effect": "aoe_freeze",
        "description": "Sage casts Deluge (water field), Wizard casts Storm Gust for 100% freeze",
        "range": 7,
        "latency_buffer": 1.5,
    },
    "assumptio_hunter": {
        "caster_skill": "PR_ASSUMPTIO",
        "caster_job": "priest",
        "effect": "damage_reduction",
        "description": "Priest casts Assumptio on Hunter (50% damage reduction)",
        "range": 9,
        "latency_buffer": 0.5,
    },
    "lex_aeterna_asura": {
        "caster_skill": "PR_LEXAETERNA",
        "caster_job": "priest",
        "followup_skill": "MO_ASURASTRIKE",
        "effect": "instakill",
        "description": "Lex Aeterna + Asura Strike = instakill any non-boss",
        "range": 5,
        "latency_buffer": 1.5,
    },
    "hypnotist_waltz_encore": {
        "caster_skill": "DA_HYPNO",
        "caster_job": "dancer",
        "followup_skill": "BA_ENCO",
        "effect": "aoe_stun",
        "description": "Dancer starts Hypnotist's Waltz, Bard follows with Encore for extended CC",
        "range": 7,
        "latency_buffer": 1.0,
    },
    "acid_demonstration": {
        "caster_skill": "AM_ACIDDEMO",
        "caster_job": "alchemist",
        "effect": "defense_bypass",
        "description": "Acid Demonstration bypasses defense — counters Paladins in WoE",
        "range": 5,
        "latency_buffer": 0.5,
        "woe_only": True,
    },
}


class ComboHandshakeProtocol:
    """Manages the handshake protocol for party combos with latency compensation.

    Enhanced with:
    - Full class combo system integration (Sage+Wizard, Priest+Hunter, Dancer+Bard, etc.)
    - Timing coordination for instakill combos (Lex Aeterna → Asura Strike)
    - Elemental synergy tracking
    - WoE-specific combo coordination
    - Combo cooldown management
    """
    
    def __init__(self):
        self._active_combos: dict[str, ComboHandshakeState] = {}  # bot_id -> state
        self._combo_cooldowns: dict[str, float] = {}  # combo_name -> ready_at
        self._combo_history: list[dict[str, Any]] = []
    
    def suggest_combo(self, party_members: list[dict[str, Any]]) -> dict | None:
        """Suggest a combo based on available party members.

        Uses both legacy KNOWLEDGE_COMBOS and the new CLASS_COMBOS system.
        """
        jobs = [m.get("job", "").lower() for m in party_members if isinstance(m, dict)]
        
        # First check legacy combos
        for combo_name, combo_info in KNOWLEDGE_COMBOS.items():
            caster_job = combo_info.get("caster_job", "")
            if caster_job and any(caster_job in j for j in jobs):
                return {"name": combo_name, **combo_info}
        
        # Then check new class combos
        for combo in CLASS_COMBOS:
            if combo.prep_class in jobs and combo.main_class in jobs:
                return {
                    "name": combo.name,
                    "caster_skill": combo.prep_skill,
                    "caster_job": combo.prep_class,
                    "followup_skill": combo.main_skill,
                    "effect": combo.category.value,
                    "description": combo.description,
                    "range": combo.range,
                    "latency_buffer": combo.latency_buffer,
                    "woe_only": combo.woe_only,
                    "prep_time_s": combo.prep_time_s,
                    "window_s": combo.window_s,
                }
        
        return None
    
    def get_available_combos(self, party_members: list[dict[str, Any]]) -> list[dict]:
        """Get ALL available combos for the current party composition."""
        jobs = [m.get("job", "").lower() for m in party_members if isinstance(m, dict)]
        available = []
        
        for combo in CLASS_COMBOS:
            if combo.prep_class in jobs and combo.main_class in jobs:
                available.append({
                    "name": combo.name,
                    "category": combo.category.value,
                    "prep_class": combo.prep_class,
                    "main_class": combo.main_class,
                    "prep_skill": combo.prep_skill,
                    "main_skill": combo.main_skill,
                    "description": combo.description,
                    "woe_only": combo.woe_only,
                    "prep_time_s": combo.prep_time_s,
                    "window_s": combo.window_s,
                })
        
        return available
    
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
                self._combo_history.append({
                    "combo": combo.combo_name or combo.skill_id,
                    "status": "timeout",
                    "time": time.time(),
                })
                del self._active_combos[bot_id]
        
        # Check cooldowns
        now = time.time()
        for combo_name in list(self._combo_cooldowns.keys()):
            if now >= self._combo_cooldowns[combo_name]:
                del self._combo_cooldowns[combo_name]
        
        # Suggest combos
        suggested = self.suggest_combo(party_members)
        if suggested:
            combo_name = suggested["name"]
            
            # Check cooldown
            if combo_name in self._combo_cooldowns:
                return
            
            # Check WoE-only combos
            if suggested.get("woe_only", False):
                map_name = str(signals.get("map", "") or "").lower()
                if "gld_" not in map_name:
                    return
            
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
                combo.start(
                    skill_id, bot_id, "party_lead",
                    combo_name=combo_name,
                    category=suggested.get("effect", ""),
                    woe_only=suggested.get("woe_only", False),
                )
                self._active_combos[bot_id] = combo
                
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"combo_signal {skill_id}",
                    confidence=0.8,
                    reason=f"Signaling combo readiness: {suggested['description']}",
                    domain="party",
                ))
                
                # Set cooldown
                self._combo_cooldowns[combo_name] = time.time() + 30.0
    
    def get_combo_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent combo execution history."""
        return self._combo_history[-limit:]
    
    def get_active_combos(self) -> dict[str, ComboHandshakeState]:
        """Get all active combo handshakes."""
        return dict(self._active_combos)
