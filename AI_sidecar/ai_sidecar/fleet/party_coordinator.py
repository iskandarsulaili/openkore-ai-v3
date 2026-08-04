"""Party coordination — reflex rules + heuristics for multi-bot team activities."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

CoordinationSignal = dict[str, Any]


@dataclass(slots=True)
class CoordinationAction:
    kind: str
    command: str
    confidence: float
    target_bot: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class PartyCoordinator:
    """Handles party formation, combat coordination, resource sharing, MVP hunting,
    PVP/GVG teaming, trade coordination, and refine/enhance pooling."""

    def __init__(self, fleet_coordinator=None):
        self._fleet = fleet_coordinator
        self._lock = threading.RLock()
        self._cooldowns: dict[str, float] = defaultdict(float)

    def assess_party_formation(self, signals: CoordinationSignal, bot_id: str) -> CoordinationAction | None:
        if self._fleet is None:
            return None
        bot = self._fleet.get_bot(bot_id)
        if bot is None or bot.party_id:
            return None
        fleet_bots = self._fleet.list_bots(online_only=True)
        nearby = [b for b in fleet_bots if b.bot_id != bot_id and b.map_name == bot.map_name]
        if not nearby:
            return None
        my_role = bot.current_role
        for other in nearby:
            if self._roles_complement(my_role, other.current_role):
                # RULE.md party doctrine: ZERO party commands from the sidecar except
                # god_mode party_organize. Forming a party reflexively whenever two bots
                # share a map distracts them from the kill loop (spams party_create/invite,
                # bots stop farming). So this is OBSERVABLE-ONLY: the swarm DETECTS the
                # complementary-role opportunity, but the actual party formation decision is
                # the conscious tier's (LLM/CrewAI), not a reflexive side-effect. Emit a
                # log intent so the opportunity is visible without freezing the bots.
                return CoordinationAction(
                    kind="party_invite_observe", command=f"party_invite {other.bot_id}",
                    confidence=0.7, target_bot=other.bot_id,
                    payload={"party_leader": bot_id, "members": [bot_id, other.bot_id],
                             "observe_only": True},
                    reason=f"[OBSERVE] Complementary roles {my_role}+{other.current_role} — party formation is the conscious tier's call",
                )
        return None

    def _roles_complement(self, role_a: str, role_b: str) -> bool:
        comp = {("tank", "healer"), ("tank", "dps_melee"), ("tank", "dps_ranged"),
                ("tank", "buffer"), ("healer", "dps_melee"), ("healer", "dps_ranged"),
                ("healer", "dps_magic"), ("buffer", "dps_melee"), ("buffer", "dps_ranged"),
                ("support", "dps_melee"), ("debuff", "dps_melee"),
                ("merchant", "crafter"), ("farmer", "merchant"),
                ("mvp_hunter", "healer"), ("mvp_hunter", "tank"),
                ("pvp_attacker", "healer"), ("gvg_frontline", "gvg_support")}
        return (role_a, role_b) in comp or (role_b, role_a) in comp

    def assess_combat_coordination(self, signals: CoordinationSignal, bot_id: str) -> CoordinationAction | None:
        if self._fleet is None:
            return None
        bot = self._fleet.get_bot(bot_id)
        if bot is None:
            return None
        now = time.time()
        if now - self._cooldowns.get(bot_id, 0) < 10:
            return None
        hp_pct = bot.hp_pct()
        party_members = self._fleet.party_members(bot.party_id) if bot.party_id else []
        if hp_pct < 0.4:
            healer = next((m for m in party_members if m.current_role in ("healer", "support", "buffer")), None)
            if healer:
                self._cooldowns[bot_id] = now
                return CoordinationAction(
                    kind="request_heal", command=f"heal_me {bot_id}",
                    confidence=0.85, target_bot=healer.bot_id,
                    payload={"hp_pct": hp_pct, "position": list(bot.position), "map": bot.map_name},
                    reason=f"HP critical ({hp_pct:.0%}), healer {healer.bot_id} in party",
                )
        for member in party_members:
            if member.bot_id != bot_id and member.hp_pct() < 0.3:
                self._cooldowns[bot_id] = now
                return CoordinationAction(
                    kind="coordinate_attack", command=f"assist {member.bot_id}",
                    confidence=0.75, target_bot=member.bot_id,
                    payload={"target": member.active_objective or signals.get("target_monster", "")},
                    reason=f"Party member {member.bot_id} needs assistance (HP {member.hp_pct():.0%})",
                )
        return None

    def assess_resource_sharing(self, signals: CoordinationSignal, bot_id: str) -> CoordinationAction | None:
        if self._fleet is None:
            return None
        bot = self._fleet.get_bot(bot_id)
        if bot is None or not bot.party_id:
            return None
        now = time.time()
        if now - self._cooldowns.get(f"share_{bot_id}", 0) < 60:
            return None
        party_members = self._fleet.party_members(bot.party_id)
        if bot.zeny > 100000 and len(party_members) > 1:
            avg_zeny = sum(m.zeny for m in party_members) / len(party_members)
            for member in party_members:
                if member.bot_id != bot_id and member.zeny < avg_zeny * 0.5:
                    self._cooldowns[f"share_{bot_id}"] = now
                    amount = min(bot.zeny // 4, 50000)
                    return CoordinationAction(
                        kind="share_loot", command=f"give_zeny {member.bot_id} {amount}",
                        confidence=0.6, target_bot=member.bot_id,
                        payload={"amount": amount, "reason": "wealth_equalization"},
                        reason=f"Sharing {amount} zeny with {member.bot_id}",
                    )
        if bot.weight_pct() > 0.8:
            for member in party_members:
                if member.bot_id != bot_id and member.weight_pct() < 0.5:
                    self._cooldowns[f"share_{bot_id}"] = now
                    return CoordinationAction(
                        kind="share_loot", command=f"transfer_loot {member.bot_id}",
                        confidence=0.5, target_bot=member.bot_id,
                        payload={"map": bot.map_name}, reason="Weight management: transferring loot",
                    )
        return None

    def assess_quest_coordination(self, signals: CoordinationSignal, bot_id: str) -> CoordinationAction | None:
        if self._fleet is None:
            return None
        quest_item = signals.get("needs_quest_item", "")
        quest_mob = signals.get("needs_quest_mob", "")
        if not quest_item and not quest_mob:
            return None
        fleet_bots = self._fleet.list_bots(online_only=True)
        helpers = [b for b in fleet_bots if b.bot_id != bot_id and b.current_role in (
            "dps_melee", "dps_ranged", "dps_magic", "tank", "support", "farmer")]
        if not helpers:
            return None
        h = helpers[0]
        return CoordinationAction(
            kind="coordinate_attack", command=f"help_quest {h.bot_id} {quest_item or quest_mob}",
            confidence=0.65, target_bot=h.bot_id,
            payload={"quest_item": quest_item, "quest_mob": quest_mob, "map": signals.get("map_name", "")},
            reason=f"Requesting help for quest: {quest_item or quest_mob}",
        )

    def assess_mvp_coordination(self, signals: CoordinationSignal, bot_id: str) -> CoordinationAction | None:
        if self._fleet is None:
            return None
        mvp_spotted = signals.get("mvp_spotted", "")
        if not mvp_spotted:
            return None
        now = time.time()
        if now - self._cooldowns.get(f"mvp_{bot_id}", 0) < 30:
            return None
        bot = self._fleet.get_bot(bot_id)
        pos = list(bot.position) if bot else []
        fleet_bots = self._fleet.list_bots(online_only=True)
        hunters = [b for b in fleet_bots if b.bot_id != bot_id and b.current_role in (
            "mvp_hunter", "dps_melee", "dps_ranged", "dps_magic", "tank", "healer")]
        if not hunters:
            return None
        self._cooldowns[f"mvp_{bot_id}"] = now
        return CoordinationAction(
            kind="coordinate_attack", command=f"mvp_alert {mvp_spotted}",
            confidence=0.9, target_bot="*",
            payload={"mvp_name": mvp_spotted, "map": signals.get("map_name", ""),
                     "position": pos, "hunters_needed": len(hunters)},
            reason=f"MVP {mvp_spotted} spotted on {signals.get('map_name', '?')}",
        )

    def assess_pvp_gvg_coordination(self, signals: CoordinationSignal, bot_id: str) -> CoordinationAction | None:
        if self._fleet is None:
            return None
        war_type = signals.get("war_type", "")
        if war_type not in ("pvp", "gvg"):
            return None
        now = time.time()
        if now - self._cooldowns.get(f"war_{bot_id}", 0) < 30:
            return None
        fleet_bots = self._fleet.list_bots(online_only=True)
        bot = self._fleet.get_bot(bot_id)
        if war_type == "pvp":
            teammates = [b for b in fleet_bots if b.bot_id != bot_id and b.current_role in (
                "pvp_attacker", "pvp_defender", "healer", "debuff")]
            if teammates:
                self._cooldowns[f"war_{bot_id}"] = now
                all_members = ([bot] + teammates) if bot else teammates
                return CoordinationAction(
                    kind="coordinate_attack", command="pvp_form_team", confidence=0.7,
                    payload={"teammates": [t.bot_id for t in teammates],
                             "roles": {t.bot_id: t.current_role for t in all_members if t}},
                    reason=f"Forming PVP team with {len(teammates)} bots",
                )
        elif war_type == "gvg":
            gvg_roles = [b for b in fleet_bots if b.bot_id != bot_id and b.current_role in (
                "gvg_frontline", "gvg_siege", "gvg_support", "healer")]
            if gvg_roles:
                self._cooldowns[f"war_{bot_id}"] = now
                return CoordinationAction(
                    kind="coordinate_attack", command="gvg_form_squad", confidence=0.75,
                    payload={"squad": [t.bot_id for t in gvg_roles],
                             "frontline": [t.bot_id for t in gvg_roles if t.current_role == "gvg_frontline"],
                             "siege": [t.bot_id for t in gvg_roles if t.current_role == "gvg_siege"]},
                    reason=f"Forming GVG squad with {len(gvg_roles)} bots",
                )
        return None

    def assess_trade_coordination(self, signals: CoordinationSignal, bot_id: str) -> CoordinationAction | None:
        if self._fleet is None:
            return None
        if not signals.get("inventory_full", False) and not signals.get("crafted_items", []):
            return None
        fleet_bots = self._fleet.list_bots(online_only=True)
        merchant = next((b for b in fleet_bots if b.current_role in ("merchant", "crafter")), None)
        if merchant:
            return CoordinationAction(
                kind="coordinate_attack", command=f"trade_with {merchant.bot_id}",
                confidence=0.65, target_bot=merchant.bot_id,
                payload={"items": list(signals.get("crafted_items", [])), "map": signals.get("map_name", "")},
                reason=f"Coordinating trade with merchant {merchant.bot_id}",
            )
        return None

    def assess_refine_pooling(self, signals: CoordinationSignal, bot_id: str) -> CoordinationAction | None:
        if self._fleet is None:
            return None
        if not signals.get("needs_refine", False):
            return None
        refine_material = signals.get("refine_material", "")
        if not refine_material:
            return None
        fleet_bots = self._fleet.list_bots(online_only=True)
        refiner = next((b for b in fleet_bots if b.current_role == "refiner"), None)
        if refiner:
            return CoordinationAction(
                kind="coordinate_attack", command=f"pool_refine {refiner.bot_id} {refine_material}",
                confidence=0.6, target_bot=refiner.bot_id,
                payload={"material": refine_material, "contributors": [bot_id]},
                reason=f"Pooling resources for refine with {refiner.bot_id}",
            )
        return None

    def assess(self, signals: CoordinationSignal, bot_id: str) -> CoordinationAction | None:
        best = None
        best_c = 0.0
        for fn in [self.assess_party_formation, self.assess_combat_coordination,
                    self.assess_resource_sharing, self.assess_quest_coordination,
                    self.assess_mvp_coordination, self.assess_pvp_gvg_coordination,
                    self.assess_trade_coordination, self.assess_refine_pooling]:
            try:
                a = fn(signals, bot_id)
                if a and a.confidence > best_c:
                    best_c, best = a.confidence, a
            except Exception as e:
                logger.warning("PartyCoordinator check failed: %s", e)
        return best
