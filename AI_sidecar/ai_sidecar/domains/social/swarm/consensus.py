"""Swarm consensus — group decision making for the bot swarm.

Handles:
  - Map voting: each bot votes on which map to hunt, leader chooses
  - Retreat decisions: bots vote on whether to retreat based on HP/resources
  - Hunt target selection: pick the best monster to farm as a group
  - Migration decisions: when to move as a group to a new map

The party leader (highest-level bot) collects votes, applies consensus
rules, and makes the final decision.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ai_sidecar.domains.social.swarm.communication import BotSwarmState

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
#  Vote types
# ────────────────────────────────────────────────────────────────

@dataclass
class HuntMapVote:
    """A bot's vote on which map to hunt."""
    bot_name: str
    preferred_map: str
    confidence: float  # 0.0–1.0 how strongly they feel
    reason: str = ""

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class RetreatVote:
    """A bot's vote on whether the party should retreat."""
    bot_name: str
    should_retreat: bool
    reason: str = ""
    urgency: float = 0.5  # 0.0 = mild, 1.0 = critical


@dataclass
class HuntTargetVote:
    """A bot's preferred monster target."""
    bot_name: str
    monster_name: str
    level_diff: int = 0  # difference from bot's level
    kill_efficiency: float = 0.5  # how fast they can kill it
    confidence: float = 0.5


@dataclass
class ConsensusResult:
    """The result of a consensus decision."""
    topic: str
    decision: str | bool
    threshold: float
    agreement: float  # 0.0–1.0
    total_votes: int
    votes_for: int
    votes_against: int
    abstentions: int
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


# ────────────────────────────────────────────────────────────────
#  ConsensusEngine
# ────────────────────────────────────────────────────────────────

class ConsensusEngine:
    """Handles group decision-making for the swarm.

    The party leader calls these methods to aggregate bot opinions,
    determine consensus, and produce a final decision.
    """

    def __init__(self, default_threshold: float = 0.66) -> None:
        self._default_threshold = default_threshold
        self._last_decisions: dict[str, ConsensusResult] = {}
        self._decision_cooldowns: dict[str, float] = {}

    # ── Hunt map voting ─────────────────────────────────────────

    def decide_hunt_map(
        self,
        bot_states: dict[str, BotSwarmState],
        available_maps: list[str] | None = None,
        threshold: float | None = None,
        cooldown_key: str = "hunt_map",
    ) -> ConsensusResult:
        """Decide which map the party should hunt on.

        Each bot's vote is in bot_states[bot_name].vote_hunt_map.
        The leader considers:
          1. Map with most votes (weighted by confidence)
          2. If tied, leader picks based on level-appropriate content
          3. Cooldown prevents flipping maps too often

        Returns a ConsensusResult with the decided map name.
        """
        now = time.time()
        cooldown = self._decision_cooldowns.get(cooldown_key, 0)
        if now - cooldown < 30:
            # Return last decision within cooldown
            last = self._last_decisions.get(cooldown_key)
            if last:
                return last

        threshold = threshold or self._default_threshold
        votes: dict[str, float] = {}

        for bot_name, state in bot_states.items():
            preferred = state.vote_hunt_map or state.current_hunt_map or ""
            vote_conf = state.vote_confidence if state.vote_hunt_map else 0.3

            if preferred:
                votes[preferred] = votes.get(preferred, 0.0) + vote_conf

        if not votes:
            return ConsensusResult(
                topic=cooldown_key, decision="",
                threshold=threshold, agreement=0.0,
                total_votes=0, votes_for=0, votes_against=0,
                abstentions=len(bot_states),
                reason="No bots voted for any map",
            )

        # Count weighted votes
        total_weight = sum(votes.values())
        best_map = max(votes, key=lambda m: votes[m])
        best_weight = votes[best_map]
        agreement = best_weight / max(0.001, total_weight)

        final_map = best_map

        # If available_maps is given, constrain the choice
        if available_maps and final_map not in available_maps:
            # Pick best map from available
            available_votes = {m: v for m, v in votes.items() if m in available_maps}
            if available_votes:
                final_map = max(available_votes, key=lambda m: available_votes[m])
            else:
                final_map = available_maps[0] if available_maps else final_map
                agreement = 0.3  # low agreement since no one voted for it

        abstentions = sum(
            1 for _, s in bot_states.items()
            if not s.vote_hunt_map and not s.current_hunt_map
        )

        result = ConsensusResult(
            topic=cooldown_key,
            decision=final_map,
            threshold=threshold,
            agreement=agreement,
            total_votes=len(bot_states),
            votes_for=sum(1 for _, v in votes.items() if v > 0),
            votes_against=0,
            abstentions=abstentions,
            reason=f"Map '{final_map}' won with {best_weight:.1f}/{total_weight:.1f} weight ({agreement:.0%} agreement)",
            timestamp=now,
        )

        self._last_decisions[cooldown_key] = result
        self._decision_cooldowns[cooldown_key] = now
        return result

    # ── Retreat voting ──────────────────────────────────────────

    def decide_retreat(
        self,
        bot_states: dict[str, BotSwarmState],
        team_hp_avg: float | None = None,
        threshold: float | None = None,
    ) -> ConsensusResult:
        """Decide whether the party should retreat.

        Each bot's vote is in bot_states[bot_name].vote_retreat.
        The leader also considers overall party HP.
        """
        threshold = threshold or self._default_threshold
        total_bots = len(bot_states)
        retreat_votes = 0
        reasons: list[str] = []
        max_urgency = 0.0

        for bot_name, state in bot_states.items():
            if state.vote_retreat:
                retreat_votes += 1
                max_urgency = max(max_urgency, state.vote_confidence)
                reasons.append(f"{bot_name} HP={state.hp_pct:.0%} urgent={state.vote_confidence:.2f}")

        # Also auto-retreat if average party HP is critically low
        force_retreat = False
        if team_hp_avg is not None and team_hp_avg < 0.2:
            force_retreat = True
            reasons.append(f"Party HP critical ({team_hp_avg:.0%})")

        # Also if any single bot is critical (HP < 15%)
        critical_bots = [n for n, s in bot_states.items() if s.hp_pct < 0.15]
        if critical_bots:
            force_retreat = True
            reasons.append(f"Critical bots: {', '.join(critical_bots)}")

        retreat_ratio = retreat_votes / max(1, total_bots)
        should_retreat = force_retreat or (retreat_ratio >= threshold)
        max_urgency = 1.0 if force_retreat else max_urgency

        result = ConsensusResult(
            topic="retreat",
            decision=should_retreat,
            threshold=threshold,
            agreement=retreat_ratio if not force_retreat else 1.0,
            total_votes=total_bots,
            votes_for=retreat_votes,
            votes_against=total_bots - retreat_votes,
            abstentions=0,
            reason="; ".join(reasons) if reasons else "No retreat votes",
            timestamp=time.time(),
        )
        self._last_decisions["retreat"] = result
        return result

    # ── Hunt target selection ──────────────────────────────────

    def decide_hunt_target(
        self,
        bot_states: dict[str, BotSwarmState],
        available_monsters: list[dict[str, Any]] | None = None,
        leader_level: int = 1,
    ) -> ConsensusResult:
        """Choose the best monster for the party to hunt collectively.

        Args:
            bot_states: All bot swarm states.
            available_monsters: List of monster dicts with 'name', 'level', etc.
            leader_level: The party leader's base level.

        Returns:
            ConsensusResult with the monster name as the decision.
        """
        votes: dict[str, float] = {}

        for bot_name, state in bot_states.items():
            preferred = state.target_monster or ""
            vote_conf = state.vote_confidence if state.target_monster else 0.0
            if preferred:
                votes[preferred] = votes.get(preferred, 0.0) + vote_conf

        if not votes and available_monsters:
            # No votes — pick best monster for party level
            suitable = [
                m for m in available_monsters
                if abs(m.get("level", 50) - leader_level) <= 15
            ]
            if not suitable:
                suitable = available_monsters
            best = max(suitable, key=lambda m: m.get("exp", 1))
            return ConsensusResult(
                topic="hunt_target",
                decision=best.get("name", ""),
                threshold=0.0,
                agreement=0.0,
                total_votes=0,
                votes_for=0,
                votes_against=0,
                abstentions=len(bot_states),
                reason=f"Auto-selected '{best.get('name', '')}' based on party level {leader_level}",
            )

        if not votes:
            return ConsensusResult(
                topic="hunt_target", decision="",
                threshold=0.0, agreement=0.0,
                total_votes=0, votes_for=0, votes_against=0,
                abstentions=len(bot_states),
                reason="No votes and no available monsters",
            )

        total_weight = sum(votes.values())
        best_target = max(votes, key=lambda m: votes[m])
        best_weight = votes[best_target]
        agreement = best_weight / max(0.001, total_weight)

        # Count how many bots voted for the best target vs others
        votes_for = sum(1 for m, v in votes.items() if v > 0 and m == best_target)
        votes_against = sum(1 for m, v in votes.items() if v > 0 and m != best_target)
        abstentions = sum(1 for _, s in bot_states.items() if not s.target_monster)

        return ConsensusResult(
            topic="hunt_target",
            decision=best_target,
            threshold=0.33,  # Lower threshold for target since it changes fast
            agreement=agreement,
            total_votes=votes_for + votes_against,
            votes_for=votes_for,
            votes_against=votes_against,
            abstentions=abstentions,
            reason=f"Target '{best_target}' with {best_weight:.1f}/{total_weight:.1f} weight",
            timestamp=time.time(),
        )

    # ── Party composition ──────────────────────────────────────

    def analyze_composition(self, bot_states: dict[str, BotSwarmState]) -> dict[str, Any]:
        """Analyze the party composition from bot states.

        Returns:
            Dict with:
              - roles: dict[str, list[str]] — role -> [bot_names]
              - missing_roles: list[str] — essential roles that are unfilled
              - acolytes: list[str] — bots that can cast Blessing/AGI
              - has_tank: bool
              - has_healer: bool
              - has_dps: bool
              - party_strength: str — strong / adequate / weak
              - recommendations: list[str] — suggestions for improvement
        """
        roles: dict[str, list[str]] = {}
        acolytes: list[str] = []
        online_bots = 0

        for bot_name, state in bot_states.items():
            role = state.role or "idle"
            if role not in roles:
                roles[role] = []
            roles[role].append(bot_name)
            online_bots += 1

            if state.acolyte_can_buff:
                acolytes.append(bot_name)

        # Check essential roles
        has_tank = "tank" in roles
        has_healer = "healer" in roles or "support" in roles
        has_dps = any(r in roles for r in ("dps_melee", "dps_ranged", "dps_magic"))

        missing: list[str] = []
        if not has_tank:
            missing.append("tank")
        if not has_healer:
            missing.append("healer")
        if not has_dps:
            missing.append("dps")

        if not has_healer:
            party_strength = "weak" if online_bots >= 3 else "adequate"
        elif has_tank and has_dps:
            party_strength = "strong"
        else:
            party_strength = "adequate"

        recommendations: list[str] = []
        if online_bots < 3:
            recommendations.append("Need more bots for full party efficiency")
        if not has_tank:
            recommendations.append("No tank — party lacks aggro management")
        if not has_healer:
            recommendations.append("No healer/support — no sustain in combat")
        if not has_dps:
            recommendations.append("No DPS — party can't kill efficiently")

        return {
            "roles": roles,
            "missing_roles": missing,
            "acolytes": acolytes,
            "has_tank": has_tank,
            "has_healer": has_healer,
            "has_dps": has_dps,
            "party_strength": party_strength,
            "recommendations": recommendations,
            "online_bots": online_bots,
        }

    # ── Utility ─────────────────────────────────────────────────

    def decide_migration(
        self,
        bot_states: dict[str, BotSwarmState],
        current_map: str,
        target_map: str,
        threshold: float | None = None,
    ) -> ConsensusResult:
        """Decide if/when the party should migrate to a new map.

        Migration happens when:
          - Leader decides a new hunt map
          - Majority of bots agree
          - Current map is exhausted (low spawn rate, too crowded)
        """
        threshold = threshold or self._default_threshold
        now = time.time()
        cooldown_key = f"migrate_{current_map}->{target_map}"

        cooldown = self._decision_cooldowns.get(cooldown_key, 0)
        if now - cooldown < 60:
            last = self._last_decisions.get(cooldown_key)
            if last:
                return last

        # Count how many bots are already on target map
        already_there = sum(1 for _, s in bot_states.items() if s.map_name == target_map)
        total = len(bot_states)
        agreement = already_there / max(1, total)
        should_migrate = agreement >= threshold or (total >= 2 and already_there >= total // 2)

        result = ConsensusResult(
            topic=cooldown_key,
            decision=should_migrate,
            threshold=threshold,
            agreement=agreement,
            total_votes=total,
            votes_for=already_there,
            votes_against=total - already_there,
            abstentions=0,
            reason=f"Migration {current_map} -> {target_map}: {already_there}/{total} already there ({agreement:.0%})",
            timestamp=now,
        )
        self._last_decisions[cooldown_key] = result
        self._decision_cooldowns[cooldown_key] = now
        return result
