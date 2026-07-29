"""Arena PvP tactics — threat scoring, target prioritisation, buff tracking.

Provides:
  - ThreatScore / ThreatProfile: multi-factor threat assessment
  - ArenaTactics: heuristic assessment for arena PvP maps
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = __import__("logging").getLogger(__name__)


# ── Threat scoring models ────────────────────────────────────────────────

@dataclass
class ThreatScore:
    """Normalised threat score for one opponent (0.0 → harmless, 1.0 → kill now)."""
    raw: float = 0.0
    normalised: float = 0.0
    is_healer: bool = False
    is_squishy: bool = False
    is_tank: bool = False
    is_priority: bool = False
    reasons: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"<ThreatScore:{self.normalised:.2f} "
            f"{'HEALER ' if self.is_healer else ''}"
            f"{'SQUISHY ' if self.is_squishy else ''}"
            f"{'TANK ' if self.is_tank else ''}"
            f"{'PRIORITY' if self.is_priority else ''}>"
        )


@dataclass
class ThreatProfile:
    """Aggregated threat assessment of a single player opponent."""
    name: str
    job_class: str
    hp_ratio: float         # 0.0–1.0 estimated remaining HP
    sp_ratio: float         # 0.0–1.0 estimated remaining SP
    base_level: int
    distance: float         # cells away
    is_attacking_us: bool   # currently targeting us
    visible_buffs: list[str] = field(default_factory=list)
    visible_debuffs: list[str] = field(default_factory=list)

    # Derived
    score: ThreatScore = field(default_factory=ThreatScore)

    @property
    def is_low_hp(self) -> bool:
        return self.hp_ratio < 0.35

    @property
    def is_low_sp(self) -> bool:
        return self.sp_ratio < 0.20

    @property
    def is_close(self) -> bool:
        return self.distance < 8


# ── Arena tactics engine ─────────────────────────────────────────────────

# Jobs considered "squishy" — low HP / low def, high damage
SQUISHY_JOBS: frozenset[str] = frozenset({
    "mage", "wizard", "high wizard", "warlock",
    "archer", "hunter", "sniper", "minstrel", "wanderer",
    "soul linker", "ninja", "kagerou", "oboro",
    "sorcerer", "warlock",
})

# Jobs that provide heavy healing or support
HEALER_JOBS: frozenset[str] = frozenset({
    "acolyte", "priest", "high priest", "arch bishop",
    "monk", "champion", "sura",
    "paladin", "royal guard",
})

# Jobs that are heavy tanks / defensive
TANK_JOBS: frozenset[str] = frozenset({
    "swordman", "knight", "lord knight", "rune knight",
    "crusader", "paladin", "royal guard",
})

# Critical-distance for melee vs ranged
MELEE_RANGE: int = 6
RANGED_RANGE: int = 14


def score_threat(profile: ThreatProfile, my_job: str) -> ThreatScore:
    """Compute a multi-factor threat score for *profile*.

    Priority order for target selection:
      1. Anyone attacking us (highest urgency)
      2. Low-HP squishies (easiest kills)
      3. Healers (force multiplier)
      4. Tanks (last — they take too long)
    """
    score = ThreatScore()
    reasons: list[str] = []

    # ── Base urgency from HP ──
    if profile.is_low_hp:
        score.raw += 20.0
        reasons.append("low HP")

    if profile.is_low_sp:
        score.raw += 5.0
        reasons.append("low SP")

    # ── Class-based profiles ──
    job_key = profile.job_class.lower()
    if job_key in SQUISHY_JOBS:
        score.is_squishy = True
        score.raw += 15.0
        reasons.append("squishy class")
    elif job_key in HEALER_JOBS:
        score.is_healer = True
        score.raw += 18.0
        reasons.append("healer/support — high priority")
    elif job_key in TANK_JOBS:
        score.is_tank = True
        score.raw += 5.0
        reasons.append("tank — low priority")

    # ── Attacking us ──
    if profile.is_attacking_us:
        score.raw += 25.0
        score.is_priority = True
        reasons.append("currently attacking us")

    # ── Distance factor ──
    my_is_melee = my_job.lower() in TANK_JOBS or my_job.lower() in {
        "thief", "assassin", "rogue", "stalker", "guillotine cross",
        "swordman", "knight", "lord knight", "rune knight",
        "monk", "champion", "sura",
    }
    if my_is_melee and profile.distance <= MELEE_RANGE:
        score.raw += 10.0  # In attack range
        reasons.append("in melee range")
    elif not my_is_melee and profile.distance <= RANGED_RANGE:
        score.raw += 8.0   # In attack range
        reasons.append("in ranged range")

    # ── Level advantage ──
    if profile.base_level >= 90:
        score.raw += 3.0
        reasons.append("high level")

    # ── Normalise to 0–1 range ──
    # Raw max ~85 (low HP + healer + attacking + close)
    score.normalised = min(1.0, score.raw / 70.0)
    score.reasons = reasons

    return score


# ── Notable buffs and debuffs ────────────────────────────────────────────

KILLER_BUFFS: frozenset[str] = frozenset({
    "energy coat", "safe wall", "pneuma",
    "kyrie eleison", "assumptio", "reflect shield",
    "defender", "autoberserk", "concentration",
    "true sight", "wind walk", "sword mastery",
})

DANGEROUS_BUFFS: frozenset[str] = frozenset({
    "magnificat", "gloria", "impositio manus",
    "blessing", "increase agility", "kaahi",
    "kaina", "kaizel",
})

PRIORITY_DEBUFFS: frozenset[str] = frozenset({
    "lex aeterna", "lex divina", "basilica",
    "spell breaker", "magic rod",
})


def has_offensive_buff(buffs: list[str]) -> bool:
    """Check if the target has any notable offensive buff active."""
    return any(b.lower() in KILLER_BUFFS for b in buffs)


def has_defensive_buff(buffs: list[str]) -> bool:
    """Check if the target has any notable defensive buff active."""
    return any(b.lower() in DANGEROUS_BUFFS for b in buffs)


# ── ArenaTactics domain helper ───────────────────────────────────────────

class ArenaTactics:
    """Arena PvP threat assessment and target prioritisation.

    This is used by PvPDomain.assess() for non-WoE PvP maps.
    """

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Evaluate arena PvP state and queue actions."""
        players = signals.get("players", []) or []
        if not players:
            return

        my_name = str(signals.get("name", "") or "")
        my_job = str(signals.get("job_name", "novice") or "novice")
        my_hp = int(signals.get("hp", 1) or 1)
        my_max_hp = int(signals.get("max_hp", 100) or 100)
        hp_ratio = my_hp / max(my_max_hp, 1)

        # Build threat profiles for each visible opponent
        profiles: list[ThreatProfile] = []
        for p in players:
            if isinstance(p, dict):
                p_name = str(p.get("name", "") or "")
            else:
                p_name = str(p)
            if not p_name or p_name == my_name:
                continue

            profile = self._build_profile(p, my_name)
            if profile is None:
                continue
            profile.score = score_threat(profile, my_job)
            profiles.append(profile)

        if not profiles:
            return

        # Sort by threat (highest first)
        profiles.sort(key=lambda pr: pr.score.normalised, reverse=True)

        top = profiles[0]
        logger.info(
            "[Arena] %s: top threat %s (score=%.2f, %s)",
            bot_id, top.name, top.score.normalised, top.score.reasons,
        )

        # ── Emergency: low HP retreat ──
        if hp_ratio < 0.25:
            self._emit_retreat(actions, bot_id, profiles)
            return

        # ── Attack highest-threat target ──
        self._emit_attack(actions, bot_id, top)

        # ── Track visible buffs on high-threat targets ──
        if top.score.normalised > 0.5 and top.visible_buffs:
            self._emit_buff_awareness(actions, top)

    # ------------------------------------------------------------------
    def _build_profile(
        self,
        player: Any,
        my_name: str,
    ) -> ThreatProfile | None:
        """Extract a ThreatProfile from a player signal dict."""
        if not isinstance(player, dict):
            # Fallback: just the name
            return ThreatProfile(
                name=str(player),
                job_class="unknown",
                hp_ratio=1.0,
                sp_ratio=1.0,
                base_level=1,
                distance=99.0,
                is_attacking_us=False,
            )
        p_name = str(player.get("name", "") or "")
        if not p_name or p_name == my_name:
            return None

        return ThreatProfile(
            name=p_name,
            job_class=str(player.get("job_name", "unknown") or "unknown"),
            hp_ratio=float(player.get("hp_ratio", 1.0) or 1.0),
            sp_ratio=float(player.get("sp_ratio", 1.0) or 1.0),
            base_level=int(player.get("base_level", 1) or 1),
            distance=float(player.get("distance", 99.0) or 99.0),
            is_attacking_us=bool(player.get("targeting_me", False)),
            visible_buffs=[str(b) for b in (player.get("buffs", []) or [])],
            visible_debuffs=[str(d) for d in (player.get("debuffs", []) or [])],
        )

    # ------------------------------------------------------------------
    def _emit_attack(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        target: ThreatProfile,
    ) -> None:
        """Emit an attack command for *target*."""
        reason = (
            f"Arena PvP: attacking {target.name} "
            f"({target.job_class}, HP={target.hp_ratio:.0%}, "
            f"score={target.score.normalised:.2f}) — {target.score.reasons}"
        )
        actions.append(HeuristicAction(
            kind="command",
            command=f"attack {target.name}",
            confidence=0.92,
            domain="pvp",
            reason=reason,
            metadata={
                "target": target.name,
                "job": target.job_class,
                "threat": target.score.normalised,
                "mode": "arena",
            },
        ))

    # ------------------------------------------------------------------
    def _emit_retreat(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        threats: list[ThreatProfile],
    ) -> None:
        """Low HP retreat — use Fly Wing or move to save zone."""
        logger.info("[Arena] %s: retreating at low HP, %d threats", bot_id, len(threats))
        actions.append(HeuristicAction(
            kind="command",
            command="use 601",  # Fly Wing
            confidence=0.99,
            domain="pvp",
            reason=f"Arena retreat: HP critical with {len(threats)} opponents",
        ))

    # ------------------------------------------------------------------
    def _emit_buff_awareness(
        self,
        actions: list[HeuristicAction],
        target: ThreatProfile,
    ) -> None:
        """Log awareness of dangerous buffs on a high-threat target."""
        buffs_str = ", ".join(target.visible_buffs)
        actions.append(HeuristicAction(
            kind="log",
            command="",
            confidence=1.0,
            domain="pvp",
            reason=f"[Arena buff watch] {target.name} has: {buffs_str}",
            metadata={"target": target.name, "buffs": target.visible_buffs},
        ))
