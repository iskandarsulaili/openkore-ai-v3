"""
Breakpoint Gear Scorer — evaluates equipment by how well it hits stat breakpoints.

Determines whether swapping a piece of gear will push key stats over important
thresholds in pre-renewal Ragnarok Online:

  Breakpoint         |  Target  |  Effect
  -------------------+----------+---------------------------------
  DEX                |  150     |  Instant cast (var cast → 0)
  STR (every 10)     |  10/20…  |  STR bonus ATK stair-step
  ASPD               |  190     |  Max attack speed (2 attacks/sec)
  VIT                |  100     |  Soft/hard DEF threshold, HP

Uses damage_formulas.py for cast time & ASPD calculations, and gear_swapper's
GearSet conventions for item representation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ai_sidecar.combat.damage_formulas import calculate_aspd_interval

logger = logging.getLogger(__name__)

# ── Pre-defined breakpoints ────────────────────────────────────────

BREAKPOINTS: dict[str, dict[str, Any]] = {
    "dex": {
        "breakpoint": 150,
        "label": "Instant Cast (DEX 150)",
        "effect": "Variable cast time reduced to 0.",
        "priority": 100,
        "interval": None,
    },
    "str": {
        "breakpoint": None,
        "label": "STR Bonus Breakpoint (every 10)",
        "effect": "+1 STR bonus ATK per 10 STR.",
        "priority": 80,
        "interval": 10,
    },
    "aspd": {
        "breakpoint": 190,
        "label": "Max ASPD (190)",
        "effect": "2 attacks/sec — minimum attack interval.",
        "priority": 90,
        "interval": None,
    },
    "vit": {
        "breakpoint": 100,
        "label": "VIT DEF Threshold (100)",
        "effect": "Soft/hard DEF bonus, HP scaling.",
        "priority": 70,
        "interval": None,
    },
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "dex": 1.0,
    "str": 0.8,
    "aspd": 0.9,
    "vit": 0.7,
}


@dataclass
class StatContribution:
    """How a piece of gear contributes to each stat."""
    str_: int = 0
    agi: int = 0
    vit: int = 0
    int_: int = 0
    dex: int = 0
    luk: int = 0
    atk: int = 0
    matk: int = 0
    aspd_bonus: int = 0


@dataclass
class GearItem:
    """A piece of equipment in the inventory, compatible with bridge snapshot format."""
    name: str
    slot_type: str
    refine: int = 0
    slots: int = 0
    cards: list[str] = field(default_factory=list)
    item_id: int = 0
    stats: StatContribution = field(default_factory=StatContribution)
    equipped: bool = False


@dataclass
class BreakpointGap:
    """Distance from current stat to the next breakpoint."""
    stat_name: str
    current_value: int
    next_breakpoint: int | None
    gap: int
    is_met: bool
    effective_percent: float
    description: str


@dataclass
class ItemScore:
    """Score for a single item against current stats and breakpoints."""
    item: GearItem
    total_score: float
    breakpoint_scores: dict[str, float]
    gap_changes: dict[str, int]
    recommendation: str


@dataclass
class UpgradeRecommendation:
    """Recommended gear swap with before/after comparison."""
    current_item: GearItem | None
    proposed_item: GearItem
    score_before: float
    score_after: float
    stat_deltas: dict[str, int]
    breakpoints_improved: list[str]
    breakpoints_harmed: list[str]
    reason: str


_DEFAULT_STATS = {
    "str": 1,
    "agi": 1,
    "vit": 1,
    "int": 1,
    "dex": 1,
    "luk": 1,
}


def _compute_stat_breakpoint_distance(
    stat_name: str,
    current_stat: int,
    gear_bonus: int = 0,
) -> BreakpointGap:
    """Compute distance to the next meaningful breakpoint for a stat."""
    bp_info = BREAKPOINTS.get(stat_name, {})
    bp_threshold = bp_info.get("breakpoint")
    interval = bp_info.get("interval")
    effective = current_stat + gear_bonus

    if interval:
        next_bp = ((effective // interval) + 1) * interval
        gap = next_bp - effective if effective > 0 else next_bp
        is_met = effective >= 1 and effective % interval == 0
        effective_percent = 1.0 - (gap / interval) if interval > 0 else 1.0
        if is_met or effective == 0:
            effective_percent = 1.0 if is_met else 0.0
        desc = f"{stat_name.upper()} {effective}: next breakpoint at {next_bp} (gap={gap})"
    elif bp_threshold:
        next_bp = bp_threshold
        gap = max(0, next_bp - effective)
        is_met = effective >= next_bp
        effective_percent = min(1.0, effective / next_bp) if next_bp else 1.0
        if is_met:
            desc = f"{stat_name.upper()} {effective} \u2265 {next_bp} \u2713"
        else:
            desc = f"{stat_name.upper()} {effective}/{next_bp}: need {gap} more"
    else:
        next_bp = None
        gap = 0
        is_met = True
        effective_percent = 1.0
        desc = f"{stat_name.upper()} {effective}: no breakpoint defined"

    return BreakpointGap(
        stat_name=stat_name,
        current_value=effective,
        next_breakpoint=next_bp,
        gap=gap,
        is_met=is_met,
        effective_percent=min(1.0, effective_percent),
        description=desc,
    )


def _str_bonus_atk(str_stat: int) -> int:
    """STR bonus ATK in pre-renewal RO: (STR / 10)² + extra at STR thresholds."""
    return str_stat + ((str_stat // 10) ** 2)


def _compute_aspd_breakpoints(aspd: int) -> dict[str, Any]:
    """Compute attack interval and attacks-per-second from ASPD."""
    interval = calculate_aspd_interval(aspd)
    attacks_per_sec = 1.0 / interval if interval > 0 else 0
    return {
        "aspd": aspd,
        "interval_s": round(interval, 3),
        "attacks_per_sec": round(attacks_per_sec, 2),
        "at_max": aspd >= 190,
    }


class GearScorer:
    """Evaluates equipment by stat breakpoint contribution.

    Compares current character stats against pre-defined RO breakpoints and
    scores each item by how much it moves toward thresholds.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    # ── Public API ─────────────────────────────────────────────────

    def score_item(
        self,
        item: GearItem,
        current_stats: dict[str, int],
        current_aspd: int = 140,
    ) -> ItemScore:
        """Score a single gear item against current stats.

        Args:
            item: The gear item to evaluate.
            current_stats: Dict of base stats (str, agi, vit, int, dex, luk).
            current_aspd: Current ASPD value (100-190).

        Returns:
            ItemScore with total_score, per-breakpoint breakdown, and recommendation.
        """
        stats = self._norm_stats(current_stats)
        contrib = item.stats

        bp_scores: dict[str, float] = {}
        gaps_before: dict[str, int] = {}
        gaps_after: dict[str, int] = {}

        # DEX
        dex_before = stats.get("dex", 0)
        dex_after = dex_before + contrib.dex
        gap_before = _compute_stat_breakpoint_distance("dex", dex_before)
        gap_after = _compute_stat_breakpoint_distance("dex", dex_after)
        gaps_before["dex"] = gap_before.gap
        gaps_after["dex"] = gap_after.gap
        bp_scores["dex"] = self._score_gap_change(gap_before, gap_after, "dex")

        # STR
        str_before = stats.get("str", 0)
        str_after = str_before + contrib.str_
        gap_before = _compute_stat_breakpoint_distance("str", str_before)
        gap_after = _compute_stat_breakpoint_distance("str", str_after)
        gaps_before["str"] = gap_before.gap
        gaps_after["str"] = gap_after.gap
        bp_scores["str"] = self._score_gap_change(gap_before, gap_after, "str")

        # VIT
        vit_before = stats.get("vit", 0)
        vit_after = vit_before + contrib.vit
        gap_before = _compute_stat_breakpoint_distance("vit", vit_before)
        gap_after = _compute_stat_breakpoint_distance("vit", vit_after)
        gaps_before["vit"] = gap_before.gap
        gaps_after["vit"] = gap_after.gap
        bp_scores["vit"] = self._score_gap_change(gap_before, gap_after, "vit")

        # ASPD
        aspd_effective = current_aspd + contrib.aspd_bonus
        gap_before = _compute_stat_breakpoint_distance("aspd", current_aspd)
        gap_after = _compute_stat_breakpoint_distance("aspd", aspd_effective)
        gaps_before["aspd"] = gap_before.gap
        gaps_after["aspd"] = gap_after.gap
        bp_scores["aspd"] = self._score_gap_change(gap_before, gap_after, "aspd")

        # ATK/MATK direct bonus
        atk_score = contrib.atk * 0.5 + contrib.matk * 0.3
        bp_scores["atk"] = atk_score

        total = sum(
            bp_scores.get(k, 0.0) * self.weights.get(k, 1.0)
            for k in ("dex", "str", "vit", "aspd", "atk")
        )

        rec_parts = []
        for bp_name in ("dex", "str", "vit", "aspd"):
            bg = gaps_before.get(bp_name, 0)
            ag = gaps_after.get(bp_name, 0)
            if bg != ag:
                delta = bg - ag
                if delta > 0:
                    rec_parts.append(f"{bp_name.upper()}: closes gap by {delta}")
                else:
                    rec_parts.append(f"{bp_name.upper()}: widens gap by {abs(delta)}")
        if contrib.atk:
            rec_parts.append(f"+{contrib.atk} ATK")
        if contrib.matk:
            rec_parts.append(f"+{contrib.matk} MATK")
        recommendation = "; ".join(rec_parts) if rec_parts else "No breakpoint effect"

        return ItemScore(
            item=item,
            total_score=round(total, 2),
            breakpoint_scores=bp_scores,
            gap_changes=gaps_after,
            recommendation=recommendation,
        )

    def best_upgrade(
        self,
        current_item: GearItem | None,
        inventory: list[GearItem],
        current_stats: dict[str, int],
        current_aspd: int = 140,
    ) -> UpgradeRecommendation | None:
        """Find the best gear upgrade from inventory for a given slot.

        Compares every item in *inventory* against the current stats and
        returns the one that yields the highest breakpoint score.

        Args:
            current_item: The currently equipped item (or None if slot is empty).
            inventory: List of candidate GearItems to consider.
            current_stats: Current base stats.
            current_aspd: Current ASPD value.

        Returns:
            An UpgradeRecommendation, or None if inventory is empty.
        """
        if not inventory:
            return None

        before_score = 0.0
        if current_item:
            before = self.score_item(current_item, current_stats, current_aspd)
            before_score = before.total_score

        best_candidate: GearItem | None = None
        best_score = -1.0
        best_item_score: ItemScore | None = None

        for candidate in inventory:
            if current_item and candidate.name == current_item.name:
                continue
            scored = self.score_item(candidate, current_stats, current_aspd)
            if scored.total_score > best_score:
                best_score = scored.total_score
                best_candidate = candidate
                best_item_score = scored

        if best_candidate is None or best_item_score is None:
            return None

        stat_deltas: dict[str, int] = {}
        if current_item:
            cur_c = current_item.stats
            new_c = best_candidate.stats
            stat_deltas["str"] = new_c.str_ - cur_c.str_
            stat_deltas["dex"] = new_c.dex - cur_c.dex
            stat_deltas["vit"] = new_c.vit - cur_c.vit
            stat_deltas["agi"] = new_c.agi - cur_c.agi
            stat_deltas["int"] = new_c.int_ - cur_c.int_
            stat_deltas["luk"] = new_c.luk - cur_c.luk
            stat_deltas["aspd"] = new_c.aspd_bonus - cur_c.aspd_bonus
        else:
            stat_deltas = {
                "str": best_candidate.stats.str_,
                "dex": best_candidate.stats.dex,
                "vit": best_candidate.stats.vit,
                "agi": best_candidate.stats.agi,
                "int": best_candidate.stats.int_,
                "luk": best_candidate.stats.luk,
                "aspd": best_candidate.stats.aspd_bonus,
            }

        improved = [
            k for k in ("dex", "str", "vit", "aspd")
            if best_item_score.breakpoint_scores.get(k, 0) > 0
        ]
        harmed = [
            k for k in ("dex", "str", "vit", "aspd")
            if best_item_score.breakpoint_scores.get(k, 0) < 0
        ]

        delta_parts = []
        for s, v in stat_deltas.items():
            if v > 0:
                delta_parts.append(f"{s.upper()}+{v}")
            elif v < 0:
                delta_parts.append(f"{s.upper()}{v}")
        delta_str = ", ".join(delta_parts) if delta_parts else "no stat change"
        reason = (
            f"{best_candidate.name} (score {best_score:.1f}) "
            f"vs {'current' if current_item else 'empty'} ({before_score:.1f}) "
            f"\u2014 \u0394: {delta_str}. "
            f"Improved: {', '.join(improved) if improved else 'none'}. "
            f"{best_item_score.recommendation}"
        )

        return UpgradeRecommendation(
            current_item=current_item,
            proposed_item=best_candidate,
            score_before=round(before_score, 2),
            score_after=round(best_score, 2),
            stat_deltas=stat_deltas,
            breakpoints_improved=improved,
            breakpoints_harmed=harmed,
            reason=reason,
        )

    def breakpoint_gap(
        self,
        current_stats: dict[str, int],
        current_aspd: int = 140,
    ) -> list[BreakpointGap]:
        """Analyse every breakpoint and return the distance to each threshold.

        Args:
            current_stats: Dict of base stats.
            current_aspd: Current ASPD value.

        Returns:
            List of BreakpointGap sorted by gap size (smallest first).
        """
        stats = self._norm_stats(current_stats)

        results: list[BreakpointGap] = []
        for stat_name in ("dex", "str", "vit"):
            val = stats.get(stat_name, 0)
            results.append(_compute_stat_breakpoint_distance(stat_name, val))

        aspd_gap = _compute_stat_breakpoint_distance("aspd", current_aspd)
        results.append(aspd_gap)

        results.sort(key=lambda g: g.gap if g.next_breakpoint else 9999)
        return results

    def evaluate_build(
        self,
        stats: dict[str, int],
        aspd: int,
        gear: list[GearItem],
    ) -> dict[str, Any]:
        """Full build breakpoint audit.

        Args:
            stats: Base stats.
            aspd: Current ASPD.
            gear: All equipped gear items.

        Returns:
            Dict with overall_score, gaps, aspd_analysis, str_bonus_atk,
            gear_scores.
        """
        gaps = self.breakpoint_gap(stats, aspd)
        aspd_info = _compute_aspd_breakpoints(aspd)
        str_atk = _str_bonus_atk(stats.get("str", 0))
        gear_scores = [self.score_item(g, stats, aspd) for g in gear]

        stat_scores: dict[str, float] = {}
        for g in gaps:
            if g.stat_name in self.weights:
                stat_scores[g.stat_name] = g.effective_percent * self.weights[g.stat_name]
        overall = (
            sum(stat_scores.values()) / sum(self.weights.get(k, 1.0) for k in stat_scores)
            if stat_scores else 0.0
        )

        return {
            "overall_score": round(overall * 100, 1),
            "gaps": gaps,
            "aspd_analysis": aspd_info,
            "str_bonus_atk": str_atk,
            "gear_scores": gear_scores,
        }

    def make_gear_item(
        self,
        name: str,
        slot_type: str,
        *,
        refine: int = 0,
        slots: int = 0,
        cards: list[str] | None = None,
        item_id: int = 0,
        str_: int = 0,
        agi: int = 0,
        vit: int = 0,
        int_: int = 0,
        dex: int = 0,
        luk: int = 0,
        atk: int = 0,
        matk: int = 0,
        aspd_bonus: int = 0,
        equipped: bool = False,
    ) -> GearItem:
        """Convenience factory to create a GearItem with stat contributions."""
        return GearItem(
            name=name,
            slot_type=slot_type,
            refine=refine,
            slots=slots,
            cards=cards or [],
            item_id=item_id,
            stats=StatContribution(
                str_=str_,
                agi=agi,
                vit=vit,
                int_=int_,
                dex=dex,
                luk=luk,
                atk=atk,
                matk=matk,
                aspd_bonus=aspd_bonus,
            ),
            equipped=equipped,
        )

    # ── Internal helpers ──────────────────────────────────────────

    @staticmethod
    def _norm_stats(stats: dict[str, int]) -> dict[str, int]:
        """Normalise stat dict so both ``int`` and ``int_`` keys work, fill defaults."""
        norm: dict[str, int] = dict(_DEFAULT_STATS)
        for k, v in stats.items():
            key = k.rstrip("_")
            if key in norm:
                norm[key] = v
        return norm

    def _score_gap_change(
        self,
        gap_before: BreakpointGap,
        gap_after: BreakpointGap,
        stat_name: str,
    ) -> float:
        """Score the improvement (or regression) from before to after.

        Returns a signed float: positive when closing toward a breakpoint,
        negative when widening the gap.
        """
        weight = self.weights.get(stat_name, 1.0)

        if gap_before.is_met and gap_after.is_met:
            return 0.0

        if not gap_before.is_met and gap_after.is_met:
            return 20.0 * weight

        gap_reduction = gap_before.gap - gap_after.gap
        if gap_reduction > 0:
            return gap_reduction * weight * 2.0
        if gap_reduction < 0:
            return gap_reduction * weight * 2.0

        return 0.0


# ── Global Singleton ──────────────────────────────────────────────

_scorer: GearScorer | None = None


def get_gear_scorer() -> GearScorer:
    """Get the global GearScorer singleton."""
    global _scorer
    if _scorer is None:
        _scorer = GearScorer()
    return _scorer
