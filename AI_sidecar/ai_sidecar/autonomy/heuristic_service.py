from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HeuristicAction:
    kind: str  # "command" | "macro" | "reflex_override"
    command: str
    confidence: float
    domain: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeuristicAssessment:
    horizon: str
    actions: list[HeuristicAction]
    confidence: float
    actionable: bool
    top_domain: str
    signals: dict[str, Any] = field(default_factory=dict)


class HeuristicService:
    """Produces heuristic actions from game state signals without calling LLM.
    
    Maps the existing decision_service opportunistic signals to executable actions.
    The confidence score determines whether the PDCA loop skips the LLM entirely.
    """

    def __init__(self):
        self._last_assessment: dict[str, HeuristicAssessment] = {}
        self._domain_weights: dict[str, float] = {
            "recovery": 0.15,
            "grind": 0.30,
            "economy": 0.25,
            "quest": 0.10,
            "exploration": 0.20,
        }

    def set_domain_weights(self, weights: dict[str, float]) -> None:
        self._domain_weights.update(weights)

    def assess(self, signals: dict[str, Any]) -> HeuristicAssessment:
        actions: list[HeuristicAction] = []
        total_confidence = 0.0
        weighted_domains: dict[str, float] = {}

        # Enriched state signals (emergent discovery)
        # These are optional — the assess() always has fallbacks for missing keys
        # New signals can be added here without changing behavior profiles
        _enriched = signals.get("_enriched", None)
        
        # Check recovery signal
        if signals.get("hp_ratio", 1.0) < 0.5:
            hp = signals["hp_ratio"]
            if hp < 0.2:
                actions.append(HeuristicAction(
                    kind="reflex_override", command="emergency_heal",
                    confidence=0.95, domain="recovery",
                    reason=f"Critical HP ({hp:.0%})",
                ))
                weighted_domains["recovery"] = 0.95
                total_confidence = max(total_confidence, 0.95)
            elif hp < 0.5:
                actions.append(HeuristicAction(
                    kind="command", command="sit",
                    confidence=0.75, domain="recovery",
                    reason=f"Low HP ({hp:.0%})",
                ))
                weighted_domains["recovery"] = 0.75
                total_confidence = max(total_confidence, 0.75)

        # Check combat/aggro signal
        hostiles = signals.get("nearby_hostiles", 0)
        if hostiles > 0:
            if hostiles <= 3:
                actions.append(HeuristicAction(
                    kind="command", command="attack",
                    confidence=0.65, domain="grind",
                    reason=f"{hostiles} nearby hostiles (manageable)",
                ))
                weighted_domains["grind"] = 0.65
                total_confidence = max(total_confidence, 0.65)
            else:
                actions.append(HeuristicAction(
                    kind="command", command="flee",
                    confidence=0.85, domain="recovery",
                    reason=f"{hostiles} nearby hostiles (overwhelming)",
                ))
                weighted_domains["recovery"] = max(weighted_domains.get("recovery", 0), 0.85)
                total_confidence = max(total_confidence, 0.85)

        # Check known map signal
        if signals.get("map_known", False):
            actions.append(HeuristicAction(
                kind="command", command="auto",
                confidence=0.55, domain="grind",
                reason="Known map, resuming auto mode",
            ))
            weighted_domains["grind"] = max(weighted_domains.get("grind", 0), 0.55)
            total_confidence = max(total_confidence, 0.55)

        # Check weight/encumbrance
        weight_ratio = signals.get("weight_ratio", 0)
        if weight_ratio and weight_ratio > 0.8:
            actions.append(HeuristicAction(
                kind="command", command="return_to_vendor",
                confidence=0.70, domain="economy",
                reason=f"Near encumbered ({weight_ratio:.0%})",
            ))
            weighted_domains["economy"] = max(weighted_domains.get("economy", 0), 0.70)
            total_confidence = max(total_confidence, 0.70)

        # Check recent death
        if signals.get("recent_death", False):
            actions.append(HeuristicAction(
                kind="command", command="respawn_and_recover",
                confidence=0.90, domain="recovery",
                reason="Recent death detected",
            ))
            weighted_domains["recovery"] = max(weighted_domains.get("recovery", 0), 0.90)
            total_confidence = max(total_confidence, 0.90)

        # Determine top domain
        top_domain = "none"
        if weighted_domains:
            top_domain = str(max(weighted_domains, key=lambda k: float(weighted_domains.get(k, 0.0))))

        assessment = HeuristicAssessment(
            horizon=signals.get("horizon", "short_term"),
            actions=actions,
            confidence=total_confidence,
            actionable=len(actions) > 0,
            top_domain=top_domain,
            signals=dict(signals),
        )
        bot_id = signals.get("bot_id", "default")
        if not hasattr(self, '_last_assessment'):
            object.__setattr__(self, '_last_assessment', {})
        self._last_assessment[bot_id] = assessment
        return assessment

    def confidence_for(self, horizon: str, signals: dict | None = None, bot_id: str = "default") -> float:
        """Called by PDCA loop to check if heuristic can replace LLM for this horizon.
        
        Returns the confidence from the last assessment for this bot_id.
        """
        if signals is not None:
            sigs = dict(signals)
            sigs.setdefault("bot_id", bot_id)
            result = self.assess(sigs)
            return result.confidence
        last = self._last_assessment.get(bot_id) if hasattr(self, '_last_assessment') else None
        if last is not None:
            return last.confidence
        return 0.0
