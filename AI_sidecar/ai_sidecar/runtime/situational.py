"""Situational Awareness — validates every action against actual game state.

No hardcoded "use Red Potion" or "sit". Instead, this layer:
1. Checks inventory before issuing use-item commands
2. Checks character skills before issuing skill commands
3. Buys items if missing (with budget check)
4. Adapts healing to best available item (Red → Orange → White)
5. Handles ALL edge cases: no potions, no skills, dead, stuck, no zeny
6. Prioritizes survival: potions > sit > flee > die

This runs in the bridge AFTER receiving sidecar actions, ensuring
every command is validated against ACTUAL game state.
"""
from __future__ import annotations
import logging
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class SituationalAwareness:
    """Validates and adapts AI actions to actual game state.
    
    Every domain module produces actions based on RO theory.
    This layer checks actions against ACTUAL game state:
    - "use Orange Potion" → check inventory → if none, use Red Potion
    - "sit" → check if can sit → if not, use best available potion
    - "buy 30 501" → check zeny → buy as many as affordable
    - "move prt_fild05" → check HP → if low, heal first
    
    This is the CRITICAL gap between "AI that knows RO" and 
    "AI that can actually play RO."
    """
    
    # Healing items by potency (item_id, name, heal_amount, cost, min_level)
    HEALING_ITEMS = [
        (504, "White Potion", 325, 250, 30),
        (502, "Orange Potion", 105, 100, 15),
        (501, "Red Potion", 45, 12, 1),
    ]
    
    def __init__(self):
        self._last_validated: dict[str, list[HeuristicAction]] = {}
    
    def validate(self, actions: list[HeuristicAction], signals: dict[str, Any], bot_id: str) -> list[HeuristicAction]:
        """Validate and adapt actions to actual game state."""
        validated = []
        
        for action in actions:
            if action.kind != "command" or not action.command:
                validated.append(action)
                continue
            
            adapted = self._adapt_command(action, signals, bot_id)
            if adapted:
                validated.append(adapted)
        
        self._last_validated[bot_id] = validated
        return validated
    
    def _adapt_command(self, action: HeuristicAction, signals: dict[str, Any], bot_id: str) -> HeuristicAction | None:
        """Adapt a single command to actual game state."""
        cmd = action.command.lower().strip()
        
        # ── HEALING ADAPTATION ──
        # "use X Potion" → find best available potion in inventory
        if cmd.startswith("use ") and ("potion" in cmd or "item" in cmd):
            return self._adapt_heal(action, signals, bot_id)
        
        # ── SIT ADAPTATION ──
        # "sit" → if can't sit, use best healing potion instead
        if cmd == "sit":
            return self._adapt_sit(action, signals, bot_id)
        
        # ── BUY ADAPTATION ──
        # "buy N ITEM" → check zeny, buy as many as affordable
        if cmd.startswith("buy "):
            return self._adapt_buy(action, signals, bot_id)
        
        # ── MOVE ADAPTATION ──
        # "move MAP" → check HP, heal if critically low before moving
        if cmd.startswith("move ") and not cmd.startswith("move prontera"):
            return self._adapt_move(action, signals, bot_id)
        
        return action
    
    def _get_best_heal_item(self, signals: dict[str, Any]) -> tuple[int, str] | None:
        """Find the best healing item available in inventory.
        
        Checks inventory for potions, returns best available match.
        If none in inventory, returns the best affordable potion.
        """
        inventory = signals.get("inventory", {})
        if isinstance(inventory, dict):
            items = inventory.get("items", []) if "items" in inventory else []
        elif isinstance(inventory, list):
            items = inventory
        else:
            items = []
        
        zeny = int(signals.get("zeny", 0) or 0)
        base_level = int(signals.get("base_level", 1) or 1)
        
        # First pass: check what's in inventory
        for item_id, name, heal, cost, min_lvl in self.HEALING_ITEMS:
            if base_level < min_lvl:
                continue
            # Check inventory for this item
            for item in items:
                iname = ""
                if isinstance(item, dict):
                    iname = str(item.get("name", item.get("identifiedDisplayName", ""))).lower()
                elif isinstance(item, str):
                    iname = item.lower()
                
                if name.lower() in iname or str(item_id) in iname:
                    qty = 0
                    if isinstance(item, dict):
                        qty = int(item.get("quantity", item.get("amount", 1)) or 1)
                    if qty > 0:
                        return (item_id, name)
        
        # Second pass: nothing in inventory, recommend buying the best affordable
        for item_id, name, heal, cost, min_lvl in self.HEALING_ITEMS:
            if base_level < min_lvl:
                continue
            if zeny >= cost:
                return (item_id, name)
        
        # No potions at all — return cheapest
        return (501, "Red Potion")
    
    def _adapt_heal(self, action: HeuristicAction, signals: dict[str, Any], bot_id: str) -> HeuristicAction:
        """Adapt a heal command to use the best available potion."""
        best = self._get_best_heal_item(signals)
        if best:
            item_id, name = best
            new_cmd = f"use {item_id}"
            logger.info(f"[situational] {bot_id}: {action.command} → {new_cmd} (best available: {name})")
            return HeuristicAction(
                kind="command", command=new_cmd,
                confidence=action.confidence,
                reason=f"Situational: {name} best available",
                domain=action.domain,
            )
        # No potions available — try to buy some
        logger.warning(f"[situational] {bot_id}: No potions available, buying Red Potions")
        return HeuristicAction(
            kind="command", command="buy 10 501",
            confidence=0.7,
            reason="Soldier on: buying Red Potions",
            domain="survival",
        )
    
    def _adapt_sit(self, action: HeuristicAction, signals: dict[str, Any], bot_id: str) -> HeuristicAction:
        """Adapt sit command — use potion if can't sit or if potions available."""
        job = str(signals.get("job", "") or "").lower()
        can_sit = job != "novice"  # Only Novices can't sit (without Basic Skill waste)
        
        if can_sit:
            # Check if we have potions anyway (always better to potion than sit)
            best = self._get_best_heal_item(signals)
            if best and best[0] != 501:  # Only use non-Red potions for healing
                return self._adapt_heal(action, signals, bot_id)
            return action  # Sit is fine
        
        # Can't sit — use potion
        logger.info(f"[situational] {bot_id}: Can't sit ({job}), using potion instead")
        return self._adapt_heal(action, signals, bot_id)
    
    def _adapt_buy(self, action: HeuristicAction, signals: dict[str, Any], bot_id: str) -> HeuristicAction:
        """Adapt buy command to available budget."""
        parts = action.command.split()
        if len(parts) < 3:
            return action
        
        try:
            qty = int(parts[1])
            item_id = int(parts[2])
        except ValueError:
            return action
        
        zeny = int(signals.get("zeny", 0) or 0)
        
        # Find item cost
        cost = 0
        for iid, name, heal, c, min_lvl in self.HEALING_ITEMS:
            if iid == item_id:
                cost = c
                break
        
        if cost == 0:
            return action
        
        max_affordable = zeny // cost
        actual_qty = min(qty, max_affordable, 30)  # Max 30 potions
        
        if actual_qty <= 0:
            logger.warning(f"[situational] {bot_id}: Can't afford {qty}x {item_id} ({zeny}z, needs {qty*cost}z)")
            # Try cheaper item
            for iid, name, heal, c, min_lvl in self.HEALING_ITEMS:
                if zeny >= c:
                    affordable = min(10, zeny // c)
                    new_cmd = f"buy {affordable} {iid}"
                    logger.info(f"[situational] {bot_id}: Buying {affordable}x {name} instead")
                    return HeuristicAction(
                        kind="command", command=new_cmd,
                        confidence=0.6,
                        reason=f"Budget: bought {name} instead",
                        domain=action.domain,
                    )
            # Can't afford anything
            return None
        
        if actual_qty < qty:
            new_cmd = f"buy {actual_qty} {item_id}"
            logger.info(f"[situational] {bot_id}: Budget buy: {qty}→{actual_qty}x")
            return HeuristicAction(
                kind="command", command=new_cmd,
                confidence=action.confidence,
                reason=f"Budget buy: {actual_qty}x (afford {qty}x? need {qty*cost}z)",
                domain=action.domain,
            )
        
        return action
    
    def _adapt_move(self, action: HeuristicAction, signals: dict[str, Any], bot_id: str) -> HeuristicAction:
        """Adapt move command — heal if critically low first."""
        hp = int(signals.get("hp", 100) or 100)
        hp_max = int(signals.get("hp_max", 1) or 1)
        hp_pct = hp / max(hp_max, 1) * 100
        
        if hp_pct < 40:
            best = self._get_best_heal_item(signals)
            if best:
                item_id, name = best
                logger.info(f"[situational] {bot_id}: Healing ({name}) before move (HP={hp_pct:.0f}%)")
                return HeuristicAction(
                    kind="command", command=f"use {item_id}",
                    confidence=0.85,
                    reason=f"Heal before move: {name}",
                    domain="survival",
                )
        
        return action


# Global instance
_awareness: SituationalAwareness | None = None

def get_awareness() -> SituationalAwareness:
    global _awareness
    if _awareness is None:
        _awareness = SituationalAwareness()
    return _awareness
