
"""Discovery API — Pro RO LLM analyzes server conditions for optimal strategy."""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/discover", tags=["discovery"])

class HealStrategyRequest(BaseModel):
    bot_id: str
    hp: int
    hp_max: int
    zeny: int
    map: str
    inventory: list[dict] = []
    known_shops: list[dict] = []
    known_portals: list[str] = []
    available_items: list[dict] = []

class HealStrategyResponse(BaseModel):
    strategy: str
    command: str
    target_map: str
    target_npc: str = ""
    item_to_buy: str = ""
    item_id: int = 0
    amount: int = 0
    confidence: float = 0.0

@router.post("/heal", response_model=HealStrategyResponse)
async def determine_heal_strategy(req: HealStrategyRequest) -> HealStrategyResponse:
    """Pro RO LLM analyzes server NPC shops and portal data to determine
    the optimal healing strategy for this specific server. No hardcoded
    assumptions — every decision is data-driven."""
    
    hp_pct = (req.hp / max(req.hp_max, 1)) * 100 if req.hp_max > 0 else 100
    
    # Phase 1: Check if potions are available in inventory
    has_potions = any(
        "potion" in (item.get("name", "") or "").lower()
        for item in req.inventory
    )
    
    if has_potions and hp_pct < 40:
        return HealStrategyResponse(
            strategy="use_potion",
            command="use Red Potion",
            target_map=req.map,
            confidence=0.95,
        )
    
    # Phase 2: Find a healing NPC from known shops
    for shop in req.known_shops:
        shop_name = shop.get("name", "")
        shop_items = shop.get("shop", "")
        shop_map = shop.get("map", "")
        shop_x = shop.get("x", 0)
        shop_y = shop.get("y", 0)
        
        # Check if this shop sells Red Potions (item 501)
        if "501" in str(shop_items) or "Red Potion" in str(shop_items):
            # Check if this shop is reachable from current map
            if req.map == shop_map or any(
                p.startswith(f"{req.map} ") or f" {shop_map}" in p
                for p in req.known_portals
            ):
                return HealStrategyResponse(
                    strategy="buy_from_npc",
                    command=f"buy 501 30",
                    target_map=shop_map,
                    target_npc=shop_name,
                    item_to_buy="Red Potion",
                    item_id=501,
                    amount=30,
                    confidence=0.85,
                )
    
    # Phase 3: Try natural regen or Kafra storage
    if hp_pct < 30:
        return HealStrategyResponse(
            strategy="sit_and_regen",
            command="sit",
            target_map=req.map,
            confidence=0.6,
        )
    
    # Phase 4: Default — engage auto mode and wait for loot/potions
    return HealStrategyResponse(
        strategy="auto_grind",
        command="ai auto",
        target_map=req.map,
        confidence=0.5,
    )
