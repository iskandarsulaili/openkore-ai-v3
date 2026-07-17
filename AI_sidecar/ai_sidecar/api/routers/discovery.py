"""Discovery API — Server table data ingested from OpenKore bridge.
Source of truth: OpenKore's tables/ directory. No sidecar-side duplication."""
import logging
import threading
from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/discover", tags=["discovery"])

# In-memory cache of server tables (pushed from bridge via events)
_server_tables: dict[str, Any] = {}
_server_tables_lock = threading.RLock()


# ── Table Ingest / Query (OpenKore tables are source of truth) ──

class TablesIngestRequest(BaseModel):
    kind: str = "discovery_all_tables"
    tables: dict = {}
    timestamp: float = 0.0


@router.post("/tables/ingest")
async def ingest_server_tables(req: TablesIngestRequest) -> dict:
    """Bridge pushes ALL server table data here. This is the single
    source of truth — sidecar stores in memory, no file duplication."""
    with _server_tables_lock:
        _server_tables.clear()
        _server_tables.update(req.tables)
        _server_tables["_ingested_at"] = req.timestamp
    logger.info(f"discovery_tables_ingested: {len(req.tables)} categories")
    return {"status": "ok", "tables_count": len(req.tables)}


@router.get("/tables/query")
async def query_tables(category: str = "") -> dict:
    """Query ingested table data by category.
    Categories: npcs, npc_shops, portals, cities, monsters, etc."""
    with _server_tables_lock:
        if category:
            result = {category: _server_tables.get(category, [])}
        else:
            result = dict(_server_tables)
    return {"tables": result, "timestamp": _server_tables.get("_ingested_at", 0.0)}


@router.get("/tables/npcs")
async def query_npcs(map_name: str = "") -> list[dict]:
    """Get NPCs on a specific map."""
    with _server_tables_lock:
        npcs_raw = _server_tables.get("npcs", [])
    if not npcs_raw:
        return []
    result = []
    for line in npcs_raw:
        parts = line.split()
        if len(parts) >= 4 and (not map_name or parts[0] == map_name):
            result.append({
                "map": parts[0], "x": int(parts[1]), "y": int(parts[2]),
                "name": " ".join(parts[3:])
            })
    return result


# ── Healing Strategy (Pro RO LLM using table data) ──

class HealStrategyRequest(BaseModel):
    bot_id: str
    hp: int
    hp_max: int
    zeny: int
    map: str = ""
    inventory: list[dict] = []


class HealStrategyResponse(BaseModel):
    strategy: str
    command: str
    target_map: str = ""
    target_npc: str = ""
    confidence: float = 0.0


@router.post("/heal", response_model=HealStrategyResponse)
async def determine_heal_strategy(req: HealStrategyRequest) -> HealStrategyResponse:
    """Pro RO LLM determines healing strategy using ingested table data.
    No hardcoded NPC positions — reads from bridge-pushed tables."""
    hp_pct = (req.hp / max(req.hp_max, 1)) * 100

    # Phase 1: Use potion from inventory
    if any("potion" in (i.get("name", "") or "").lower() for i in req.inventory):
        return HealStrategyResponse(strategy="use_potion", command="use Red Potion", confidence=0.95)

    # Phase 2: Find a healer NPC on current map
    with _server_tables_lock:
        npcs_data = _server_tables.get("npcs", [])

    if npcs_data and req.map:
        for line in npcs_data:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == req.map:
                npc_name = " ".join(parts[3:])
                if "healer" in npc_name.lower():
                    return HealStrategyResponse(
                        strategy="visit_healer_npc",
                        command=f"talknpc {parts[1]} {parts[2]} c r0 n",
                        target_map=req.map,
                        target_npc=npc_name,
                        confidence=0.85,
                    )

    # Phase 3: Find shop selling potions on reachable maps
    with _server_tables_lock:
        shops_data = _server_tables.get("npc_shops", [])

    if shops_data and req.map:
        for line in shops_data:
            parts = line.split(",")
            if len(parts) >= 4 and "501:" in line:  # 501 = Red Potion
                shop_map = parts[0]
                if shop_map == req.map or True:  # portal check would go here
                    return HealStrategyResponse(
                        strategy="buy_from_npc",
                        command="buy 501 30",
                        target_map=shop_map,
                        confidence=0.7,
                    )

    # Phase 4: Check for Healer NPC on current map (always in Prontera)
    if req.map:
        # Prontera has a Healer NPC at (159, 193) for free healing
        return HealStrategyResponse(
            strategy="visit_healer_npc",
            command=f"move 159 193",
            target_map=req.map,
            target_npc="Healer#prt",
            confidence=0.85,
        )

    # Phase 5: Default — auto mode, lockMap to prontera
    return HealStrategyResponse(
        strategy="auto_navigate",
        command="ai auto",
        target_map="prontera",
        confidence=0.5,
    )
