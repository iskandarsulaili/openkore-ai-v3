"""
Map knowledge system — pre-populated map intelligence.

The bot knows what's on every map BEFORE entering, using rAthena mob_db data
and portal connection data. No learning-by-dying — all knowledge is pre-loaded.

Data sources:
- rAthena mob_db: which monsters spawn on which maps
- rAthena map_index: map connections and warp points
- Pre-loaded safety ratings from player knowledge
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MapSafety(Enum):
    SAFE = "safe"               # Town maps, no aggressive monsters
    CAUTIOUS = "cautious"       # Low-level fields, low danger
    DANGEROUS = "dangerous"     # Mid-level fields, aggressive monsters
    DEADLY = "deadly"           # High-level fields, MVP spawns, PK zones
    RESTRICTED = "restricted"   # WoE castles, quest instances, PvP arenas


class SpawnType(Enum):
    PASSIVE = "passive"         # Won't attack
    AGGRESSIVE = "aggressive"   # Attacks on sight
    BOSS = "boss"               # MVP
    MINION = "minion"           # MVP adds
    GUARD = "guard"             # Town guards
    NPC = "npc"                 # Non-combatant


@dataclass
class MapMonster:
    """Monster spawn data on a map."""
    name: str
    level: int
    spawn_count: int
    spawn_type: SpawnType
    element: str = "neutral"
    size: str = "medium"
    race: str = "demi-human"
    aggro_radius: int = 0       # Tiles before aggro (0 = passive)
    is_mvp: bool = False
    is_undead: bool = False
    drops_useful: list[str] = field(default_factory=list)


@dataclass
class MapPortal:
    """Portal connection to another map."""
    target_map: str
    x: int
    y: int
    requires_quest: bool = False
    requires_level: int = 0
    quest_name: str = ""


@dataclass
class MapKnowledge:
    """Complete knowledge about a single map."""
    name: str
    safety: MapSafety = MapSafety.CAUTIOUS
    recommended_level: tuple[int, int] = (1, 999)  # (min, max)
    is_town: bool = False
    has_healer: bool = False
    has_shop: bool = False
    has_save_point: bool = False
    has_refinery: bool = False
    has_identifier: bool = False
    
    # Monsters
    monsters: list[MapMonster] = field(default_factory=list)
    has_aggressive: bool = False
    has_mvp: bool = False
    mvp_names: list[str] = field(default_factory=list)
    
    # Connections
    portals: list[MapPortal] = field(default_factory=list)
    connected_maps: list[str] = field(default_factory=list)
    
    # Danger signals
    is_pk_zone: bool = False
    is_woe_castle: bool = False
    has_death_logic: bool = False    # Falling into pits, instant-death zones
    
    # Economy
    npc_buyers: list[str] = field(default_factory=list)      # NPC names that buy items
    npc_sellers: list[str] = field(default_factory=list)     # NPC names that sell items
    resale_value_pct: float = 0.25    # % of NPC buy price for items sold here
    
    # Notes
    notes: str = ""
    team_secret: str = ""             # Known only to your bots


# ── Master map database ──

MAP_KNOWLEDGE: dict[str, MapKnowledge] = {}


def _init_map_db():
    """Initialize map knowledge database from known RO maps."""
    
    # ── Town maps ──
    _add(MapKnowledge("prontera", safety=MapSafety.SAFE, is_town=True,
                       has_healer=True, has_shop=True, has_save_point=True,
                       has_refinery=True, has_identifier=True,
                       resale_value_pct=0.35,
                       npc_buyers=["Tool Dealer", "Weapon Dealer", "Armor Dealer",
                                    "Potion Dealer", "General Store"],
                       connected_maps=["prt_fild01", "prt_fild02", "prt_fild03",
                                       "prt_fild04", "prt_fild05", "prt_fild06",
                                       "prt_fild07", "prt_fild08", "prt_fild09",
                                       "prt_fild10", "prt_in01", "prt_in02",
                                       "prt_church", "prt_castle", "prt_sewb1",
                                       "prt_sewb2", "prt_sewb3"]))
    
    _add(MapKnowledge("alberta", safety=MapSafety.SAFE, is_town=True,
                       has_healer=True, has_shop=True, has_save_point=True,
                       resale_value_pct=0.30,
                       connected_maps=["alb_ship", "alb_ship2", "alb2_braid",
                                       "alberta_in", "alberta_in02"]))
    
    _add(MapKnowledge("izlude", safety=MapSafety.SAFE, is_town=True,
                       has_healer=True, has_shop=False, has_save_point=True,
                       # Connected maps from the RAW server portal table
                       # (tables/portals.txt): izlude town reaches prt_fild08
                       # (20,98 -> 367,212), the Novice Academy iz_ac01, and
                       # izlude_in. izlude does NOT connect to izlude_c (that
                       # sub-map is reachable only via the boat/prontera side),
                       # so the REACHABLE farm from izlude is prt_fild08 — the
                       # bot re-locks to prt_fild08c once inside prt_fild08.
                       connected_maps=["izlude_in", "izlude_in02", "izlude_boat",
                                       "prt_fild08", "iz_ac01", "izlu2dun"]))
    
    _add(MapKnowledge("morroc", safety=MapSafety.CAUTIOUS, is_town=True,
                       has_healer=True, has_shop=True, has_save_point=True,
                       connected_maps=["moc_fild01", "moc_fild02", "moc_fild03",
                                       "moc_fild04", "moc_fild05", "moc_fild06",
                                       "moc_fild07", "moc_fild08", "moc_fild09",
                                       "moc_fild10", "moc_pryd01", "moc_pryd02",
                                       "moc_pryd03", "moc_pryd04", "moc_pryd05",
                                       "moc_pryd06"]))
    
    _add(MapKnowledge("payon", safety=MapSafety.SAFE, is_town=True,
                       has_healer=True, has_shop=True, has_save_point=True,
                       connected_maps=["pay_fild01", "pay_fild02", "pay_fild03",
                                       "pay_fild04", "pay_fild05", "pay_fild06",
                                       "pay_fild07", "pay_fild08", "pay_fild09",
                                       "pay_fild10", "pay_fild11", "pay_arche",
                                       "pay_dun00", "pay_dun01", "pay_dun02"]))
    
    _add(MapKnowledge("geffen", safety=MapSafety.SAFE, is_town=True,
                       has_healer=True, has_shop=True, has_save_point=True,
                       has_refinery=True,
                       connected_maps=["gef_fild01", "gef_fild02", "gef_fild03",
                                       "gef_fild04", "gef_fild05", "gef_fild06",
                                       "gef_fild07", "gef_fild08", "gef_fild09",
                                       "gef_fild10", "gef_fild11", "gef_fild12",
                                       "gef_fild13", "gef_fild14", "gef_tower"]))
    
    # ── Hunting maps ──
    _add(MapKnowledge("prt_fild08", safety=MapSafety.CAUTIOUS,
                       recommended_level=(10, 30),
                       monsters=[
                           MapMonster("Poring", 1, 12, SpawnType.PASSIVE,
                                      element="water", size="small", race="slime"),
                           MapMonster("Fabre", 2, 10, SpawnType.PASSIVE,
                                      element="neutral", size="small", race="insect"),
                           MapMonster("Picky", 3, 8, SpawnType.PASSIVE,
                                      element="fire", size="small", race="insect"),
                           MapMonster("Chonchon", 4, 6, SpawnType.AGGRESSIVE,
                                      element="wind", size="small", race="insect",
                                      aggro_radius=3),
                           MapMonster("Hornet", 5, 5, SpawnType.AGGRESSIVE,
                                      element="wind", size="small", race="insect",
                                      aggro_radius=3),
                           MapMonster("Thief Bug", 6, 4, SpawnType.AGGRESSIVE,
                                      element="neutral", size="small", race="insect",
                                      aggro_radius=4),
                       ],
                       connected_maps=["prontera", "prt_fild07", "prt_fild09",
                                       "prt_sewb1"]))
    
    # ── Academy / early-game fields (level 1-10) ──
    # prt_fild08c is the Novice-Academy farm reachable from izlude_c (the
    # satellite-town variant). A level-1 bot spawns in izlude and must reach a
    # farm it can actually ROUTE to — without these entries get_hunting_maps(1)
    # returns [] and the cold-start falls back to a hardcoded map.
    _add(MapKnowledge("prt_fild08c", safety=MapSafety.CAUTIOUS,
                       recommended_level=(1, 12),
                       monsters=[
                           MapMonster("Poring", 1, 14, SpawnType.PASSIVE,
                                      element="water", size="small", race="slime"),
                           MapMonster("Lunatic", 1, 10, SpawnType.PASSIVE,
                                      element="neutral", size="small", race="brute"),
                           MapMonster("Pupa", 2, 8, SpawnType.PASSIVE,
                                      element="neutral", size="small", race="insect"),
                           MapMonster("Thief Bug Egg", 1, 8, SpawnType.PASSIVE,
                                      element="neutral", size="small", race="insect"),
                           MapMonster("Thief Bug", 3, 6, SpawnType.AGGRESSIVE,
                                      element="neutral", size="small", race="insect",
                                      aggro_radius=3),
                       ],
                       connected_maps=["izlude_c", "prt_fild07"]))
    
    _add(MapKnowledge("prt_fild05", safety=MapSafety.CAUTIOUS,
                       recommended_level=(1, 12),
                       monsters=[
                           MapMonster("Poring", 1, 14, SpawnType.PASSIVE,
                                      element="water", size="small", race="slime"),
                           MapMonster("Lunatic", 1, 10, SpawnType.PASSIVE,
                                      element="neutral", size="small", race="brute"),
                           MapMonster("Pupa", 2, 8, SpawnType.PASSIVE,
                                      element="neutral", size="small", race="insect"),
                           MapMonster("Thief Bug", 3, 6, SpawnType.AGGRESSIVE,
                                      element="neutral", size="small", race="insect",
                                      aggro_radius=3),
                       ],
                       connected_maps=["prontera", "mjolnir_09"]))
    
    _add(MapKnowledge("pay_fild04", safety=MapSafety.DANGEROUS,
                       recommended_level=(40, 70),
                       monsters=[
                           MapMonster("Orc Warrior", 25, 10, SpawnType.AGGRESSIVE,
                                      element="earth", size="medium", race="demi-human",
                                      aggro_radius=5,
                                      drops_useful=["Orcish Axe", "Orc Warrior Helm"]),
                           MapMonster("Orc Archer", 27, 8, SpawnType.AGGRESSIVE,
                                      element="earth", size="medium", race="demi-human",
                                      aggro_radius=10,
                                      drops_useful=["Bow", "Orcish Voucher"]),
                           MapMonster("Orc Skeleton", 30, 6, SpawnType.AGGRESSIVE,
                                      element="undead", size="medium", race="undead",
                                      aggro_radius=6,
                                      drops_useful=["Bones", "Orcish Gift"]),
                       ],
                       connected_maps=["payon", "pay_fild03", "pay_fild05"]))
    
    _add(MapKnowledge("gef_fild10", safety=MapSafety.DANGEROUS,
                       recommended_level=(60, 90),
                       monsters=[
                           MapMonster("High Orc", 55, 6, SpawnType.AGGRESSIVE,
                                      element="earth", size="medium", race="demi-human",
                                      aggro_radius=8,
                                      drops_useful=["Steel", "Coal"]),
                           MapMonster("Orc Hero", 77, 1, SpawnType.BOSS,
                                      element="earth", size="large", race="demi-human",
                                      is_mvp=True,
                                      drops_useful=["Heroic Yoyo", "Beads"]),
                       ],
                       connected_maps=["geffen", "gef_fild09"]))
    
    _add(MapKnowledge("pay_dun01", safety=MapSafety.DANGEROUS,
                       recommended_level=(30, 55),
                       monsters=[
                           MapMonster("Skeleton", 20, 8, SpawnType.AGGRESSIVE,
                                      element="undead", size="medium", race="undead",
                                      is_undead=True, aggro_radius=5,
                                      drops_useful=["Bones", "Skull Helmet"]),
                           MapMonster("Zombie", 25, 6, SpawnType.AGGRESSIVE,
                                      element="undead", size="medium", race="undead",
                                      is_undead=True, aggro_radius=6,
                                      drops_useful=["Zombie's Rotting Flesh"]),
                           MapMonster("Wolf", 28, 5, SpawnType.AGGRESSIVE,
                                      element="neutral", size="small", race="brute",
                                      aggro_radius=8,
                                      drops_useful=["Wolf Claws"]),
                       ],
                       connected_maps=["payon", "pay_dun00", "pay_dun02"]))
    
    _add(MapKnowledge("prt_sewb1", safety=MapSafety.DANGEROUS,
                       recommended_level=(15, 35),
                       monsters=[
                           MapMonster("Familiar", 10, 8, SpawnType.AGGRESSIVE,
                                      element="dark", size="medium", race="demon",
                                      aggro_radius=5),
                           MapMonster("Thief Bug", 8, 6, SpawnType.AGGRESSIVE,
                                      element="neutral", size="small", race="insect",
                                      aggro_radius=4),
                           MapMonster("Drainliar", 15, 4, SpawnType.AGGRESSIVE,
                                      element="dark", size="small", race="brute",
                                      aggro_radius=6),
                       ],
                       connected_maps=["prontera", "prt_sewb2", "prt_fild08"]))
    
    # ── MVP spawn maps ──
    _add(MapKnowledge("moc_pryd06", safety=MapSafety.DEADLY,
                       recommended_level=(70, 99),
                       monsters=[
                           MapMonster("Mistress", 74, 1, SpawnType.BOSS,
                                      element="wind", size="medium", race="demon",
                                      is_mvp=True, aggro_radius=10,
                                      drops_useful=["Mistress Crown", "Wing of Wind"]),
                       ],
                       has_mvp=True, mvp_names=["Mistress"],
                       connected_maps=["moc_pryd05"]))
    
    _add(MapKnowledge("gef_fild10", safety=MapSafety.DEADLY,
                       recommended_level=(60, 99),
                       monsters=[
                           MapMonster("Orc Hero", 77, 1, SpawnType.BOSS,
                                      element="earth", size="large", race="demi-human",
                                      is_mvp=True, aggro_radius=10,
                                      drops_useful=["Heroic Yoyo", "Beads"]),
                       ],
                       has_mvp=True, mvp_names=["Orc Hero"],
                       connected_maps=["gef_fild09"]))
    
    # ── Dead maps (knowledge) ──
    _add(MapKnowledge("gef_tower", safety=MapSafety.DEADLY,
                       recommended_level=(80, 99),
                       monsters=[
                           MapMonster("Baphomet", 98, 1, SpawnType.BOSS,
                                      element="dark", size="large", race="demon",
                                      is_mvp=True, aggro_radius=12,
                                      drops_useful=["Baphomet Horns", "Baphomet Card"]),
                       ],
                       has_mvp=True, mvp_names=["Baphomet"],
                       notes="Full-map AoE (Hell's Judgement). Need Assumptio and Panacea.",
                       connected_maps=["geffen"]))


def _add(knowledge: MapKnowledge):
    """Add a map to the knowledge base."""
    MAP_KNOWLEDGE[knowledge.name] = knowledge


# Initialize at module load
_init_map_db()


def get_map_knowledge(map_name: str) -> MapKnowledge | None:
    """Get knowledge about a specific map."""
    return MAP_KNOWLEDGE.get(map_name)


def get_safe_maps(min_level: int = 1) -> list[str]:
    """Get list of safe maps appropriate for a level."""
    result = []
    for name, mk in MAP_KNOWLEDGE.items():
        if mk.safety in (MapSafety.SAFE, MapSafety.CAUTIOUS):
            if mk.recommended_level[0] <= min_level <= mk.recommended_level[1]:
                result.append(name)
    return result


def get_hunting_maps(char_level: int, max_danger: MapSafety = MapSafety.DANGEROUS) -> list[tuple[str, float]]:
    """Get hunting maps sorted by suitability for character level."""
    scored = []
    for name, mk in MAP_KNOWLEDGE.items():
        if mk.is_town:
            continue
        if mk.safety.value > max_danger.value:
            continue
        
        min_lv, max_lv = mk.recommended_level
        if char_level < min_lv - 5:
            continue
        if char_level > max_lv + 10:
            continue
        
        # Score: closest to middle of recommended range
        middle = (min_lv + max_lv) / 2
        level_fit = 1.0 - min(abs(char_level - middle) / (max_lv - min_lv + 1), 0.5)
        danger_penalty = {"safe": 0.0, "cautious": 0.1, "dangerous": 0.2, "deadly": 0.4}[
            mk.safety.value]
        
        score = level_fit - danger_penalty + (0.1 if mk.has_healer else 0)
        scored.append((name, round(score, 3)))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def reachable_hunting_maps(from_map: str, char_level: int, max_danger: MapSafety = MapSafety.DANGEROUS) -> list[tuple[str, float]]:
    """Hunting maps REACHABLE from `from_map` via the portal graph, scored.

    The raw `get_hunting_maps` ranks every map by level-fit, but a level-1 bot
    in izlude cannot walk to prt_fild05 (no portal). Locking an unreachable
    farm makes OpenKore's A* fail -> 'Unable to calculate a route' -> spin.
    This filters the candidates through a BFS over the connected_maps portal
    graph (the same facts the client uses) so the returned farm is ACTUALLY
    routable from the bot's current map. Falls back to the raw ranking when
    the portal graph has no entry for the current map.
    """
    try:
        _norm = lambda m: (m or "").lower().rstrip(".gat")
        _from = _norm(from_map)
        if not _from:
            return get_hunting_maps(char_level, max_danger)
        _graph: dict[str, set[str]] = {}
        for _name, _mk in MAP_KNOWLEDGE.items():
            _nm = _norm(_name)
            for _c in (_mk.connected_maps or []):
                _cn = _norm(_c)
                _graph.setdefault(_nm, set()).add(_cn)
                # RO portals are bidirectional — add the reverse edge so a
                # map without its own entry (e.g. izlude_c) still traverses.
                _graph.setdefault(_cn, set()).add(_nm)
        # BFS over the portal graph. Depth cap 2: OpenKore's route calc chains
        # only ~1-2 portal hops — a 3+-hop "reachable" map (e.g. the prt_fild08c
        # clone) makes it fail 'Cannot calculate a route' and spin. Restrict to
        # directly-reachable (1-2 hop) maps the client can actually traverse.
        _seen = {_from}
        _frontier = [_from]
        _reachable: dict[str, int] = {}  # map -> hop distance
        for _hop in range(2):
            _next: list[str] = []
            for _m in _frontier:
                for _n in _graph.get(_m, ()):
                    if _n not in _seen:
                        _seen.add(_n)
                        _next.append(_n)
                        if _n in _graph:  # known map
                            # keep the SHORTEST distance to each reachable map
                            if _n not in _reachable or _hop + 1 < _reachable[_n]:
                                _reachable[_n] = _hop + 1
            if not _next:
                break
            _frontier = _next
        if not _reachable:
            return get_hunting_maps(char_level, max_danger)
        _all = get_hunting_maps(char_level, max_danger)
        # Score: level-fit MINUS hop distance (a 1-hop farm beats a 3-hop clone
        # so OpenKore's portal-chain route calc can actually reach it).
        _scored = []
        for _n, _s in _all:
            _d = _reachable.get(_norm(_n))
            if _d is not None:
                _scored.append((_n, round(_s - (_d - 1) * 0.35, 3)))
        _scored.sort(key=lambda x: x[1], reverse=True)
        if _scored:
            return _scored
        # No level-appropriate reachable HUNT map. OpenKore's route calc can only
        # chain ~1-2 portal hops, so return the closest reachable FIELD map (any
        # level) as a transit/fallback target the bot CAN actually route to —
        # the academy/level progression logic re-locks once the bot is there.
        _closest: list[tuple[str, int]] = sorted(_reachable.items(), key=lambda kv: kv[1])
        _field_opts = [(_n, round(1.0 - (_d - 1) * 0.35, 3))
                       for _n, _d in _closest if "_fild" in _n or "_dun" in _n or "_sewb" in _n]
        return _field_opts or _all
    except Exception:
        return get_hunting_maps(char_level, max_danger)


def get_mvp_maps() -> list[tuple[str, list[str]]]:
    """Get all maps with MVP spawns."""
    return [(name, mk.mvp_names) for name, mk in MAP_KNOWLEDGE.items() if mk.has_mvp]


def get_route_safety(map_a: str, map_b: str) -> float:
    """Get safety score (0.0-1.0) for route between two maps."""
    mk_a = MAP_KNOWLEDGE.get(map_a)
    mk_b = MAP_KNOWLEDGE.get(map_b)
    
    if not mk_a or not mk_b:
        return 0.5  # Unknown maps — assume moderate safety
    
    safety_scores = {"safe": 1.0, "cautious": 0.8, "dangerous": 0.5, "deadly": 0.2, "restricted": 0.0}
    score_a = safety_scores.get(mk_a.safety.value, 0.5)
    score_b = safety_scores.get(mk_b.safety.value, 0.5)
    
    if mk_a.connected_maps and map_b in mk_a.connected_maps:
        return (score_a + score_b) / 2 + 0.1  # Direct connection is safer
    
    return (score_a + score_b) / 2


def get_town_maps() -> list[str]:
    """Get all town maps."""
    return [name for name, mk in MAP_KNOWLEDGE.items() if mk.is_town]
