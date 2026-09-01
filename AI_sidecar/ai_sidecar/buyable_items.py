"""Buyable item set — data-driven from the RAW server's actual shop NPC scripts.

AGNOSTIC (RULE.md): the set of items a bot can actually BUY from an NPC shop is
derived by parsing the server's shop-type NPC scripts (npc/**/*.txt `shop`
lines), NOT a hardcoded list. This is the ground truth for "is this item
purchasable" — the item DB alone cannot tell buyable from event/rare rewards
(event items like Blue Twohand Axe id 28103 have Buy=10 but no shop sells them).

The set is cached in-process (parsing ~800 shop lines is cheap) and refreshed
on demand. Falls back to an empty set (callers then treat nothing as buyable)
if the server tree is unavailable — never a hardcoded literal.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Candidate server roots (repo layout varies). First existing wins.
_SERVER_ROOTS = [
    Path("/home/lot399/rathena-AI-world"),   # RAW live server
    Path(__file__).parent.parent.parent.parent / "rathena-AI-world",
    Path(__file__).parent.parent.parent / "rathena-AI-world",
]

_lock = threading.RLock()
_cache: set[int] | None = None
_cache_root: str = ""


def _find_server_root() -> Path | None:
    for root in _SERVER_ROOTS:
        if root.exists() and (root / "npc").is_dir():
            return root
    return None


def _parse_shop_items(root: Path) -> set[int]:
    """Parse all `shop`-type NPC lines under npc/ and collect buyable item IDs."""
    buyable: set[int] = set()
    npc_dir = root / "npc"
    if not npc_dir.is_dir():
        return buyable
    for f in npc_dir.rglob("*.txt"):
        if ".bak" in f.name:
            continue
        try:
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "\tshop\t" not in line and "\tshop " not in line:
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                # parts[3] = "look,item:price,item:price,..."
                items = parts[3].split(",")[1:]
                for it in items:
                    m = re.match(r"(\d+):", it.strip())
                    if m:
                        buyable.add(int(m.group(1)))
        except Exception:
            continue
    return buyable


def get_buyable_items(force_reload: bool = False) -> set[int]:
    """Return the set of item IDs buyable from the server's NPC shops."""
    global _cache, _cache_root
    with _lock:
        if _cache is not None and not force_reload:
            return _cache
        root = _find_server_root()
        if root is None:
            logger.warning("buyable_items: no server root found; empty set")
            _cache = set()
            _cache_root = ""
            return _cache
        _cache = _parse_shop_items(root)
        _cache_root = str(root)
        logger.info("buyable_items_loaded: %d items from %s", len(_cache), root)
        return _cache


def is_buyable(item_id: int | str) -> bool:
    """True if the item is sold by an NPC shop on the server."""
    try:
        return int(item_id) in get_buyable_items()
    except (TypeError, ValueError):
        return False


def buyable_stats() -> dict[str, Any]:
    with _lock:
        return {
            "count": len(_cache) if _cache is not None else 0,
            "root": _cache_root,
            "loaded": _cache is not None,
        }
