"""
Element attribute database — parses rAthena attr_fix.yml for all 4 levels.

A pro player knows that ElementLevel matters enormously:
  Level 1: Standard chart (Fire vs Earth = 150%)
  Level 2: Fire vs Earth = 175% (much more advantageous)
  Level 3: Fire vs Earth = 200%, Water vs Fire = 200%
  Level 4: Fire vs Earth = 200%, Water vs Fire = 200%
           But Ghost becomes 25% to almost everything!
           Same-element attacks become 200% on Level 4!

This module replaces the hardcoded ELEMENT_MULT in combat_tactics.py.
"""

from __future__ import annotations

import logging
import os
from threading import RLock
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_ATTR_FIX_PATH = os.path.join(
    _PROJECT_ROOT,
    "knowledge", "rathena_db", "db", "pre-re", "attr_fix.yml",
)


# ---------------------------------------------------------------------------
# Parser (uses PyYAML — already a dependency of ai_sidecar)
# ---------------------------------------------------------------------------

def _parse_attr_fix(path: str) -> dict[int, dict[str, dict[str, int]]]:
    """Parse attr_fix.yml into {level: {attack_elem: {def_elem: percentage}}}.

    The YAML structure is:
      Body:
        - Level: 1
          Neutral:
            Neutral: 100
            Water: 100
            ...
          Water:
            Neutral: 100
            Water: 25
            ...
        - Level: 2
          ...
    """
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not available — cannot parse attr_fix.yml")
        return {}

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    body = data.get("Body", [])
    if not isinstance(body, list):
        logger.warning("attr_fix.yml Body is not a list")
        return {}

    charts: dict[int, dict[str, dict[str, int]]] = {}
    for entry in body:
        if not isinstance(entry, dict):
            continue
        level = int(entry.get("Level", 0))
        if level < 1 or level > 4:
            continue

        chart: dict[str, dict[str, int]] = {}
        for attack_elem, def_map in entry.items():
            if attack_elem == "Level":
                continue
            if not isinstance(def_map, dict):
                continue
            chart[attack_elem.lower()] = {}
            for def_elem, value in def_map.items():
                try:
                    chart[attack_elem.lower()][def_elem.lower()] = int(value)
                except (ValueError, TypeError):
                    continue

        if chart:
            charts[level] = chart
            logger.info("element_db loaded level %d: %d attack elements", level, len(chart))

    return charts


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_element_charts: Optional[dict[int, dict[str, dict[str, int]]]] = None
_chart_lock = RLock()


def load_element_charts(path: Optional[str] = None) -> dict[int, dict[str, dict[str, int]]]:
    """Load all element charts from attr_fix.yml.

    Returns {1: chart, 2: chart, 3: chart, 4: chart}.
    Each chart is {attack_elem: {def_elem: percentage}}.
    """
    path = path or _ATTR_FIX_PATH
    if not os.path.isfile(path):
        logger.warning("attr_fix.yml not found at %s", path)
        return {}

    charts = _parse_attr_fix(path)

    logger.info(
        "element_db loaded: %d levels from %s",
        len(charts), path,
    )
    return charts


def get_element_charts() -> dict[int, dict[str, dict[str, int]]]:
    """Return the global element charts (lazy-loaded from attr_fix.yml)."""
    global _element_charts
    with _chart_lock:
        if _element_charts is None:
            _element_charts = load_element_charts()
        return _element_charts


def get_element_multiplier(
    attack_element: str,
    defense_element: str,
    element_level: int = 1,
) -> float:
    """Get damage multiplier from the parsed rAthena data.

    Args:
        attack_element: The element of the attacking skill (e.g. "fire").
        defense_element: The element of the target monster (e.g. "earth").
        element_level: ElementLevel of the target (1-4, default 1).

    Returns:
        Multiplier as a float (e.g. 1.75 for 175%).
    """
    charts = get_element_charts()
    if not charts:
        logger.warning("No element charts loaded — defaulting to 1.0 multiplier")
        return 1.0

    if element_level not in charts:
        element_level = 1

    chart = charts[element_level]
    atk = attack_element.lower()
    def_ = defense_element.lower()

    def_map = chart.get(atk)
    if def_map is None:
        def_map = chart.get("neutral", {})
    mult = def_map.get(def_, 100)
    return mult / 100.0
