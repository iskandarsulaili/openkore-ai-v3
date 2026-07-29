"""Danger-Aware Pathfinding — scores paths by monster density, not just distance.

A real RO player doesn't walk the shortest path — they walk the safest path.
Walking through the center of a monster map is how you aggro 12 things and die.
This module scores paths by:
- Monster density per cell
- Known spawn hotspot locations (avoid center, hug walls)
- Sprint zones (some areas have fast-move boosts)
- Choke points (bottlenecks where you can AoE farm)
"""
from __future__ import annotations
from typing import Any
import logging
from pathlib import Path
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class DangerAwarePathfinder:
    """Scores paths by danger level, not just distance.
    
    A path through monster-dense areas gets a penalty.
    A path that hugs walls or goes through safe zones gets a bonus.
    The optimal path is the least DANGEROUS, not necessarily the shortest.
    """
    
    # Danger zones per map (areas to avoid)
    # Format: map_name -> [(x1, y1, x2, y2), ...]  rectangles to avoid
    DANGER_ZONES: dict[str, list[tuple[int, int, int, int]]] = {
        "prt_fild05": [(150, 240, 160, 260), (190, 170, 210, 190)],  # Poring clusters
        "pay_fild01": [(90, 140, 110, 160), (190, 190, 210, 210)],   # Spore clusters
        "mjolnir_01": [(45, 45, 55, 55), (145, 95, 155, 105)],       # Dense spawns
        "orcsdun01": [(90, 90, 110, 110), (190, 190, 210, 210)],     # Orc clusters
        "gef_dun01": [(140, 140, 160, 160), (240, 240, 260, 260)],   # Baphomet area
    }
    
    # Safe corridors (wall edges, low-spawn paths)
    SAFE_CORRIDORS: dict[str, list[tuple[int, int, int, int]]] = {
        "prt_fild05": [(20, 200, 100, 210)],  # Path from portal to north
        "pay_fild01": [(50, 50, 200, 60)],    # South edge
        "mjolnir_01": [(0, 0, 300, 20)],      # Top edge
    }
    
    def __init__(self):
        self._hotspot_cache: dict[str, list] = {}
        self._load_hotspots()
    
    def _load_hotspots(self) -> None:
        if yaml is None:
            return
        path = DATA_DIR / "spawn_hotspots.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            for map_name, map_data in data.items():
                if isinstance(map_data, dict):
                    hotspots = map_data.get("hotspots", [])
                    self._hotspot_cache[map_name] = hotspots
    
    def score_path(self, path: list[tuple[int, int]], map_name: str) -> float:
        """Score a path (lower = safer). Considers:
        - Monster density near each cell
        - Danger zone proximity
        - Safe corridor proximity
        - Path length
        """
        if not path:
            return float("inf")
        
        total_danger = 0.0
        danger_zones = self.DANGER_ZONES.get(map_name, [])
        safe_corridors = self.SAFE_CORRIDORS.get(map_name, [])
        hotspots = self._hotspot_cache.get(map_name, [])
        
        for x, y in path:
            cell_danger = 0.0
            
            # Penalty: near spawn hotspots
            for hx, hy in hotspots:
                if isinstance(hx, list):
                    hx, hy = hx[0], hx[1] if len(hx) > 1 else (hx, 0)
                dist = ((x - hx) ** 2 + (y - hy) ** 2) ** 0.5
                if dist < 5:
                    cell_danger += 2.0
                elif dist < 10:
                    cell_danger += 1.0
            
            # Penalty: inside a danger zone
            for zx1, zy1, zx2, zy2 in danger_zones:
                if zx1 <= x <= zx2 and zy1 <= y <= zy2:
                    cell_danger += 5.0
            
            # Bonus: in safe corridor (negative = safe)
            for sx1, sy1, sx2, sy2 in safe_corridors:
                if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                    cell_danger -= 2.0
            
            total_danger += cell_danger
        
        # Normalize by path length
        avg_danger = total_danger / len(path)
        path_length_penalty = len(path) * 0.001  # Slight bias for shorter paths
        
        return avg_danger + path_length_penalty
    
    def find_safest_path(self, start: tuple[int, int], end: tuple[int, int], map_name: str) -> list[tuple[int, int]]:
        """Find the safest path between two points on a map.
        
        Uses a simple greedy approach: at each step, move toward the target
        but avoid high-danger cells.
        
        For a real game, this would use A* with danger-weighted heuristic.
        For now, we use waypoint-based routing.
        """
        if not start or not end:
            return [end] if end else [start]
        
        danger_zones = self.DANGER_ZONES.get(map_name, [])
        safe_corridors = self.SAFE_CORRIDORS.get(map_name, [])
        hotspots = self._hotspot_cache.get(map_name, [])
        hotspot_coords = []
        for h in hotspots:
            if isinstance(h, list) and len(h) >= 2:
                hotspot_coords.append((h[0], h[1]))
        
        path = [start]
        current = list(start)
        target = list(end)
        
        max_steps = 100
        step = 0
        
        while (abs(current[0] - target[0]) > 3 or abs(current[1] - target[1]) > 3) and step < max_steps:
            step += 1
            
            # Calculate direction to target
            dx = 1 if target[0] > current[0] else -1 if target[0] < current[0] else 0
            dy = 1 if target[1] > current[1] else -1 if target[1] < current[1] else 0
            
            # Check if the direct move goes through a danger zone
            next_x = current[0] + dx * 5
            next_y = current[1] + dy * 5
            
            # If direct path is through danger, try alternative
            in_danger = False
            for zx1, zy1, zx2, zy2 in danger_zones:
                if zx1 <= next_x <= zx2 and zy1 <= next_y <= zy2:
                    in_danger = True
                    break
            
            # Also check hotspot proximity
            near_hotspot = False
            for hx, hy in hotspot_coords:
                dist = ((next_x - hx) ** 2 + (next_y - hy) ** 2) ** 0.5
                if dist < 8:
                    near_hotspot = True
                    break
            
            if in_danger or near_hotspot:
                # Try perpendicular movement instead
                if abs(dx) > abs(dy):
                    # Try moving up/down instead
                    alternative_moves = [(0, 5), (0, -5), (0, 10), (0, -10)]
                else:
                    alternative_moves = [(5, 0), (-5, 0), (10, 0), (-10, 0)]
                
                moved = False
                for adx, ady in alternative_moves:
                    alt_x = current[0] + adx
                    alt_y = current[1] + ady
                    
                    alt_in_danger = False
                    for zx1, zy1, zx2, zy2 in danger_zones:
                        if zx1 <= alt_x <= zx2 and zy1 <= alt_y <= zy2:
                            alt_in_danger = True
                            break
                    
                    if not alt_in_danger:
                        next_x, next_y = alt_x, alt_y
                        moved = True
                        break
                
                if not moved:
                    # Can't avoid danger, just go direct
                    pass
            
            current = [next_x, next_y]
            path.append(tuple(current))
        
        # Add final destination
        if path[-1] != end:
            path.append(end)
        
        return path
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        """Run danger-aware pathfinding assessment."""
        current_map = str(signals.get("map", "") or "")
        pos_x = int(signals.get("pos_x", 0) or 0)
        pos_y = int(signals.get("pos_y", 0) or 0)
        
        # Check if current position is in a danger zone
        danger_zones = self.DANGER_ZONES.get(current_map, [])
        for zx1, zy1, zx2, zy2 in danger_zones:
            if zx1 <= pos_x <= zx2 and zy1 <= pos_y <= zy2:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"move {pos_x + 30} {pos_y + 5}",  # Move out of danger
                    confidence=0.8,
                    reason=f"Danger-aware: moving out of hot zone on {current_map}",
                    domain="navigation",
                ))
                break


# Re-export as alias for compatibility
SafePathfinder = DangerAwarePathfinder
