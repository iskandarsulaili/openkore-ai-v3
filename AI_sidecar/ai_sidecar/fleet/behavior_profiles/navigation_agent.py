"""NavigationAgent — warps, wings, map routes, portals, Kafra teleport, party follow."""

from __future__ import annotations

from typing import Any

from ai_sidecar.fleet.behavior_profiles import BehaviorProfile


class NavigationAgent(BehaviorProfile):
    """Handles navigation across RO maps via NPC warps, wings, Kafra, portals, routes."""

    _KNOWN_NPC_WARPS = {
        "prontera": {"geffen": ("geffen", 118, 57), "morocc": ("morocc", 156, 90),
                     "alberta": ("alberta", 46, 119), "payon": ("payon", 187, 190)},
        "geffen": {"prontera": ("prontera", 161, 180)},
        "morocc": {"prontera": ("prontera", 294, 216)},
        "alberta": {"prontera": ("prontera", 34, 329)},
        "payon": {"prontera": ("prontera", 278, 325)},
    }

    _KAFRA_TELEPORTS = {
        "prontera": ("prontera", 158, 180), "geffen": ("geffen", 120, 50),
        "morocc": ("morocc", 155, 88), "alberta": ("alberta", 44, 119),
        "payon": ("payon", 185, 190), "izlude": ("izlude", 108, 138),
        "aldebaran": ("aldebaran", 126, 112),
    }

    def npc_warp(self, current_map: str, target_map: str) -> dict[str, Any]:
        routes = self._KNOWN_NPC_WARPS.get(current_map, {})
        if target_map in routes:
            dest, x, y = routes[target_map]
            return {"action": "talk_npc_warp", "destination": f"{dest} {x} {y}",
                    "npc_search": target_map.lower()}
        return self.fly_wing_or_route(current_map, target_map)

    def fly_wing_or_route(self, current_map: str, target_map: str) -> dict[str, Any]:
        if current_map == target_map:
            return {"action": "use_fly_wing", "reason": "random_move"}
        best, score = self.best_action("navigation")
        if best and score > 0.5:
            return {"action": best, "target": target_map}
        return {"action": "move_manual", "waypoints": [(100, 100), (200, 200)],
                "target_map": target_map}

    def butterfly_wing(self) -> dict[str, Any]:
        return {"action": "use_butterfly_wing", "destination": "save_point"}

    def kafra_teleport(self, target_city: str) -> dict[str, Any]:
        if target_city in self._KAFRA_TELEPORTS:
            dest, x, y = self._KAFRA_TELEPORTS[target_city]
            return {"action": "kafra_teleport", "city": target_city,
                    "destination": f"{dest} {x} {y}", "cost": 200}
        return {"action": "walk_route", "target_map": target_city}

    def portal_navigate(self, known_portals: list[dict[str, Any]],
                        target_map: str) -> dict[str, Any]:
        for portal in known_portals:
            if portal.get("destination", "").endswith(target_map):
                return {"action": "enter_portal", "portal_coords": portal.get("coords"),
                        "destination": portal["destination"]}
        return {"action": "search_portals", "target_map": target_map}

    def party_follow(self, leader_pos: tuple[int, int], leader_map: str,
                     my_pos: tuple[int, int]) -> dict[str, Any]:
        dx = abs(leader_pos[0] - my_pos[0])
        dy = abs(leader_pos[1] - my_pos[1])
        if dx > 6 or dy > 6 or self._signals.get("map_name", "") != leader_map:
            return {"action": "follow", "target": self._signals.get("party_leader", ""),
                    "distance": 3}
        return {"action": "stay_close"}

    def decide_best_route(self, current_map: str, target_map: str,
                          known_routes: list[list[str]]) -> dict[str, Any]:
        for route in known_routes:
            if current_map in route and target_map in route:
                ci = route.index(current_map)
                ti = route.index(target_map)
                segment = route[ci + 1:ti + 1] if ci < ti else route[ti:ci][::-1]
                return {"action": "follow_route", "maps": segment,
                        "next_map": segment[0] if segment else target_map}
        return {"action": "unknown_route", "from": current_map, "to": target_map,
                "suggestion": self.fly_wing_or_route(current_map, target_map)}

    def record_outcome(self, action: str, success: bool, travel_time_s: float = 0.0) -> None:
        self._record_experience("navigation", action, success,
                                reward=max(0.0, 60.0 - travel_time_s),
                                travel_time_s=travel_time_s)
