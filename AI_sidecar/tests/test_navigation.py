"""
Route tests for the navigation module.

Verifies BFS pathfinding yields plausible routes through the
pre-renewal RO map topology graph.
"""

from __future__ import annotations

from ai_sidecar.combat.navigation import MAP_CONNECTIONS, Router


def make_router() -> Router:
    return Router()


def test_connections_are_bidirectional() -> None:
    """Every map listed in a neighbour's list must reciprocate."""
    for map_name, neighbours in MAP_CONNECTIONS.items():
        for n in neighbours:
            assert n in MAP_CONNECTIONS, (
                f"{map_name} lists {n} but {n} is missing from MAP_CONNECTIONS"
            )
            assert map_name in MAP_CONNECTIONS[n], (
                f"{map_name} lists {n} but {n} does not list {map_name} back"
            )


def test_major_towns_exist() -> None:
    """All expected classic towns are present in the graph."""
    expected = {
        "prontera", "morocc", "geffen", "payon", "aldebaran",
        "izlude", "alberta", "comodo", "yuno", "xmas",
        "einbroch", "lighthalzen", "hugel", "rachel",
        "gonryun", "amatsu", "ayothaya", "louyang", "umbala",
        "niflheim",
    }
    for town in expected:
        assert town in MAP_CONNECTIONS, f"Missing town: {town}"
        assert MAP_CONNECTIONS[town], f"Town {town} has no connections"


def test_same_map_returns_single_element() -> None:
    router = make_router()
    path = router.find_path("prontera", "prontera")
    assert path == ["prontera"]


def test_direct_neighbour() -> None:
    router = make_router()
    path = router.find_path("prontera", "prt_fild01")
    assert path == ["prontera", "prt_fild01"]


def test_prontera_to_payon_route() -> None:
    """Route from Prontera to Payon must exist and be plausible."""
    router = make_router()
    path = router.find_path("prontera", "payon")
    assert len(path) >= 2, f"Expected at least 2 maps, got {path}"
    assert path[0] == "prontera"
    assert path[-1] == "payon"
    assert len(path) <= 12, (
        f"Route prontera -> payon too long ({len(path)} hops): {path}"
    )
    for a, b in zip(path, path[1:]):
        assert b in MAP_CONNECTIONS[a], (
            f"No edge between {a} and {b} in route {path}"
        )


def test_morroc_to_aldebaran_route() -> None:
    """Route from Morroc to Aldebaran must exist and be plausible."""
    router = make_router()
    path = router.find_path("morocc", "aldebaran")
    assert len(path) >= 2, f"Expected at least 2 maps, got {path}"
    assert path[0] == "morocc"
    assert path[-1] == "aldebaran"
    assert len(path) <= 15, (
        f"Route morroc -> aldebaran too long ({len(path)} hops): {path}"
    )
    for a, b in zip(path, path[1:]):
        assert b in MAP_CONNECTIONS[a], (
            f"No edge between {a} and {b} in route {path}"
        )


def test_get_next_map_returns_first_hop() -> None:
    router = make_router()
    nxt = router.get_next_map("prontera", "payon")
    assert nxt is not None
    assert nxt.startswith("prt_") or nxt.startswith("gef_") or nxt.startswith("pay_")
    assert nxt != "payon"


def test_get_next_map_same_map() -> None:
    router = make_router()
    assert router.get_next_map("prontera", "prontera") is None


def test_get_next_map_unreachable() -> None:
    router = make_router()
    assert router.get_next_map("prontera", "nonexistent") is None


def test_unreachable_map_returns_empty() -> None:
    router = make_router()
    path = router.find_path("prontera", "this_map_does_not_exist")
    assert path == []


def test_navigate_commands_basic_structure() -> None:
    router = make_router()
    cmds = router.get_navigate_commands("prontera", "payon")
    assert len(cmds) >= 1
    for cmd in cmds:
        assert "kind" in cmd
        assert cmd["kind"] == "navigate"
        assert "command" in cmd
        assert "map" in cmd


def test_navigate_commands_same_map() -> None:
    router = make_router()
    cmds = router.get_navigate_commands("prontera", "prontera")
    assert len(cmds) == 1
    assert cmds[0]["action"] == "arrived"
    assert cmds[0]["command"] == "go save"


def test_navigate_commands_unreachable() -> None:
    router = make_router()
    cmds = router.get_navigate_commands("prontera", "no_such_map")
    assert len(cmds) >= 1
    actions = {c["action"] for c in cmds}
    assert actions & {"retreat", "emergency"}, f"Expected retreat/emergency, got {actions}"


def test_navigate_with_teleport() -> None:
    router = make_router()
    cmds = router.get_navigate_commands("prontera", "prt_fild01", use_teleport=True)
    assert cmds[0]["action"] == "teleport"
    assert cmds[0]["command"] == "tele"


def test_navigate_town_safe_coords() -> None:
    router = make_router()
    cmds = router.get_navigate_commands("prontera", "prt_fild01")
    move_cmds = [c for c in cmds if c["action"] == "move"]
    assert any("prontera 156 191" in c["command"] for c in move_cmds), (
        f"No prontera safe-coord move in {cmds}"
    )


def test_route_cache_hit() -> None:
    router = make_router()
    path1 = router.find_path("geffen", "morocc")
    path2 = router.find_path("geffen", "morocc")
    assert path1 == path2
    assert path1[0] == "geffen"
    assert path1[-1] == "morocc"


def test_geffen_to_morocc_via_desert() -> None:
    """Geffen to Morocc should route through shared fields."""
    router = make_router()
    path = router.find_path("geffen", "morocc")
    assert "gef_fild04" in path or "gef_fild05" in path, (
        f"Expected route via Geffen-Morocc connector fields, got {path}"
    )


def test_prontera_to_izlude_route() -> None:
    """Prontera to Izlude must exist."""
    router = make_router()
    path = router.find_path("prontera", "izlude")
    assert len(path) >= 2
    assert path[0] == "prontera"
    assert path[-1] == "izlude"
    for a, b in zip(path, path[1:]):
        assert b in MAP_CONNECTIONS[a]


def test_prontera_alberta_route() -> None:
    """Prontera to Alberta must exist."""
    router = make_router()
    path = router.find_path("prontera", "alberta")
    assert len(path) >= 2
    assert path[0] == "prontera"
    assert path[-1] == "alberta"


def test_prontera_to_geffen_short_route() -> None:
    """Prontera to Geffen should be short."""
    router = make_router()
    path = router.find_path("prontera", "geffen")
    assert len(path) <= 6, f"Route too long: {path}"


def test_payon_dungeon_route() -> None:
    """Payon to its dungeon floor 4 should chain through all floors."""
    router = make_router()
    path = router.find_path("payon", "pay_dun04")
    assert path[0] == "payon"
    assert path[-1] == "pay_dun04"
    assert "pay_dun00" in path
    assert "pay_dun01" in path


def test_byalan_dungeon_route() -> None:
    """Izlude to Byalan floor 4 should chain through all Byalan floors."""
    router = make_router()
    path = router.find_path("izlude", "iz_dun04")
    assert path[0] == "izlude"
    assert path[-1] == "iz_dun04"
    assert "iz_dun00" in path
    assert "iz_dun01" in path


def test_clear_cache() -> None:
    router = make_router()
    router.find_path("prontera", "morocc")
    assert len(router._route_cache) >= 1
    router.clear_cache()
    assert len(router._route_cache) == 0
