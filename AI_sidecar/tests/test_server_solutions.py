"""Tests for the DB-backed server-solutions knowledge store (server-agnostic)."""
from ai_sidecar.server_adaptation import ServerSolutionsStore


def test_store_default_and_set_get():
    """A store with no DB can still set/get per-server facts."""
    s = ServerSolutionsStore(server_key="s1")
    assert s.get("anything", "none") == "none"
    s.set("potion_solution", {"buy_command": "buy 501 30"}, value_json='{"buy_command": "buy 501 30"}')
    got = s.get_json("potion_solution")
    assert got.get("buy_command") == "buy 501 30"
    s.set("farm_map", "prt_fild08c")
    assert s.get("farm_map") == "prt_fild08c"


def test_store_per_server_isolation():
    """Different server keys must not leak solution facts."""
    a = ServerSolutionsStore(server_key="srvA")
    b = ServerSolutionsStore(server_key="srvB")
    a.set("safe_town", "prontera")
    assert b.get("safe_town", "none") == "none"
    assert a.get("safe_town") == "prontera"
