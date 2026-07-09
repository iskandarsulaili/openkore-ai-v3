"""Integration test — starts the server, verifies health endpoint, checks PDCA.

Run: python ai_sidecar/test_integration.py
"""

from __future__ import annotations

import sys
import time
import json
import http.client
import threading
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

SERVER_READY_TIMEOUT_S = 60


def _wait_for_server(host: str = "127.0.0.1", port: int = 18081, timeout: float = SERVER_READY_TIMEOUT_S) -> bool:
    """Wait until the server health endpoint responds with 200."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=3)
            conn.request("GET", "/health/live")
            resp = conn.getresponse()
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                if data.get("status") == "live":
                    conn.close()
                    return True
            conn.close()
        except Exception:
            pass
        time.sleep(2)
    return False


def test_server_live():
    """Verify the server starts and responds to health checks."""
    assert _wait_for_server(), "Server did not become ready in time"
    conn = http.client.HTTPConnection("127.0.0.1", 18081, timeout=5)
    conn.request("GET", "/health/live")
    resp = conn.getresponse()
    assert resp.status == 200, f"Expected 200, got {resp.status}"
    data = json.loads(resp.read().decode())
    assert data.get("status") == "live", f"Expected live, got {data}"
    conn.close()
    print("  /health/live: 200 OK")


def test_server_ready():
    """Verify the ready endpoint returns runtime state."""
    conn = http.client.HTTPConnection("127.0.0.1", 18081, timeout=5)
    conn.request("GET", "/health/ready")
    resp = conn.getresponse()
    assert resp.status == 200, f"Expected 200, got {resp.status}"
    data = json.loads(resp.read().decode())
    assert "pdca_running" in data, "Missing pdca_running"
    assert "bots_registered" in data, "Missing bots_registered"
    assert "startup_gate_mode" in data, "Missing startup_gate_mode"
    print(f"  /health/ready: pdca={data['pdca_running']} bots={data['bots_registered']} gate={data['startup_gate_mode']}")


def test_cost_tracker_persist():
    """Verify cost tracker persists to SQLite."""
    from ai_sidecar.cost_tracker import CostTracker
    import tempfile, os
    ct = CostTracker()
    ct.record_call(5000, model="test", tier="standard", bot_id="test")
    tmp = tempfile.mktemp(suffix=".db")
    ct.persist(tmp)
    ct2 = CostTracker()
    ct2.restore(tmp)
    assert ct2._daily_tokens == 5000
    os.unlink(tmp)
    print("  cost_tracker persist/restore: OK")


def test_reflex_yaml_rules():
    """Verify reflex YAML rules load correctly."""
    from ai_sidecar.reflex.rule_engine import ReflexRuleEngine
    engine = ReflexRuleEngine(
        workspace_root=Path("."),
        contract_version="v1",
        action_ttl_seconds=120,
    )
    count = engine.load_rules_from_yaml(str(_HERE / "reflex" / "reflex_rules.yaml"))
    assert count > 0, f"Expected >0 YAML rules, got {count}"
    rules = engine.list_rules(bot_id="default")
    assert len(rules) > 15, f"Expected >15 total rules, got {len(rules)}"
    print(f"  reflex rules: {len(rules)} total ({count} from YAML)")


def test_behavior_profiles():
    """Verify all behavior profiles load and can_handle."""
    from ai_sidecar.crewai.agents import get_all_profiles
    profiles = get_all_profiles()
    assert len(profiles) >= 17, f"Expected >=17 profiles, got {len(profiles)}"
    for p in profiles:
        confidence = p.can_handle({"vitals.hp_ratio": 0.3, "combat.aggro_count": 2})
        assert isinstance(confidence, (int, float)), f"{p.agent_id}: can_handle returned non-float"
    print(f"  profiles: {len(profiles)} all have can_handle()")


def test_server_boot():
    """Boot the server, verify health, then shut down. Requires uvicorn."""
    import subprocess, sys, time, json, http.client
    # Start server in background
    proc = subprocess.Popen(
        [sys.executable, "-m", "ai_sidecar.app"],
        cwd=str(_HERE.parent),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # Wait up to 60s for health endpoint
        deadline = time.monotonic() + 60
        ok = False
        while time.monotonic() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", 18081, timeout=3)
                conn.request("GET", "/health/live")
                resp = conn.getresponse()
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    if data.get("status") == "live":
                        ok = True
                    conn.close()
                    break
                conn.close()
            except Exception:
                pass
            time.sleep(3)
        assert ok, "Server did not become ready in 60s"
        # Verify ready endpoint
        conn = http.client.HTTPConnection("127.0.0.1", 18081, timeout=5)
        conn.request("GET", "/health/ready")
        resp = conn.getresponse()
        data = json.loads(resp.read().decode())
        conn.close()
        assert "pdca_running" in data
        print(f"  Server boot: live, pdca={data['pdca_running']}, bots={data['bots_registered']}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    print("  Server shutdown: clean")

if __name__ == "__main__":
    print("=" * 50)
    print("Integration Tests")
    print("=" * 50)

    tests = [("Server boot", test_server_boot),
        ("Cost tracker persist", test_cost_tracker_persist),
        ("Reflex YAML rules", test_reflex_yaml_rules),
        ("Behavior profiles", test_behavior_profiles),
        ("Server live", test_server_live),
        ("Server ready", test_server_ready),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("ALL INTEGRATION TESTS PASSED")
