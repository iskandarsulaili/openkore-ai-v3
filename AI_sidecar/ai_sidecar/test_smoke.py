"""Smoke test for the AI sidecar.
Verifies the server starts, all routes register, and key services initialize.

Run: python -m pytest ai_sidecar/test_smoke.py -v
Or:  python ai_sidecar/test_smoke.py
"""

from __future__ import annotations

import sys
import time
import json
import http.client
from pathlib import Path

# Add parent to path for direct execution
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))


def test_config_loads():
    """Verify config loads with defaults."""
    from ai_sidecar.config import settings
    assert settings.app_name == "openkore-ai-sidecar"
    assert settings.host == "127.0.0.1"
    assert settings.port == 18081
    assert settings.llm_cost_tier in ("off", "economy", "standard", "premium")
    print(f"  Config: host={settings.host} port={settings.port} tier={settings.llm_cost_tier}")


def test_imports():
    """Verify all critical modules import without error."""
    from ai_sidecar.providers.deepseek_adapter import DeepseekAdapter
    from ai_sidecar.providers.prompt_guard import PromptGuard
    from ai_sidecar.autonomy.heuristic_service import HeuristicService
    from ai_sidecar.cost_tracker import CostTracker
    from ai_sidecar.autonomy.pdca_loop import PDCALoop, PDCAConfig
    from ai_sidecar.crewai.agents import get_profile, get_all_profiles, BehaviorProfile
    from ai_sidecar.reflex.rule_engine import ReflexRuleEngine
    from ai_sidecar.api.middleware import AuthMiddleware, add_auth_middleware
    
    # Verify heuristic service works
    hs = HeuristicService()
    result = hs.assess({"hp_ratio": 0.2, "combat.aggro_count": 3})
    assert result.confidence > 0
    assert result.actionable
    print(f"  Heuristic: confidence={result.confidence:.2f} actions={len(result.actions)} top={result.top_domain}")
    
    # Verify cost tracker works
    ct = CostTracker()
    allowed, reason = ct.check(daily_budget_tokens=100000, max_calls_per_hour=30, tier="standard")
    assert allowed
    ct.record_call(500, model="deepseek-v4-flash", tier="standard")
    assert ct.snapshot().daily_tokens_used == 500
    print(f"  CostTracker: daily_tokens={ct.snapshot().daily_tokens_used}")
    
    # Verify behavior profiles
    profiles = get_all_profiles()
    assert len(profiles) >= 17
    profile_ids = [p.agent_id for p in profiles]
    assert "combat" in profile_ids
    assert "safety" in profile_ids
    assert "economy" in profile_ids
    print(f"  Profiles: {len(profiles)} registered")
    
    # Verify YAML rules load
    engine = ReflexRuleEngine(
        workspace_root=Path("."),
        contract_version="v1",
        action_ttl_seconds=120,
    )
    count = engine.load_rules_from_yaml(str(_HERE / "reflex" / "reflex_rules.yaml"))
    assert count > 0
    rules = engine.list_rules(bot_id="default")
    assert len(rules) > 10
    print(f"  Reflex rules: {len(rules)} total ({count} from YAML)")


def test_app_creates():
    """Verify the FastAPI app creates and registers routes."""
    from ai_sidecar.app import create_app
    app = create_app()
    assert app is not None
    assert len(app.router.routes) >= 20
    print(f"  Routes: {len(app.router.routes)} registered ({sum(1 for r in app.router.routes if 'IncludedRouter' in type(r).__name__)} routers)")


if __name__ == "__main__":
    print("=" * 50)
    print("AI Sidecar Smoke Tests")
    print("=" * 50)
    
    tests = [
        ("Config loads", test_config_loads),
        ("Critical imports", test_imports),
        ("App creates", test_app_creates),
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
    print("ALL TESTS PASSED")
