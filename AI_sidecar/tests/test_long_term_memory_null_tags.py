"""Regression: LTM search must tolerate a record whose 'tags' key is present
but null (None) — the real OpenMemory backend returns tags: null for records
stored without tags, and `m.get("tags", [])` only defaults on an ABSENT key,
so `t in None` raised TypeError and search aborted (logged
'long_term_memory_search_failed: argument of type NoneType is not iterable').

This surfaced in every domain test run and on any live search whose store
returned a null-tags doc. Fix: coalesce `data.get('tags') or []` in both the
OpenMemory path (search) and the local JSON fallback (_search_local)."""

from __future__ import annotations

from pathlib import Path

from ai_sidecar.memory.long_term_memory import LongTermMemory


def _null_tags_local_memory(db_dir: Path) -> LongTermMemory:
    """Seed a local memory store with a record whose tags is explicit null and
    one with real tags, then return a LongTermMemory backed by it (no
    OpenMemory backend present in test env -> uses the local fallback)."""
    store = {
        "content": "Thief bug farming spot safe at level 20",
        "category": "farming_spot",
        "tags": None,  # null on disk is the killer
        "importance": 8,
        "timestamp": "2026-09-13T00:00:00+00:00",
        "metadata": {},
    }
    tagged = {
        "content": "alberta airship costs 500 zeny",
        "category": "economy_trend",
        "tags": ["alberta", "airship"],
        "importance": 7,
        "timestamp": "2026-09-13T00:00:00+00:00",
        "metadata": {},
    }
    db = db_dir / "openkore_memory.json"
    import json
    db.write_text(json.dumps([store, tagged]))

    m = LongTermMemory.__new__(LongTermMemory)
    m._lock = __import__("threading").RLock()
    m._memory = None
    m._initialized = True
    m._stats = {"stores": 0, "retrievals": 0, "deletes": 0, "errors": 0}
    return m


def test_search_tolerates_null_tags(tmp_path) -> None:
    m = _null_tags_local_memory(tmp_path)
    m._local_path = Path(tmp_path) / "openkore_memory.json"
    # tag filter must skip the null-tags record without crashing and return the tagged one
    res = m.search(query="alberta", tags=["alberta"], limit=10)
    assert any(r.get("content", "").find("airship") >= 0 for r in res)


def test_search_null_tags_does_not_crash_cross_category(tmp_path) -> None:
    m = _null_tags_local_memory(tmp_path)
    m._local_path = Path(tmp_path) / "openkore_memory.json"
    # no tag filter -> both records scanned, null-tags one must be tolerated
    res = m.search(query="safe", limit=10)
    assert any(r.get("content", "").find("farming spot safe") >= 0 for r in res)
