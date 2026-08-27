"""Tests for the durable charstatus.json reader + reflex fact injection."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ai_sidecar.runtime.charstatus import CharStatusReader


def _write_charstatus(tmp_path: Path, bot_id: str, *, seq: int = 1, mtime: float | None = None) -> Path:
    d = tmp_path / "data" / "charstatus"
    d.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in bot_id)
    p = d / f"charstatus_{safe}.json"
    payload = {
        "schema_version": 1,
        "seq": seq,
        "in_game": 1,
        "freshness": "live",
        "identity": {"char_id": "2000011", "name": "TestBotA"},
        "vitals": {"hp": 15, "hp_max": 45, "hp_ratio": 0.333, "status_effects": {"Poison": 5}},
        "stats": {"str": 1, "agi": 1},
        "combat": {"ai_sequence": "attack", "target_id": "1002", "target_name": "Poring", "is_in_combat": 1},
        "environment": {"map_name": "prt_fild08", "is_town": 0, "is_field": 1},
        "skills": {"cooldowns": {"Smite": 3}},
        "inventory": {"items": []},
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    if mtime is not None:
        os_utime = __import__("os").utime
        os_utime(p, (mtime, mtime))
    return p


def test_reader_reads_fresh_file(tmp_path: Path) -> None:
    _write_charstatus(tmp_path, "Local rAthena AI World:testbot99", seq=5)
    r = CharStatusReader(data_dir=tmp_path)
    data = r.get("Local rAthena AI World:testbot99")
    assert data is not None
    assert data["seq"] == 5
    assert data["identity"]["char_id"] == "2000011"
    assert data["vitals"]["status_effects"] == {"Poison": 5}


def test_reader_rejects_stale_file(tmp_path: Path) -> None:
    _write_charstatus(tmp_path, "botA", mtime=time.time() - 120)
    r = CharStatusReader(data_dir=tmp_path)
    assert r.get("botA", max_age_s=30.0) is None


def test_reader_missing_file(tmp_path: Path) -> None:
    r = CharStatusReader(data_dir=tmp_path)
    assert r.get("nobody") is None


def test_reader_caches_by_mtime(tmp_path: Path) -> None:
    p = _write_charstatus(tmp_path, "botB", seq=1)
    r = CharStatusReader(data_dir=tmp_path)
    assert r.get("botB")["seq"] == 1
    # Same mtime → cached, no re-read.
    assert r.get("botB")["seq"] == 1
    # New write with new mtime → re-read.
    time.sleep(0.01)
    _write_charstatus(tmp_path, "botB", seq=2)
    assert r.get("botB")["seq"] == 2


def test_reader_list_bots(tmp_path: Path) -> None:
    _write_charstatus(tmp_path, "botA")
    _write_charstatus(tmp_path, "botB")
    r = CharStatusReader(data_dir=tmp_path)
    bots = r.list_bots()
    assert "botA" in bots
    assert "botB" in bots
