"""Regression test: ExperienceDB must construct, persist, and load.

Root cause: SidecarSettings had NO data_dir field, but lifecycle
create_runtime references settings.data_dir — the AttributeError made
ExperienceDB construction fail at every startup, silently falling back to
the in-memory ExperienceDatabase (experience_load_failed + 0 entries every
boot, zero persistent EXP tracking). ExperienceDB.record() was also a
pass-stub (seed data vanished). This test locks the fixed behavior.
"""

from __future__ import annotations

import sqlite3
import time

from ai_sidecar.config import settings
from ai_sidecar.experience_db import ExperienceDB, ExperienceEntry


def test_settings_has_data_dir() -> None:
    assert hasattr(settings, "data_dir")
    assert settings.data_dir  # non-empty


def test_experience_db_constructs_and_loads(tmp_path) -> None:
    db = ExperienceDB(str(tmp_path / "exp.sqlite"))
    assert hasattr(db, "load")
    assert db.load(str(tmp_path / "exp.sqlite")) == 0


def test_experience_db_record_persists_seed_entry(tmp_path) -> None:
    db = ExperienceDB(str(tmp_path / "exp.sqlite"))
    entry = ExperienceEntry(
        bot_id="bot:seed", timestamp=time.time(), context_type="map",
        map_name="prt_fild05", monster_name="Poring", role="hunter",
        action_taken="attack", success=True, reward=10.0, details={},
    )
    db.record(entry)
    conn = sqlite3.connect(str(tmp_path / "exp.sqlite"))
    rows = conn.execute(
        "SELECT bot_id, map_name FROM exp_snapshots"
    ).fetchall()
    conn.close()
    assert rows == [("bot:seed", "prt_fild05")]
    assert db.load(str(tmp_path / "exp.sqlite")) == 1


def test_experience_db_record_exp_snapshot_prunes_to_cap(tmp_path) -> None:
    from ai_sidecar.experience_db import ExpSnapshot

    db = ExperienceDB(str(tmp_path / "exp.sqlite"), max_snapshots_per_bot=3)
    for i in range(6):
        db.record_exp_snapshot(ExpSnapshot(
            bot_id="bot:x", base_level=1, job_level=0,
            base_exp=i, job_exp=0, zeny=0, map_name="prt_fild05",
            timestamp=time.time() + i,
        ))
    conn = sqlite3.connect(str(tmp_path / "exp.sqlite"))
    n = conn.execute("SELECT COUNT(*) FROM exp_snapshots WHERE bot_id='bot:x'").fetchone()[0]
    conn.close()
    assert n == 3, f"cap 3 must hold, got {n}"
