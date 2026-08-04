"""Tests for the achievement knowledge module (server-agnostic)."""
from ai_sidecar.domains.progression import achievements


def test_achievements_load():
    """The module should load achievements from the server/bundled DB."""
    ach = achievements.load_achievements()
    assert isinstance(ach, list)
    assert len(ach) > 0


def test_get_achievement_by_id():
    """get_achievement_by_id returns a known achievement (110000 = Poring eating)."""
    a = achievements.get_achievement_by_id(110000)
    assert a is not None
    assert a["name"]
    # Unknown id -> None.
    assert achievements.get_achievement_by_id(99999999) is None


def test_achievement_groups():
    """Distinct achievement groups exist and include Adventure/Battle."""
    groups = achievements.achievement_groups()
    assert isinstance(groups, list)
    assert "Adventure" in groups
    assert "Battle" in groups


def test_total_score_and_count():
    """Total score and count are positive and consistent."""
    assert achievements.achievement_count() > 0
    assert achievements.total_achievement_score() > 0
