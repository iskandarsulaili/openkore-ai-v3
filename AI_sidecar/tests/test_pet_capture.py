"""Tests for the pet-capture knowledge module (server-agnostic)."""
from ai_sidecar.domains.companions import pets


def test_pet_capture_loads_from_server_db():
    """The module should load capturable pets from the server's pet_db.yml."""
    cap = pets.load_pet_capture()
    # This server configures pet capture (107 pets in pet_db.yml).
    assert isinstance(cap, dict)
    assert len(cap) > 0
    # Poring is a known capturable pet with a tame item.
    assert "poring" in cap
    assert cap["poring"]["tame_item"]


def test_get_capture_advice():
    """get_capture_advice returns tame-item data for a capturable mob."""
    advice = pets.get_capture_advice("poring")
    assert advice is not None
    assert advice["tame_item"]
    # Unknown monster -> None.
    assert pets.get_capture_advice("definitely_not_a_mob_xyz") is None


def test_capturable_monsters():
    """capturable_monsters lists the capturable mob names (lowercase)."""
    mobs = pets.capturable_monsters()
    assert isinstance(mobs, list)
    assert "poring" in mobs
