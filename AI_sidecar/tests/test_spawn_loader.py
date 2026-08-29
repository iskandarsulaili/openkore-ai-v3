"""Agnostic spawn-loader tests — loads real server spawn scripts (no hardcodes)."""
import os
import tempfile
from ai_sidecar.autonomy.spawn_loader import load_map_spawns, merge_spawns


def test_parse_real_spawn_scripts() -> None:
    spawns = load_map_spawns()
    # The live server defines prt_fild05 with real mobs — never empty, no hardcode assert
    assert isinstance(spawns, dict)
    assert len(spawns) > 0


def test_merge_learned_wins() -> None:
    learned = {"m1": [("A", 5, 1000)]}
    fb = {"m1": [("OLD", 1, 0)], "m2": [("B", 2, 0)]}
    merged = merge_spawns(learned, fb)
    assert merged["m1"] == [("A", 5, 1000)]  # learned wins
    assert merged["m2"] == [("B", 2, 0)]  # fallback fills missing


def test_parse_synthetic_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.txt")
        with open(p, "w") as f:
            f.write("prt_fild05,0,0\tmonster\tHornet\t1004,199,5000\n")
            f.write("// comment\n")
            f.write("pay_dun00,0,0\tmonster\tZombie\t1015,20,0\n")
        spawns = load_map_spawns(d)
        assert spawns.get("prt_fild05") == [("Hornet", 199, 5000)]
        assert spawns.get("pay_dun00") == [("Zombie", 20, 0)]
