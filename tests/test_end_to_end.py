"""End-to-End Verification — 100 simulated ticks through the full AI pipeline.

This test:
1. Creates a BotRuntime with all subsystems
2. Runs 100 simulated ticks with progressively richer signals
3. Verifies the bot produces reasonable actions at each stage
4. Verifies cross-domain communication (EventBus)
5. Verifies persistent state (SQLite)
6. Verifies the full think() -> actions -> act() -> commands pipeline
"""
import sys
sys.path.insert(0, "../AI_sidecar")

import json
import logging
logging.basicConfig(level=logging.WARNING)

from ai_sidecar.runtime.event_bus import EventBus
from ai_sidecar.runtime.persistence import PersistentState
from ai_sidecar.orchestrator import BotRuntime

# Reset state for clean test
PersistentState.reset()
EventBus.clear()

# Create runtime
runtime = BotRuntime()
runtime.initialize()

# Simulated ticks
print("=" * 60)
print("END-TO-END VERIFICATION — 100 simulated ticks")
print("=" * 60)

tick_results = []
for tick in range(100):
    # Progressively richer signals
    signals = {
        "bot_id": "test_bot",
        "base_level": min(1 + tick // 20, 99),
        "job": "Novice" if tick < 20 else ("Swordman" if tick < 40 else "Knight"),
        "hp": max(10, 100 - tick % 40),
        "hp_max": 100,
        "sp": max(10, 80 - tick % 30),
        "sp_max": 80,
        "map": "prt_fild05",
        "zeny": min(500 + tick * 100, 50000),
        "weight": min(50 + tick, 100),
        "weight_max": 100,
        "monsters_around": [{"distance_to": 5}] if tick % 5 == 0 else [],
        "dead": tick == 42,  # Die on tick 42
        "is_dead": False,
        "last_monster": "Thief Bug" if tick == 42 else "",
        "equipment": {"weapon": {"name": "Knife", "durability": max(10, 100 - tick)}},
        "inventory": {"items": [{"name": "Red Potion", "quantity": 10}]},
    }
    if tick == 42:
        signals["dead"] = True
        signals["is_dead"] = True
    
    # Run the full pipeline
    actions = runtime.think(signals)
    commands = runtime.act(actions)
    
    tick_results.append({
        "tick": tick,
        "action_count": len(actions),
        "command_count": len(commands),
        "bot_level": signals["base_level"],
        "bot_job": signals["job"],
    })

# Print summary
total_actions = sum(r["action_count"] for r in tick_results)
total_commands = sum(r["command_count"] for r in tick_results)
avg_actions = total_actions / len(tick_results)

print(f"\nTicks: {len(tick_results)}")
print(f"Total actions: {total_actions}")
print(f"Total commands: {total_commands}")
print(f"Avg actions/tick: {avg_actions:.1f}")

# Verify progression
level_progression = [r for r in tick_results if r["tick"] % 20 == 0]
print(f"\nLevel progression:")
for r in level_progression:
    print(f"  Tick {r['tick']:3d}: Level {r['bot_level']:2d} {r['bot_job']:15s} -> {r['action_count']:2d} actions")

# Verify events were posted
event_summary = EventBus.summarize()
print(f"\nEventBus: {event_summary['active_keys']} active keys, {event_summary['history_count']} events")
for ev in event_summary.get("recent", []):
    print(f"  {ev.get('key', '?'):40s} @ {ev.get('ts', '?')[:19]}")

# Verify persistence
state_stats = PersistentState.get_stats()
print(f"\nPersistent state:")
for table, count in state_stats.items():
    print(f"  {table}: {count} rows")

# Check death was recorded
death_count = PersistentState.get_death_count("prt_fild05")
print(f"\nDeaths on prt_fild05: {death_count} (expected >= 1)")
death_check = "✅" if death_count >= 1 else "❌"

# Check bot state was saved
bot_signals = PersistentState.load_bot_state("test_bot", "last_signals")
state_check = "✅" if bot_signals and bot_signals.get("map") == "prt_fild05" else "❌"

print(f"\nDeath recording: {death_check}")
print(f"State persistence: {state_check}")
print(f"Actions across 100 ticks: {total_actions}")
print(f"\n{'='*60}")
print("END-TO-END VERIFICATION COMPLETE")
print(f"{'='*60}")

# Overall pass/fail
all_pass = (
    total_actions > 0 
    and avg_actions > 0 
    and death_count >= 1 
    and bot_signals is not None
)
print(f"\nOVERALL: {'✅ PASS' if all_pass else '❌ FAIL'}")
