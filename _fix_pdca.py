#!/usr/bin/env python3
"""Fix pdca_loop.py: Phase 1 (config debounce) and Phase 2 (hunting config)."""

with open('AI_sidecar/ai_sidecar/autonomy/pdca_loop.py', 'r') as f:
    lines = f.readlines()

# Phase 1: Add _config_set_cache and _hunting_config_applied after self._cycle_count
for i, line in enumerate(lines):
    if 'self._cycle_count: int = 0' in line:
        indent = ' ' * (len(line) - len(line.lstrip()))
        lines.insert(i+1, indent + '# Config set cache: bot_id -> {config_key: last_set_value}\n')
        lines.insert(i+2, indent + "self._config_set_cache: dict[str, dict[str, str]] = {}\n")
        lines.insert(i+3, indent + '# Hunting config applied flag: bot_id -> bool\n')
        lines.insert(i+4, indent + "self._hunting_config_applied: dict[str, bool] = {}\n")
        print("Added config cache fields")
        break

# Phase 1: Debounce the attackAuto_inLockOnly set
for i, line in enumerate(lines):
    if 'Queue set attackAuto_inLockOnly 0 to ensure bot attacks on arrival' in line:
        indent = ' ' * (len(line) - len(line.lstrip()))
        print("Found inlock block at line", i+1)
        
        # Find the end of this try/except block
        end_idx = i + 1
        while end_idx < len(lines):
            stripped = lines[end_idx].strip()
            if stripped == 'except Exception:' or stripped == 'except:':
                if end_idx + 1 < len(lines) and lines[end_idx+1].strip() == 'pass':
                    end_idx = end_idx + 2
                    break
            end_idx += 1
        
        print("Block ends at line", end_idx)
        
        # Build replacement block
        new_lines = []
        new_lines.append(indent + '# Queue set attackAuto_inLockOnly 0 to ensure bot attacks on arrival\n')
        new_lines.append(indent + '# DEBOUNCED: only queue if not already set to this value\n')
        new_lines.append(indent + '_csc_bot = _cycle_bot_id or "default"\n')
        new_lines.append(indent + '_csc_key = "attackAuto_inLockOnly"\n')
        new_lines.append(indent + '_csc_val = "0"\n')
        new_lines.append(indent + '_csc_cache = getattr(self, "_config_set_cache", {})\n')
        new_lines.append(indent + '_csc_last = _csc_cache.get(_csc_bot, {}).get(_csc_key)\n')
        new_lines.append(indent + 'if _csc_last != _csc_val:\n')
        new_lines.append(indent + '    try:\n')
        new_lines.append(indent + '        _cr_inlock_proposal = ActionProposal(\n')
        new_lines.append(indent + '            action_id=("pro_ro_inlock_%s_%d" % (_csc_bot, int(time.monotonic()*1000))),\n')
        new_lines.append(indent + '            kind="command",\n')
        new_lines.append(indent + '            command="set %s %s" % (_csc_key, _csc_val),\n')
        new_lines.append(indent + '            priority_tier=ActionPriorityTier.tactical,\n')
        new_lines.append(indent + '            source="planner",\n')
        new_lines.append(indent + '            created_at=datetime.now(UTC),\n')
        new_lines.append(indent + '            expires_at=datetime.now(UTC) + timedelta(seconds=120),\n')
        new_lines.append(indent + '            conflict_key="combat_inlock_%s" % _csc_bot,\n')
        new_lines.append(indent + '            idempotency_key="combat_inlock_%s" % _csc_bot,\n')
        new_lines.append(indent + '            metadata={"source": "pro_ro_player", "reason": "Enable attack outside lockMap", "bot_id": _csc_bot},\n')
        new_lines.append(indent + '        )\n')
        new_lines.append(indent + '        _cr_aq.enqueue(_csc_bot, _cr_inlock_proposal)\n')
        new_lines.append(indent + '        # Update cache\n')
        new_lines.append(indent + '        if _csc_bot not in _csc_cache:\n')
        new_lines.append(indent + '            _csc_cache[_csc_bot] = {}\n')
        new_lines.append(indent + '        _csc_cache[_csc_bot][_csc_key] = _csc_val\n')
        new_lines.append(indent + '    except Exception:\n')
        new_lines.append(indent + '        pass\n')
        
        lines[i:end_idx] = new_lines
        print("Replaced", end_idx - i, "lines with", len(new_lines), "lines")
        break

# Phase 2: Add hunting config application INSIDE the route try block
for i, line in enumerate(lines):
    if '_cr_aq.enqueue(_cycle_bot_id or ' in line and '_cr_route_proposal)' in line:
        indent = ' ' * (len(line) - len(line.lstrip()))
        print("Found route enqueue at line", i+1)
        
        # Find the matching except: pass that closes this try
        for j in range(i+1, min(i+10, len(lines))):
            stripped = lines[j].strip()
            if stripped == 'except Exception:' or stripped == 'except:':
                if j+1 < len(lines) and lines[j+1].strip() == 'pass':
                    # Insert hunting config BEFORE this except
                    hunting_block = []
                    hunting_block.append(indent + '# Apply hunting config ONCE per bot (not every cycle)\n')
                    hunting_block.append(indent + '_hc_bot = _cycle_bot_id or "default"\n')
                    hunting_block.append(indent + '_hc_applied = getattr(self, "_hunting_config_applied", {})\n')
                    hunting_block.append(indent + 'if not _hc_applied.get(_hc_bot):\n')
                    hunting_block.append(indent + '    try:\n')
                    hunting_block.append(indent + '        _hc_configs = [\n')
                    hunting_block.append(indent + '            ("lockMap", _cr_map if _cr_map else "prt_fild08"),\n')
                    hunting_block.append(indent + '            ("attackAuto", "2"),\n')
                    hunting_block.append(indent + '            ("attackAuto_inLockOnly", "1"),\n')
                    hunting_block.append(indent + '            ("route_randomWalk", "2"),\n')
                    hunting_block.append(indent + '            ("teleportAuto_minAggressives", "5"),\n')
                    hunting_block.append(indent + '            ("teleportAuto_hp", "30"),\n')
                    hunting_block.append(indent + '            ("teleportAuto_minAggressivesInLock", "8"),\n')
                    hunting_block.append(indent + '        ]\n')
                    hunting_block.append(indent + '        for _hc_key, _hc_val in _hc_configs:\n')
                    hunting_block.append(indent + '            _hc_prop = ActionProposal(\n')
                    hunting_block.append(indent + '                action_id="hunt_cfg_%s_%s_%d" % (_hc_bot, _hc_key, int(time.monotonic()*1000)),\n')
                    hunting_block.append(indent + '                kind="command", command="set %s %s" % (_hc_key, _hc_val),\n')
                    hunting_block.append(indent + '                priority_tier=ActionPriorityTier.tactical, source="planner",\n')
                    hunting_block.append(indent + '                created_at=datetime.now(UTC), expires_at=datetime.now(UTC)+timedelta(seconds=300),\n')
                    hunting_block.append(indent + '                conflict_key="hunt_cfg_%s_%s" % (_hc_key, _hc_bot),\n')
                    hunting_block.append(indent + '                idempotency_key="hunt_cfg_%s_%s_%s" % (_hc_key, _hc_val, _hc_bot),\n')
                    hunting_block.append(indent + '                metadata={"source": "hunting_config", "reason": "Hunting: %s=%s" % (_hc_key, _hc_val), "bot_id": _hc_bot},\n')
                    hunting_block.append(indent + '            )\n')
                    hunting_block.append(indent + '            _cr_aq.enqueue(_hc_bot, _hc_prop)\n')
                    hunting_block.append(indent + '        # Mark as applied\n')
                    hunting_block.append(indent + '        if _hc_bot not in _hc_applied:\n')
                    hunting_block.append(indent + '            _hc_applied[_hc_bot] = True\n')
                    hunting_block.append(indent + '    except Exception:\n')
                    hunting_block.append(indent + '        pass\n')
                    
                    lines[j:j] = hunting_block
                    print("Inserted hunting config block at line", j)
                    break
            elif stripped and len(lines[j]) - len(lines[j].lstrip()) < len(indent):
                print("WARNING: Could not find matching except for route block")
                break
        break

with open('AI_sidecar/ai_sidecar/autonomy/pdca_loop.py', 'w') as f:
    f.writelines(lines)

print("Done writing pdca_loop.py")
