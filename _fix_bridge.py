#!/usr/bin/env python3
"""Fix aiSidecarBridge.pl: Phase 3 (config override fix) and Phase 6 (anti-detection delays)."""

with open('plugins/aiSidecarBridge/aiSidecarBridge.pl', 'r') as f:
    lines = f.readlines()

# Phase 6: Fix anti-detection delays
for i, line in enumerate(lines):
    if 'my $ANTI_DETECTION_MIN_DELAY_MS = 10;' in line:
        lines[i] = line.replace('= 10;', '= 200;')
        print("Fixed ANTI_DETECTION_MIN_DELAY_MS at line", i+1)
    if 'my $ANTI_DETECTION_MAX_DELAY_MS = 50;' in line:
        lines[i] = line.replace('= 50;', '= 600;')
        print("Fixed ANTI_DETECTION_MAX_DELAY_MS at line", i+1)

# Phase 3: Add _sidecar_set tracking in the "set" command handler
for i, line in enumerate(lines):
    if '# Handle set commands: "set <config_key> <value>"' in line:
        for j in range(i, min(i+15, len(lines))):
            if '$::config{$orig_key} = $set_val;' in lines[j]:
                indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                lines.insert(j+1, indent + "\t# Track that sidecar set this value so _apply_bot_config doesn't override\n")
                lines.insert(j+2, indent + "\t$bridge_cfg{\"_sidecar_set_$orig_key\"} = 1;\n")
                print("Added _sidecar_set tracking at line", j+2)
                break
        break

# Phase 3: Add "unless $bridge_cfg{...}" guards to _apply_bot_config
for i, line in enumerate(lines):
    if 'sub _apply_bot_config {' in line:
        print("Found _apply_bot_config at line", i+1)
        # Map of config key names to their _sidecar_set key
        guards = {
            'attackAuto': "_sidecar_set_attackAuto",
            'attackAuto_inLockOnly': "_sidecar_set_attackAuto_inLockOnly",
            'attackAuto_followTarget': "_sidecar_set_attackAuto_followTarget",
            'attackAuto_onlyWhenSafe': "_sidecar_set_attackAuto_onlyWhenSafe",
            'attackAuto_noMove': "_sidecar_set_attackAuto_noMove",
            'sitAuto_hp_lower': "_sidecar_set_sitAuto_hp_lower",
            'sitAuto_hp_upper': "_sidecar_set_sitAuto_hp_upper",
            'sitAuto_maxDmg': "_sidecar_set_sitAuto_maxDmg",
            'itemsTakeAuto': "_sidecar_set_itemsTakeAuto",
            'itemsTakeAuto_party': "_sidecar_set_itemsTakeAuto_party",
            'itemsGatherAuto': "_sidecar_set_itemsGatherAuto",
            'sellAuto': "_sidecar_set_sellAuto",
            'sellAuto_distance': "_sidecar_set_sellAuto_distance",
            'storageAuto': "_sidecar_set_storageAuto",
        }
        
        for j in range(i, min(i+30, len(lines))):
            for key, guard_key in guards.items():
                # Match: $::config{'key'} = ...  (with single quotes)
                if ("{'%s'}" % key) in lines[j] and 'unless' not in lines[j]:
                    line_stripped = lines[j].rstrip()
                    lines[j] = line_stripped + " unless $bridge_cfg{'" + guard_key + "'};\n"
                    print("  Guarded", key, "at line", j+1)
                    break
        
        break

with open('plugins/aiSidecarBridge/aiSidecarBridge.pl', 'w') as f:
    f.writelines(lines)

print("Done writing aiSidecarBridge.pl")
