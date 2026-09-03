# COMPLETENESS — Server-side multi-session desync burst (2026-09-03)

## Goal
Fix the map-server mid-session desync burst that kicks the bot (and multiple sessions at once) after login, so the bot stays in-game and completes the job change. Full completeness: zero mock/stub/dormant, reconcile not trim, prove with benchmark.

## Context
- Bot TestBotA 0x0436 map-login is FIXED (logs in successfully, gets "You are now in the game").
- Reconnect loop is now a MID-SESSION DESYNC: map-server logs `unsupported packet 0xXXXX, 1027 bytes` hitting MULTIPLE sessions simultaneously (17/40/41/43 at 11:59:16-18).
- Real players (ChinkeE, Mase, CloudNine, Psalm23) NOT affected — they play fine.
- This is a rathena-AI-world SERVER-SIDE issue (map-server recv/broadcast handling).

## Checklist

### Round A — Diagnose the desync burst
- [x] Identify the packet that corrupts the stream (1027-byte / 762-byte bursts)
- [x] Determine if it's a broadcast (server→all) or per-session recv issue — it's a per-session recv: playit tunnel edge forwards TLS ClientHello probes to the map port
- [x] Check if the desync correlates with a specific event — it's infrastructure (TLS probes), not game events
- [x] Capture the exact corrupting bytes (tcpdump on 5121 during a burst) — 1535-byte TLS ClientHello (0x16 0x03, SNI 17.ip.sa.play.gg) from playit edge 127.30.46.227
- [x] Check map-server build state (running binary vs source) — running binary on valid inode

### Round B — Fix the server-side desync
- [x] Root-cause the multi-session corruption — playit tunnel edge forwards TLS ClientHello probes to map port 5121; map-server reads first 2 bytes as packet ID → desync
- [x] Implement the fix (map-server recv/broadcast handling) — TLS-probe guard in clif_parse: drop unauthenticated connections whose first bytes are 0x16 0x03 (TLS ClientHello)
- [x] Rebuild map-server — DONE (binary has guard, verified via strings)
- [x] Restart map-server — DONE 14:14:27 (real players logged off; monitor auto-restarted)
- [x] Verify bot stays in-game for sustained period (benchmark: 30+ min no desync) — VERIFIED: zero `unsupported packet`/`0x05fc`/`0x0436` desyncs since restart; TLS probes from 127.30.46.227 dropped cleanly (log: "TLS probe on map port ... dropping")

### Round C — Job change completion
- [ ] Bot reaches guild (alberta_in / payon_in02)
- [ ] Bot talks to guild NPC + completes job change
- [ ] Verify class != 0 in DB (TestBotA no longer novice)
- [ ] Verify job-change skill-set fires (macro-agent)

### Round D — Full-stack completeness sweep
- [ ] All data files present + loaders verified (skill_db, randomopt, item, mob, map_index)
- [ ] monster_db loader on Renewal (re) DB — 2675 monsters
- [ ] Full test suite green
- [ ] No mock/stub/dormant in the touched paths
