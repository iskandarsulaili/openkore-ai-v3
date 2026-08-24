# ZERO_INTERVENTION_COMPLETENESS — MASTER CHECKLIST

Goal: openkore-ai-v3 fully complete, production-ready, zero-human-intervention,
player bot + capacity node, Windows launcher integration. Zero mocks/stubs/dormant.
Reconcile, never trim. Client-side (openkore-ai-v3) fixes adapt to RAW without
modifying RAW unless explicitly authorized.

## PHASE A — PLAYER BOT (blocker: reach stable farming)
- [x] A1. messageIDEncryption 0 — login packet stream aligned (bot enters map)
- [x] A2. portalCompile 1 — killed "Compile portals?" stdin deadlock
- [x] A3. ignoreInvalidLogin 1 — killed "Enter password again" stdin deadlock
- [x] A4. 0840/0841 accessible-map char-select handshake registered (bot waits, no timeout)
- [ ] A5. VERIFY char-select "all maps not ready" root cause (peer-host flap vs central ownership)
- [ ] A6. Fix char-select retry so bot enters map when central owns maps
- [ ] A7. Post-map-entry stall: register any remaining missing 20250604 packets (full-class sweep)
- [ ] A8. LIVE verify: bot enters izlude, stays >3min, gains EXP (outcome proof)

## PHASE B — CAPACITY NODE (peer host / relay / in-game mesh)
- [ ] B1. Design doc: player bot vs capacity-node roles, staging
- [ ] B2. map-server.exe peer-host bundle + ephemeral DB creds (RAW's model)
- [ ] B3. P2P relay registration + honest capacity
- [ ] B4. In-game mesh (WebRTC) movement relay 0x035F/0x0361
- [ ] B5. IPv6-first + UDP paths
- [ ] B6. Verify peer-host map serving + mesh

## PHASE C — WINDOWS LAUNCHER INTEGRATION
- [ ] C1. Build openkore-ai-v3 for Windows
- [ ] C2. dist/ + manifest entries (PlayRAW-like)
- [ ] C3. Launcher option to config + run openkore-ai-v3
- [ ] C4. Verify on Windows

## PHASE D — FULL-CLASS SWEEPS (completeness)
- [ ] D1. Full RAW active-block packet diff vs our tables (no future "Unknown switch")
- [ ] D2. Full test suite green (394 tests)
- [ ] D3. Adversarial sweep: no dead code / no dormant paths
- [ ] D4. Update all docs/checklists, commit after each batch
