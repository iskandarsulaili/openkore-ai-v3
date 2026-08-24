# ZERO_INTERVENTION_COMPLETENESS — CLIENT-SIDE BLOCKER FIXES (openkore-ai-v3 only)

Goal: openkore-ai-v3 bots route + farm on RAW over the internet, zero human intervention,
without touching/modifying RAW servers. openkore-ai-v3 adapts agnostically to RAW.

## VERIFIED DONE
- [x] C0 3 chars on testbot99 (2000011), full internet path (login 46019 / char 21544 / map 53378)
- [x] C1 `messageIDEncryption 0` — login packet stream aligned (client was encrypting, RAW is not).
      Bot now enters izlude; map-server: "TestBotA logged in", Actor Get Info for its own char.
- [x] C2 `.dist`/`.weight` field data regenerated from client GAT (bit-0 walkable, paths walkable)
- [x] C3 `portalCompile 1` — killed headless "Compile portals?" stdin deadlock
- [x] C4 `ignoreInvalidLogin 1` — killed headless "Enter password again" stdin deadlock

## IN PROGRESS — ROUTE-CALC FREEZE (the remaining blocker)
Symptom: bot enters izlude, queues lockMap route to prt_fild05, then AI loop freezes:
- no periodic keepalive (0360 Sync every ~12s) → RAW stall_time (~60s) drops session
- bot spins CPU, log stops at "Calculating lockMap route to prt_fild05"
- NOT the login packet (fixed), NOT field data (walkable), NOT portals (chain complete)

Hypotheses to verify:
- [ ] C5 CalcMapRoute maxTime honored? Infinite loop if undef/never exceeded
- [ ] C6 Does a route task block the AI main loop's processMisc (keepalive)?
- [ ] C7 ai_route_calcRoute timeout loaded in timeouts.txt?
