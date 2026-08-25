

## T. Session stability FIXED (2026-08-25, verified live)
- The one remaining blocker (bot enters map -> route calc -> keepalive starvation
  -> RAW 60s stall drop -> char-select retry loop) is RESOLVED. Bot now stays
  in-game indefinitely (verified 20+ min, drop count frozen, keepalive + sync
  replies flowing both ways every 12s).
- T1. Root cause 1: C A* timeout was checked every 100th pop. On a fast machine
  100 pops complete in <1ms, so the 100ms time_max NEVER fired and pathStep ran
  the FULL width*height expansions (each with up to 1024-iteration sift-ups) =
  minutes of blocked main loop -> keepalive starvation -> session drop.
  FIX: timeout checked EVERY pop (src/auto/XSTools/PathFinding/algorithm.cpp).
- T2. Root cause 2: CalcPath_init called calloc(width*height) with no upper
  bound; garbage dims (corrupt field) -> huge calloc -> NULL -> memset(NULL)
  hung the whole bot. FIX: dims validated (positive, sane bound, product
  <= 4M cells) + NULL-checked at XS reset AND C init; failures set
  session->failed, pathStep returns -1 cleanly.
- T3. Root cause 3: Task::Route (same-map path, noMapRoute=1) had NO keepalive
  flush; flushKeepalive only existed in Task::CalcMapRoute. FIX: _flushKeepalive
  added to Task::Route iterate (CZ_SYNC when ai_sync due).
- T4. Sidecar (AI_sidecar, uvicorn :18081) was DOWN all night (crashed 08-24
  13:44) -> bot had no AI driver -> idle/looping. RESTARTED via start-sidecar.sh;
  bot re-registered (bot_count:1, status:up). The bot is now productive
  (walking, talking to Academy Receptionist, following sidecar commands).
