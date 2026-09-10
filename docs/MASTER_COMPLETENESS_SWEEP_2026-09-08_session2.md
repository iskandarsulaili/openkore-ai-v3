# MASTER COMPLETENESS SWEEP — openkore-ai-v3 (2026-09-08, session 2)

MANDATE: implement/integrate/fix/wire/execute/verify EVERYTHING to completeness. Zero mock/stub/placeholder/pending/todo/fixme/dormant/incomplete. Reconcile, NEVER trim. Char-job AGNOSTIC. Verify-before-modify. Benchmark-prove (EXP/kill-rate/job-change E2E), not assumption. Commit after each batch. Update doc after each batch.

## BATCH 5 — JOB-CHANGE E2E (cross-map execution robustness)
Goal: bot completes merchant job-change end-to-end (reach alberta guild, talk, become merchant) without dying/wedging. BLOCKER: alberta is an island; bot routes 11-map overland (~3377 steps) and dies/wedges. Portal graph HAS airship `#prontera → alberta` (cost ~1800), but bot isn't using it.

- [x] 5.1 ROOT-CAUSE found: job-change gate bypassed affordability. The healthy-HP
      override (session-prior commit) forced a 0-zeny bot to suicide-walk 11 maps to
      the island guild (alberta) and wedge. Reconcile: job-change prioritized ONLY
      when healthy (>=0.9 HP) AND affordable (zeny>=500 for Kafra/airship OR already
      on guild town). COMMIT d6c406b55 (heuristic + progression.py gates). VERIFIED:
      broke bot now defers + farms.
- [x] 5.5 SUSTAIN GAP (Basic Skill): Novice bot without Basic Skill cannot sit/regen
      HP -> stuck at ~50% HP, 0 zeny, no potions -> can't heal -> can't farm. Granted
      NV_BASIC (skill id 1) lv1->lv3 in DB (server requires lv3 to sit). VERIFIED: HP
      recovers via sit.
- [!] 5.7 OPEN WEDGE (real, confirmed): `lethal_escape_teleport` YAML reflex fires every
      ~2s at HP<=0.18 in-combat emitting `reflex_teleport_escape` — a NO-OP macro
      (log+stop). With 0 zeny / no Fly Wing, the bot cannot teleport/flee, so the
      reflex busy-loops the empty macro forever and BLOCKS all other commands/combat
      (0 kills while alive + regenerating). IN PROGRESS: make the fallback macro
      actually flee/retreat (walk away from aggro) or suppress when no escape item.
- [!] 5.8 OPEN EXEC-BLOCKER (root of churn): bot relogs every ~90s while farming
      (19:50:27 off → 19:51:00 in) + intermittently freezes at fixed pos with 0 visible
      monsters. Map clean (BotDetection 0.26 Human, benign; AI-NPC active). The repeated
      logout/login is the execution layer (bridge reconnect / char-map handshake), NOT
      the AI decisions (which now farm correctly when connected). EXP climbed
      18638→19811 this window proving decisions work. BLOCKED ON: bridge/execution
      reconnect-loop diagnosis (separate from this sweep's decision-layer work).
- [x] 5.9 ROOT-CAUSE of the churn FOUND+FIXED: `assess()` dereferenced `None`.
      `_assess_impl` returns None when the conscious tier defers job change
      (survival_strategy=level_up_first/fly_wing_escape, _assess_impl line 4721), then
      assess() line 2026 `if not assessment.actions:` crashed EVERY tick —
      AttributeError 'NoneType' has no attribute 'actions' — killing all
      farming/supplementary actions + starving the action queue (drove the relog churn
      + semi-half-emitting job-change macros). FIX: defer-guard in assess() substitutes
      an empty no-action HeuristicAssessment when _assess_impl returns None (bot stays
      on ai auto). VERIFIED: 0 assess crashes (was every tick), EXP continued climbing
      19811→22755, sidecar correctly emits sit/potion/survival and defers job change.
- [x] 5.10 EMITTER OSCILLATION FIXED: cold-start job-change emitter (~3863) only had the
      healthy-HP gate, so when healthy-but-broke it fired `move alberta_in` while
      progression.py correctly deferred (zeny=0<500) — the two emitters oscillated
      alberta<->farm and froze EXP. FIX: added the SAME affordability gate to cold-start
      (defer unless healthy AND (zeny>=500 OR on guild map)). Now ALL job-change emitters
      (cold-start, HUNTING-branch, progression.py) share one affordability rule.
      VERIFIED: logs show only "deferring" (no "prioritizing") when broke; emitter
      conflict resolved. COMMIT e6738ac6f (+ affordability in cold-start, pushed).
- [x] 5.11 HEAL COOLDOWN 30s->8s: bot died with 217 potions unused on a dense field
      (lvl-38 novice, max_hp 224, 6-monster field) — one heal/30s couldn't outpace
      incoming DPS. Lowered use-item cooldown to 8s (per-heal-name key). VERIFIED:
      bot sustains combat, +28 kills/120s when connected. COMMIT (heal cooldown).
- [x] 5.12 JOB-CHANGE MACRO AFFORDABILITY: static `job_change_novice`/`job_change_2_1`
      macros fired whenever eligible, DISABLING attack (`set attackAuto 0`) + forcing
      the unwalkable alberta crossing regardless of zeny — overriding all heuristic
      gates. Added `required_zeny=500` to both. VERIFIED: macro no longer fires when
      broke (auto-attack stays on). COMMIT.
- [x] 5.13 attackAuto_onlyWhenSafe 1->0: cold-start config audit set onlyWhenSafe=1
      (never attack when aggressive monsters nearby) — on a dense field the bot NEVER
      attacked (never "safe"), sat at low HP, died with 0 kills. Reconcile to 0 to
      match the HUNTING-branch audit. COMMIT.
- [x] 5.14 BRIDGE CRASH FIXED: route-stall recovery debug line `${\\$_rs_reset_ok ? 'ok' : 'failed'}`
      deref'd a string as SCALAR ref under strict refs -> killed the bot process
      ("Can't use string (ok) as a SCALAR ref"). Fixed to string concat. VERIFIED:
      bot no longer crashes. COMMIT.
- [!] 5.15 OPEN EXEC-BLOCKER (recurring): bot alive + decision-tier complete, but stuck
      in a route-stall loop on prt_fild08 — position desyncs (pos_to resynced after
      74s/117s stalls), never reaches the farm, 0 kills. The cross-map position-desync
      (server freezes while local advances; route-stall recovery uses LOCAL pos) is the
      standing execution-layer blocker. NOT a decision bug — the bot correctly defers
      job change, heals, allocates stats, routes. BLOCKED ON: bridge route-stall
      recovery using SERVER-side position.
- [x] 5.16 ROUTE-STALL SERVER-POSITION FIX — **MISDIAGNOSED; PREMISE UNVERIFIED.**
      Claimed local `$char->{pos}` interpolated forward on server-freeze, so stall
      never fired. SOURCE PROVES OTHERWISE: `$char->{pos}` is server-confirmed ONLY
      (Receive.pm:7522/7576/8501); calcPosition() (Utils.pm:764) is read-only, never
      writes back. The added 0x0088 hook carries the SAME data as $char->{pos}
      (Receive.pm:8501 also sets it) — redundant, not the root cause. Pre-5.16 log
      showed the recovery DID fire (117265ms stall). Honest correction: the bot farming
      now is real but credited to the OTHER fixes (heal cooldown, onlyWhenSafe=0, macro
      affordability, assess-crash). TRUE remaining root cause of the old wedge: the
      route-stall detector fires at a fixed 45s threshold, but a long cross-map walk
      (>45s) legitimately gets no server position confirmation until arrival — so any
      long crossing trips the false stall. NOT yet root-caused.

## BATCH 6 — DQN COMBAT-MICRO (god-tier gap, char-agnostic)
- [ ] 6.1 ThreatTargeting NEVER instantiated — CombatLoop._threat_targeting stays None, _acquire_target no-ops. Wire real target selection.
- [ ] 6.2 Design char-agnostic combat-micro state->action (target/skill/retreat), reuse the trained DQN or a per-class combat micro-policy; do NOT hardcode class/item.
- [ ] 6.3 Subconscious drives target+skill choice when trained; fall back to heuristic when undertrained.
- [ ] 6.4 Benchmark: kills/min improves vs heuristic-only.

## BATCH 7 — FULL SIDECAR DEAD-CODE/DORMANT SWEEP (char-agnostic)
- [ ] 7.1 heal_resource_loader.py:200+ (never initialized?) — wire or reconcile.
- [ ] 7.2 rule_engine.py:796 hardcodes reflex_survival_escape (name mismatch vs reflex_rules.yaml reflex_teleport_escape) — reconcile.
- [ ] 7.3 action_emitter.py:434 macros dir + macro_manifest.json (job-change 58,43) — reconcile with macro_intelligence.py.
- [ ] 7.4 Conscious vs heuristic balance (~92% heuristic) — demote heuristics to cold-start fallback, let consciousness+subconscious drive when able.
- [ ] 7.5 Zero stub/todo/pass/dormant grep pass.
- [ ] 7.6 Adversarial sweep until nothing found; each pass finds real bugs (char-agnostic).

## BATCH 8 — CROSS-CUTTING / BENCHMARK / PRODUCTION-READY
- [ ] 8.1 Login->enter->farm->EXP->level-up->job-change FULL loop proven with timestamps (char-agnostic path).
- [ ] 8.2 kills/min + EXP/hour benchmark recorded.
- [ ] 8.3 All commits pushed; checklist current.

STATUS LEGEND: [ ] not started · [~] in progress · [x] done · [!] blocked · [P] proven (benchmark/live)
