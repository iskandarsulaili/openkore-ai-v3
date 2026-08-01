# COMPLETENESS TRACKER — openkore-ai-v3

Goal: every feature fully implemented, wired, and verified live. Zero
stub/placeholder/pending/todo/fixme/dormant/incomplete. Adapts to
rathena-ai-world; nothing is trimmed.

Status legend:
- [x] DONE (implemented + wired + verified + tests green)
- [ ] OPEN (incomplete / dormant / needs wiring)

Commit chain verified at start: a62ea37cb (0 ahead origin/master, clean).

────────────────────────────────────────────────────────────────────────────
## A. Server-adaptation / live-progression layer

- [x] Ref A1: cold-start level 1-5 → Cryptura Academy field (prt_fild08)
      (exp-granting Porings/Lunatics; knife-dropping).  [8b20c4cc4]
- [x] Ref A2: char-login explicitly sent on every char-select with a valid
      slot (bot no longer loops at char-select).      [f9c23967c]
- [x] Ref A3: level-1 bot registers at Academy Receptionist (iz_ac01 100,39)
      for starter gear (Novice_Knife 1243, 300 Novice_Potion). [a62ea37cb]
- [x] Ref A4: ExperienceDB.best_action() implemented (was missing →
      AttributeError every cycle).                   [dd006fbf4]
- [x] Ref A5: int_land → prt_fild08 navigation dead-end. Root cause: int_land
      is the "Secluded Island" intro; char_athena.conf start_point +
      #ship_out set the save point there, so login/respawn lands at
      int_land(77,101), and OpenKore cannot path to prt_fild08 from it (no
      portal). Fixed: a level-1 bot on int_land* walks to the WARPNPC
      `#intro_to_izlude` at (49,57) and sails to Izlude (`move 49 57` +
      `talk resp 1`), and lockMap prt_fild08 is suppressed while stranded.
      Regression test added.  [this batch]
      Verify: bot leaves Secluded Island, continues to academy/prt_fild08.

────────────────────────────────────────────────────────────────────────────
## B. Preserved pending items (from prior task list)

- [ ] Pres B1: heuristic_service.py:1103 (old) hardcoded `prontera`
      in town-data region → drive from GameKnowledgeService (maps/npc
      registry from the live server) instead of a hardcoded constant.
- [ ] Pres B2: flake — harden route_churn_count test with a deterministic
      barrier + diagnostic dump (test was flaky).
- [ ] Pres B3: model-router — validate DEFAULT_POLICY_RULES targets exist
      in registered providers (avoid route to nonexistent provider/model).
- [ ] Pres B4: registry-remove — delete domains/registry.py (verify zero
      test imports first). [Left-as-documented earlier; re-evaluate — if
      it's truly dead AND no imports, remove; if it has callers, wire it.]
- [ ] Pres B5: abstract — NotImplementedError x4 → abc.ABC + @abstractmethod,
      then verify concrete subclasses implement them (no bare
      NotImplementedError; make it enforced + complete).

────────────────────────────────────────────────────────────────────────────
## C. Sweep catalog (findings from the completeness scan)
(populated as discovered; each marked DONE with commit after verify)
