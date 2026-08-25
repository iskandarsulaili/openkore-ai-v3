# Pathfinding / Navigation / Goal-Planning Research Digest
**For openkore-ai-v3 autonomous rAthena farming bot**
Date: 2026-08-25. Sources verified via arxiv API (IDs cited).

## 1. Classical low-level engine — KEEP (already proven)

- **JPS** (Harabor & Grastien 2011, canonical, no arxiv ID) — beats A* on 8-connected
  uniform-cost grids (our case) by pruning via jump points.
- **JPS4** (arXiv:2501.14816) — JPS variant for 4-connected grids.
- **Reducing Redundant Work in JPS** (arXiv:2306.15928, 2023) — fixes JPS pathological
  rescanning/suboptimal-node expansion. DIRECTLY relevant: our "spin at high CPU" on
  complex maps is this pathology.
- **HPA*** (Botea et al. 2004, JAIR) — hierarchical abstraction; proven way to cut
  large-map A* cost.
- **D* Lite** (Koenig & Likhachev 2002) + **LPA*** (2004) — incremental replan; only
  needed if obstacles change per-step (our maps are static).
- **Weighted A*** (Pohl 1970), **ARA*** (Likhachev 2003; survey arXiv:2310.02346) —
  bounded-suboptimal anytime.
- **A-MHA*** (arXiv:2508.21637, 2025) — anytime + multiple heuristics backed by an
  admissible one; "return fast, refine later" budget.

## 2. Grid optimization for games/bots — ADOPT (proven, low risk)

- **Any-angle / Theta*** (Nash et al. 2007; survey arXiv:2310.02346) + Edge N-Level
  Sparse Visibility Graphs (arXiv:1702.01524) — straighter paths, fewer nodes.
- **Path smoothing: string-pulling / Funnel** (Diggelen & Overmars, standard) — cheap
  LOS-shortcut post-pass; high value for jittery diagonal RO paths.
- **Subgoal Graphs + RL** (arXiv:1817.01700) — precompute graph over grid.
- **Compressed Path Databases (CPDs)** (pathfinding.ai; ICAPS20) — precompute all-pairs
  paths; FASTEST lookup (won Grid-based Path Planning Competition). Memory-heavy but our
  maps are only ~100k cells — strong for static maps.
- **rAthena semantics (grounded):** rAthena `map.hpp` cells = single CELL_WALKABLE bit
  (src/map/map.cpp:3413). The RO CLIENT adds weight/avoidWalls mode: diagonal-into-corner
  allowed when one orthogonal neighbor walkable; per-cell 3x3 matrix. Our A* should encode
  diagonal-lock (both orthogonals walkable) vs weight-slide — a cost+walkability hybrid,
  not plain 4/8-connectivity. Domain modification, not a novel algo (no paper covers it).

## 3. ML/Learning-based — NOT production-viable for the bot (honest)

- **TransPath** (arXiv:2212.11730) — Transformer learns grid heuristics; beats Manhattan
  but needs per-map training, no optimality guarantee. Research-tier.
- **UPath** (arXiv:2602.23789, 2026) — universal planner across grid heterogeneity.
  Experimental.
- **iA*** (arXiv:2403.15870) — imperative learning A*; does NOT beat tuned A*/JPS on
  static grids (targets SLAM/vision).
- **DAA*** (arXiv:2507.09305) — deep angular A* for images. N/A.
- **A\* + DQN heuristic** (arXiv:2102.04518) — learned heuristics, large action spaces.
- **GNN shortest path OOD** (arXiv:2503.19173) — GNNs FAIL to extrapolate OOD →
  learned routing fragile for rotating map sets.
- **Skeleton-Guided Learning** (arXiv:2508.02270) — slower/less reliable than A* on
  sparse grids.
- **GraphCast (weather GNN)** — domain mismatch for 2D-grid bot routing.
- **Vision-informed DRL POMDP** (arXiv:2209.04801) — unknown/random maps; we have known
  maps → unnecessary.

**Consensus: no ML approach reliably beats branch-and-bound A*/JPS on known static
grids + online replan.** Learning gains only on unknown/vision-driven domains, at heavy
training cost + zero optimality guarantees.

## 4. High-level goal selection / map abstraction — OUR actual bottleneck

- **CPDs** → precompute distances between map gateways (warps/teleporters) → instant
  intra/inter-map cost estimates. THE proven win.
- **Subgoal/waypoint graphs** (arXiv:1817.01700; 2511.20993 for LLM-guided) — coarse
  network per map (offline), route high-level on it, A*/JPS fills detail.
- **"Lifelong navigation"** (AllDayNav 2606.10927, GOAT-Bench 2404.06609, Transient MAPF
  2412.04256) — RL/robotics-centric; does not fit a known-map 2D farming bot. Skip.
- **LLM-A\*** (EMNLP 2024 findings, aclanthology.org/2024.findings-emnlp.60) — LLM prunes
  heuristic search; interesting but NOT production-stable (bot must never hang).
- **LLM planning for goal selection** — solid: SCALAR (arXiv:2603.09036), LOAT
  (2403.09971), Subgoal Graph-Augmented LLM planning (2511.20993), LLM agent planning
  survey (arXiv:2402.02716), AlphaRoute (2607.19768). The goal-selection layer is the
  right place for LLM/heuristic logic ON TOP of guaranteed-fast A*/JPS+CPD.

## BOTTOM LINE

**Adopt now (proven, low risk):**
1. Keep A*/JPS as the low-level engine; add hash/bitpacked cell grid + JPS pruning
   (arXiv:2306.15928 fixes spin pathology).
2. Add diagonal-lock/weight-slide encoding for RO avoidWalls semantics.
3. Path smoothing (string-pulling/Funnel) post-pass — removes diagonal jitter.
4. Map-abstraction layers: per-map gateway waypoint graph + CPDs for instant route
   costs → solves inter-map routing ("which route to farm") with zero per-call search.
5. Goal-selection heuristics (where to farm, which map chain): rule-based scoring or an
   LLM policy layer (arXiv:2402.02716) choosing among precomputed routes — never calling
   A* for the abstraction.
6. Optional: Weighted A*/ARA*/A-MHA* (2508.21637) for bounded-suboptimal fast answers.

**Do NOT use in production bot:**
- ML pathfinding (TransPath/UPath/iA*/GNN) — no optimality guarantee, per-map training,
  OOD fragility (2503.19173).
- D*/LPA* per-step replan — expensive for mostly-static maps.
- LLM-A* directly in the hot loop — nondeterministic latency.
- "Lifelong navigation" RL stacks — domain mismatch.
