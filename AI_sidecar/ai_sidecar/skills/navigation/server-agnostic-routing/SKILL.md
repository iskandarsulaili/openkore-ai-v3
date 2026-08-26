---
name: server-agnostic-routing
description: "Route between maps via discovered portal graph, never hardcoded coords."
version: 1.0.0
triggers:
  - navigation
  - move_requested
  - stuck_route
when_to_use:
  - bot needs to change maps
  - bot is stuck / cannot calculate a route
  - cold-start escape from a starting island
when_not_to_use:
  - bot is already on the target map
metadata:
  domain: navigation
  subdomain: portal_graph
  source: dynamic_portal_discovery
  confidence: 0.9
  tags: [routing, portals, server-agnostic]
---
# Server-Agnostic Routing via Discovered Portal Graph

## Core Rule
NEVER hardcode a `move <map> x y` or `talknpc <x> <y>` coordinate in cold-start/reflex
emitters. Portals are OBSERVED, not assumed — resolve routing through the discovered
portal graph (`dynamic_portal_discovery.py`, `map_knowledge.py`, pathfinder).

## How to Route
1. Read the current map + target from the bridge snapshot.
2. Look up the portal graph (`portals.txt`) for the edge toward the goal map.
3. Emit `move` to the discovered portal, then traverse it.
4. If no route is calculable, escalate to the conscious tier (LLM) to reason about the
   layout from live observation — never guess a coordinate.

## Stuck Handling
- "Calculating route to <map>" loop = the portal graph lacks the edge. Generate the
  distance map if missing (OpenKore needs `<map>.dist`), then re-evaluate.
- A bot that reaches a farm but won't stay = cold-start step machinery re-emitting routing
  commands every cycle — rate-limit, don't flood the action queue.

## Pitfall
Over-tweaking `heuristic_service.py` with literal coordinates to fix one island is the #1
anti-pattern — it breaks server-agnosticism. Route through discovery instead.
