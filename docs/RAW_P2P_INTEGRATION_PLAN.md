# ARCHITECTURE PLAN — openkore-ai-v3 RAW P2P / IPv6 / UDP integration

Status: DRAFT (design-first, per user directive "carefully plan the architecture
and implement/wire/verify them")

## 0. Scope
Integrate into openkore-ai-v3 (client-side only) the transport capabilities RAW
already has:
  1. Peer-host map-server connectivity
  2. P2P relay
  3. In-game P2P mesh
  4. IPv6 + UDP
These must be server-agnostic, zero hardcoded server facts, zero human
intervention, live-production ready.

## 1. Transport model (mirror RAW's, don't reinvent)
RAW transport layers (from rathena-AI-world):
- LOGIN:    TCP, ipv6-raw-server.openkore-ai.com:46019 (playit)
- CHAR:     TCP, ipv6-raw-server.openkore-ai.com:21544 (playit)
- MAP:      TCP, ipv6-raw-server.openkore-ai.com:53378 (playit)
- In-game mesh: WebRTC data channel, P2P; position/state only (0x035F walk,
  0x0361 dir), combat/items/skills/chat go to server.
- Peer-host: a peer's map-server.exe registers with char, claims empty maps
  (EVE model), serves them.

## 2. What openkore-ai-v3 needs (gap analysis)
### 2a. Bot connection (REQUIRED for bot to play at all) — IN SCOPE NOW
- TCP over ipv6-raw-server... — ALREADY WORKS (login+char+map entry verified).
- BLOCKER: char-select "all maps not ready" because the central map-server's
  accessible-map ownership flaps under peer-host takeover. Bot must retry
  char-select instead of timing out (like the RAW client does).

### 2b. In-game P2P mesh (bot benefits) — design only
- The RAW client uses the mesh to relay position/state between peers, reducing
  server load. A bot could join the same mesh to be "visible" to peers with low
  latency. This requires a WebRTC implementation in openkore (libdatachannel /
  Perl bindings) — large. DEFER unless user wants the bot to host maps too.

### 2c. Peer-host map-server (bot hosts maps) — OUT OF SCOPE for the bot
- A bot running a map-server.exe (like RAW's peer host) to serve maps = a
  different product (capacity contributor). The bot is a player, not a host.

### 2d. P2P relay (bot relays) — OUT OF SCOPE for the bot
- Same as 2c: relay is a server-side capacity role.

### 2e. IPv6/UDP (bot transport)
- The bot currently connects via TCP to the IPv6 hostname (playit tunnels it).
- In-game movement uses TCP 0x035F to the server. The RAW DLL routes 0x035F
  over UDP when a direct P2P link exists — a bot has no P2P link, so it uses TCP.
- IPv6: the playit hostname resolves and works over the tunnel. Native IPv6
  would need a direct v6 route — playit tunnels it, so TCP-IPv6 already works.

## 3. Recommended sequencing (pragmatic, nothing breaks)
1. FIX char-select retry (bot waits/retries on "not ready" instead of timeout).
   THIS is what blocks a stable farming bot. Client-side, small.
2. Verify bot stays in-game >3 min, gains EXP. (Definition of done.)
3. THEN evaluate mesh/relay/host integration as a separate phase — only if the
   bot's value is as a capacity node, which conflicts with "player bot."

## 4. Decisions needed from you
- Is openkore-ai-v3 meant to be (a) a PLAYER bot (connect + farm, no hosting) or
  (b) also a CAPACITY NODE (host maps / relay / mesh like RAW's peer)? 
  These have different architectures. (a) needs only the char-select fix + TCP.
  (b) needs WebRTC mesh + map-server hosting inside the bot — much larger.

## 5. Open questions
- If (b): which maps, what auth (ephemeral DB creds?), how does the bot's
  map-server.exe get distributed/updated?
- IPv6/UDP: do you want the bot to open direct UDP P2P links to peers (needs
  NAT traversal) or just use the existing TCP-through-tunnel path?

## 6. Guardrails
- Zero hardcoded server facts (no item/map/port literals) — facts from the
  live server (recvpackets/servers.txt already carry the playit endpoints).
- Zero human intervention; reconnect/retry handled internally.
- Nothing breaks: char-select retry is additive, no logic removal.
- Commit per phase, push at reasonable stage (per standing directive).
