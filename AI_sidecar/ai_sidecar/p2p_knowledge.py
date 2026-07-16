"""
P2P Knowledge Network — True distributed learning between bots.
================================================================
Each bot is an independent agent that records experiences, shares them
with the network, and learns from other bots. No central server required.

Architecture:
- Each bot runs a local knowledge node (HTTP server on a unique port)
- Bots discover each other via the fleet coordinator
- Knowledge is shared via gossip protocol: bot A tells bot B, which tells bot C
- Each bot maintains its own ExpDB + receives updates from peers
- Conflicts are resolved by majority vote or recency

Key design: truly bottom-up. Each bot observes, records, shares, learns.
No central "brain" — the intelligence emerges from the network.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeMessage:
    """A knowledge message shared between bots."""
    msg_id: str
    sender_bot_id: str
    msg_type: str  # "experience" | "hunting_zone" | "npc_location" | "server_rate" | "alert"
                   # | "item_drop" | "item_valuation" | "item_price" | "card_drop"
                   # | "market_price" | "trade_offer" | "supply_demand"
    payload: dict[str, Any]
    timestamp: float
    ttl: int = 300  # seconds before this message is stale
    hop_count: int = 0  # how many times this has been forwarded
    max_hops: int = 5

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    def is_max_hops(self) -> bool:
        return self.hop_count >= self.max_hops


class P2PRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for P2P knowledge messages."""

    def log_message(self, format, *args):
        pass  # Suppress HTTP server log noise

    def do_POST(self):
        """Handle incoming knowledge messages. Max 1MB per message."""
        max_size = 1 * 1024 * 1024  # 1MB max
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0 or content_length > max_size:
            self.send_response(413 if content_length > max_size else 400)
            self.end_headers()
            return

        body = self.rfile.read(content_length)
        try:
            msg_data = json.loads(body)
            node = self.server._p2p_node  # type: ignore
            if node and node.receive_message(msg_data):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"duplicate"}')
        except Exception:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"status":"error"}')

    def do_GET(self):
        """Health check endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        node = self.server._p2p_node  # type: ignore
        if node:
            self.wfile.write(json.dumps({"status": "ok", "bot_id": node._bot_id}).encode())
        else:
            self.wfile.write(b'{"status":"ok"}')


class P2PKnowledgeNode:
    """A single knowledge node in the P2P network.

    Each bot runs one instance. Nodes discover each other via the
    fleet coordinator, share experiences via gossip, and maintain
    a local knowledge base that grows with every peer update.

    Thread-safe: all shared state is protected by RLock.
    """

    def __init__(self, bot_id: str, listen_port: int = 0,
                 known_peers: list[str] | None = None,
                 server_id: str = "default"):
        """Initialize P2P knowledge node.

        Args:
            bot_id: Unique bot identifier (format: "master:char_name")
            listen_port: Port for this node's HTTP server
            known_peers: Initial peer addresses
            server_id: Server identifier for knowledge isolation.
                       Each server has its own isolated knowledge network
                       so bot A's knowledge on server X doesn't contaminate
                       bot B's knowledge on server Y.
        """
        self._bot_id = bot_id
        self._listen_port = listen_port or (18090 + abs(hash(bot_id)) % 1000)
        self._server_id = server_id  # Isolates knowledge per server
        self._peers: set[str] = set(known_peers or [])
        self._lock = threading.RLock()
        self._messages: dict[str, KnowledgeMessage] = {}
        self._seen_message_ids: set[str] = set()
        self._running = False
        self._server: HTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._experience_db = None
        self._npc_discovery = None
        self._server_adaptation = None

        # Shared knowledge
        self._gossip_pool = ThreadPoolExecutor(max_workers=4)
        self._shared_hunting_zones: dict[str, dict[str, Any]] = {}
        self._shared_npc_locations: dict[str, dict[str, Any]] = {}
        self._shared_server_rates: dict[str, float] = {}
        self._shared_alerts: list[dict[str, Any]] = []
        self._shared_item_drops: dict[str, dict[str, Any]] = {}
        self._shared_item_valuations: dict[str, dict[str, Any]] = {}
        self._shared_item_prices: dict[str, dict[str, Any]] = {}
        self._shared_card_drops: dict[str, dict[str, Any]] = {}
        self._shared_market_prices: dict[str, dict[str, Any]] = {}
        self._shared_supply_demand: dict[str, dict[str, Any]] = {}
        self._gossip_cooldown: dict[str, float] = {}  # peer -> last gossip time
        self._gossip_rate_limit_s: float = 0.1  # 100ms min between messages to same peer

    def set_experience_db(self, exp_db) -> None:
        self._experience_db = exp_db

    def set_npc_discovery(self, npc_disc) -> None:
        self._npc_discovery = npc_disc

    def set_server_adaptation(self, sa) -> None:
        self._server_adaptation = sa

    def start_server(self) -> bool:
        """Start the HTTP server for receiving P2P messages.

        This is critical: without a running server, the node can send
        messages but cannot receive them. Each bot must have its own
        HTTP server listening on a unique port.
        """
        with self._lock:
            if self._server is not None:
                return True  # Already running
            try:
                from http.server import HTTPServer
                # Allow reusing the address immediately after server stops
                HTTPServer.allow_reuse_address = True
                self._server = HTTPServer(("127.0.0.1", self._listen_port), P2PRequestHandler)
                self._server._p2p_node = self  # type: ignore
                self._server_thread = threading.Thread(
                    target=self._server.serve_forever,
                    daemon=True,
                    name=f"p2p-server-{self._bot_id}",
                )
                self._server_thread.start()
                self._running = True
                logger.info(
                    "p2p_server_started: bot=%s port=%d server_id=%s",
                    self._bot_id, self._listen_port, self._server_id,
                )
                return True
            except OSError as e:
                logger.warning(
                    "p2p_server_failed: bot=%s port=%d error=%s",
                    self._bot_id, self._listen_port, e,
                )
                self._server = None
                return False

    def stop_server(self) -> None:
        """Stop the HTTP server and gossip thread pool."""
        with self._lock:
            self._running = False
            if self._server:
                self._server.shutdown()
                self._server = None
                self._server_thread = None
            # Shut down gossip thread pool
            if hasattr(self, '_gossip_pool'):
                self._gossip_pool.shutdown(wait=False)

    def add_peer(self, peer_address: str) -> None:
        """Add a peer to the knowledge network."""
        with self._lock:
            my_address = f"127.0.0.1:{self._listen_port}"
            if peer_address != my_address:
                self._peers.add(peer_address)
                logger.info("p2p_peer_added: bot=%s peer=%s", self._bot_id, peer_address)

    def remove_peer(self, peer_address: str) -> None:
        with self._lock:
            self._peers.discard(peer_address)

    def get_peers(self) -> list[str]:
        with self._lock:
            return list(self._peers)

    def broadcast_experience(self, context_type: str, map_name: str,
                              monster_name: str, action_taken: str,
                              success: bool, reward: float = 0.0) -> str:
        """Share a combat experience with all peers."""
        msg_id = f"{self._bot_id}_{context_type}_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        msg = KnowledgeMessage(
            msg_id=msg_id,
            sender_bot_id=self._bot_id,
            msg_type="experience",
            payload={
                "context_type": context_type,
                "map_name": map_name,
                "monster_name": monster_name,
                "action_taken": action_taken,
                "success": success,
                "reward": reward,
            },
            timestamp=time.time(),
        )
        self._gossip(msg)
        return msg_id

    def broadcast_hunting_zone(self, map_name: str, monster_name: str,
                                 score: float, exp_per_hp: float,
                                 danger_score: float, zeny_per_kill: float) -> str:
        msg_id = f"{self._bot_id}_zone_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        msg = KnowledgeMessage(
            msg_id=msg_id,
            sender_bot_id=self._bot_id,
            msg_type="hunting_zone",
            payload={
                "map_name": map_name,
                "monster_name": monster_name,
                "score": score,
                "exp_per_hp": exp_per_hp,
                "danger_score": danger_score,
                "zeny_per_kill": zeny_per_kill,
            },
            timestamp=time.time(),
        )
        self._gossip(msg)
        return msg_id

    def broadcast_npc_location(self, map_name: str, npc_name: str,
                                x: int, y: int, service: str) -> str:
        msg_id = f"{self._bot_id}_npc_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        msg = KnowledgeMessage(
            msg_id=msg_id,
            sender_bot_id=self._bot_id,
            msg_type="npc_location",
            payload={
                "map_name": map_name,
                "npc_name": npc_name,
                "x": x,
                "y": y,
                "service": service,
            },
            timestamp=time.time(),
        )
        self._gossip(msg)
        return msg_id

    def broadcast_server_rate(self, rate_type: str, value: float) -> str:
        msg_id = f"{self._bot_id}_rate_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        msg = KnowledgeMessage(
            msg_id=msg_id,
            sender_bot_id=self._bot_id,
            msg_type="server_rate",
            payload={
                "rate_type": rate_type,
                "value": value,
                "samples": 1,
            },
            timestamp=time.time(),
        )
        self._gossip(msg)
        return msg_id

    def broadcast_alert(self, alert_type: str, message: str,
                         data: dict[str, Any] | None = None) -> str:
        msg_id = f"{self._bot_id}_alert_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        msg = KnowledgeMessage(
            msg_id=msg_id,
            sender_bot_id=self._bot_id,
            msg_type="alert",
            payload={
                "alert_type": alert_type,
                "message": message,
                "data": data or {},
            },
            timestamp=time.time(),
        )
        self._gossip(msg)
        return msg_id

    def broadcast_item_drop(self, monster_name: str, item_name: str,
                             dropped: bool, server_rate: float = 1.0) -> str:
        """Share an item drop observation with all peers.
        
        Items, cards, and equipment vary by server. Custom servers have
        their own item IDs, drop lists, and card effects. By sharing
        drop observations across the P2P network, all bots learn which
        items actually drop, at what rate, and whether they're worth farming.
        """
        msg_id = f"{self._bot_id}_drop_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        msg = KnowledgeMessage(
            msg_id=msg_id,
            sender_bot_id=self._bot_id,
            msg_type="item_drop",
            payload={
                "monster_name": monster_name,
                "item_name": item_name,
                "dropped": dropped,
                "server_rate": server_rate,
                "is_card": "card" in item_name.lower(),
                "is_equipment": any(kw in item_name.lower() for kw in
                    ["sword", "armor", "shield", "boots", "cloak", "hat", "helm",
                     "staff", "bow", "dagger", "mace", "spear", "ring", "earring",
                     "necklace", "brooch", "glove", "muffler"]),
            },
            timestamp=time.time(),
        )
        self._gossip(msg)
        return msg_id

    def broadcast_item_valuation(self, item_name: str, value: str,
                                   action: str, reason: str) -> str:
        """Share an item valuation with all peers.
        
        What's worth keeping vs selling varies by server economy.
        When bot A learns that a custom item is valuable, bot B
        benefits without having to discover it independently.
        """
        msg_id = f"{self._bot_id}_val_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        msg = KnowledgeMessage(
            msg_id=msg_id,
            sender_bot_id=self._bot_id,
            msg_type="item_valuation",
            payload={
                "item_name": item_name,
                "value": value,  # "high" | "medium" | "low"
                "action": action,  # "keep" | "sell" | "store"
                "reason": reason,
            },
            timestamp=time.time(),
        )
        self._gossip(msg)
        return msg_id

    def broadcast_item_price(self, item_name: str, npc_buy_price: int,
                               npc_sell_price: int, market_price: int = 0) -> str:
        """Share NPC price observations with all peers.
        
        NPC buy/sell prices vary by server. Vending prices even more so.
        Shared price knowledge helps all bots value items correctly.
        """
        msg_id = f"{self._bot_id}_price_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        msg = KnowledgeMessage(
            msg_id=msg_id,
            sender_bot_id=self._bot_id,
            msg_type="item_price",
            payload={
                "item_name": item_name,
                "npc_buy_price": npc_buy_price,
                "npc_sell_price": npc_sell_price,
                "market_price": market_price,
                "timestamp": time.time(),
            },
            timestamp=time.time(),
            ttl=86400,  # Prices change slowly, 24h TTL
        )
        self._gossip(msg)
        return msg_id

    def broadcast_card_drop(self, monster_name: str, card_name: str,
                              dropped: bool) -> str:
        """Share a card drop observation with all peers.
        
        Cards are the most server-variable item. Rate, effect, and
        which monster drops which card all change between servers.
        Every bot benefits from pooled card drop observations.
        """
        msg_id = f"{self._bot_id}_card_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        msg = KnowledgeMessage(
            msg_id=msg_id,
            sender_bot_id=self._bot_id,
            msg_type="card_drop",
            payload={
                "monster_name": monster_name,
                "card_name": card_name,
                "dropped": dropped,
                "is_card": True,
            },
            timestamp=time.time(),
            ttl=86400,  # Card rates are stable, 24h TTL
        )
        self._gossip(msg)
        return msg_id

    def broadcast_market_price(self, item_name: str, price: int,
                                 listing_count: int = 1, trend: str = "stable") -> str:
        """Share observed player vending price with all peers.
        
        Market prices are driven by other players, not NPCs. When bot A
        sees an item being sold for X zeny, it shares that observation.
        Over time, the P2P network builds a real-time supply/demand map.
        """
        msg_id = f"{self._bot_id}_mkt_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        msg = KnowledgeMessage(
            msg_id=msg_id,
            sender_bot_id=self._bot_id,
            msg_type="market_price",
            payload={
                "item_name": item_name,
                "price": price,
                "listing_count": listing_count,
                "trend": trend,
                "timestamp": time.time(),
            },
            timestamp=time.time(),
            ttl=3600,
        )
        self._gossip(msg)
        return msg_id

    def broadcast_supply_demand(self, item_name: str, supply: str,
                                  demand: str, confidence: float = 0.5) -> str:
        """Share a supply/demand observation with all peers.
        
        supply: "abundant" | "common" | "scarce" | "rare"  
        demand: "low" | "medium" | "high" | "critical"
        
        When all bots agree an item is scarce with high demand, the
        network recommends farming it.
        """
        msg_id = f"{self._bot_id}_sd_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        msg = KnowledgeMessage(
            msg_id=msg_id,
            sender_bot_id=self._bot_id,
            msg_type="supply_demand",
            payload={
                "item_name": item_name,
                "supply": supply,
                "demand": demand,
                "confidence": confidence,
            },
            timestamp=time.time(),
            ttl=3600,
        )
        self._gossip(msg)
        return msg_id

    def _gossip(self, msg: KnowledgeMessage) -> None:
        """Send a message to all peers (gossip protocol)."""
        with self._lock:
            if msg.msg_id in self._seen_message_ids:
                return
            self._seen_message_ids.add(msg.msg_id)
            self._messages[msg.msg_id] = msg
            
            # Rate limit: prevent flooding the network
            now = time.time()
            self._gossip_cooldown = {k: v for k, v in self._gossip_cooldown.items() if now - v < 60.0}
            if len(self._gossip_cooldown) > 100:
                # Too many unique peers in 60s — possible flood attack
                logger.warning("p2p_rate_limit: %d unique peers in 60s, throttling", len(self._gossip_cooldown))
                return
            
            # Prevent memory leak: cap seen message IDs at 100,000
            if len(self._seen_message_ids) > 100000:
                self._seen_message_ids = set(list(self._seen_message_ids)[-50000:])
            # Cap stored messages at 10,000
            if len(self._messages) > 10000:
                # Remove oldest messages (non-expired ones)
                sorted_msgs = sorted(self._messages.items(), key=lambda x: x[1].timestamp)
                for msg_id, _ in sorted_msgs[:len(self._messages) - 10000]:
                    del self._messages[msg_id]

        # Process locally
        self._process_message(msg)

        # Forward to peers (non-blocking: run in thread)
        peers = list(self._peers)
        if peers:
            t = threading.Thread(target=self._forward_to_peers, args=(peers, msg), daemon=True)
            t.start()

    def _forward_to_peers(self, peers: list[str], msg: KnowledgeMessage) -> None:
        """Forward message to all peers in a background thread."""
        for peer in peers:
            try:
                self._send_to_peer(peer, msg)
            except Exception:
                pass

    def _send_to_peer(self, peer_address: str, msg: KnowledgeMessage) -> bool:
        """Send a message to a specific peer via HTTP."""
        try:
            data = json.dumps({
                "msg_id": msg.msg_id,
                "sender_bot_id": msg.sender_bot_id,
                "msg_type": msg.msg_type,
                "payload": msg.payload,
                "timestamp": msg.timestamp,
                "hop_count": msg.hop_count + 1,
                "max_hops": msg.max_hops,
                "ttl": msg.ttl,
            }).encode()
            req = Request(
                f"http://{peer_address}/p2p/receive",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # Use blocking socket with short timeout
            with urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except URLError:
            # Peer might be offline — remove from list
            self.remove_peer(peer_address)
            return False
        except Exception:
            return False

    def receive_message(self, msg_data: dict[str, Any]) -> bool:
        """Receive a message from a peer (called by HTTP endpoint)."""
        msg = KnowledgeMessage(
            msg_id=msg_data.get("msg_id", ""),
            sender_bot_id=msg_data.get("sender_bot_id", ""),
            msg_type=msg_data.get("msg_type", ""),
            payload=msg_data.get("payload", {}),
            timestamp=msg_data.get("timestamp", time.time()),
            hop_count=msg_data.get("hop_count", 0),
            max_hops=msg_data.get("max_hops", 5),
            ttl=msg_data.get("ttl", 300),
        )

        with self._lock:
            if msg.msg_id in self._seen_message_ids:
                return False
            if msg.is_expired():
                return False
            if msg.is_max_hops():
                return False
            self._seen_message_ids.add(msg.msg_id)
            self._messages[msg.msg_id] = msg

        # Process locally
        self._process_message(msg)

        # Forward to other peers (non-blocking)
        peers = list(self._peers)
        if peers:
            t = threading.Thread(target=self._forward_to_peers, args=(peers, msg), daemon=True)
            t.start()

        return True

    def _process_message(self, msg: KnowledgeMessage) -> None:
        """Process a received message and update local knowledge."""
        try:
            if msg.msg_type == "experience":
                self._process_experience(msg)
            elif msg.msg_type == "hunting_zone":
                self._process_hunting_zone(msg)
            elif msg.msg_type == "npc_location":
                self._process_npc_location(msg)
            elif msg.msg_type == "server_rate":
                self._process_server_rate(msg)
            elif msg.msg_type == "alert":
                self._process_alert(msg)
            elif msg.msg_type == "item_drop":
                self._process_item_drop(msg)
            elif msg.msg_type == "item_valuation":
                self._process_item_valuation(msg)
            elif msg.msg_type == "item_price":
                self._process_item_price(msg)
            elif msg.msg_type == "card_drop":
                self._process_card_drop(msg)
            elif msg.msg_type == "market_price":
                self._process_market_price(msg)
            elif msg.msg_type == "supply_demand":
                self._process_supply_demand(msg)
        except Exception:
            logger.exception("p2p_process_message_failed: type=%s", msg.msg_type)

    def _process_experience(self, msg: KnowledgeMessage) -> None:
        p = msg.payload
        if self._experience_db is not None:
            from ai_sidecar.experience_db import ExperienceEntry
            entry = ExperienceEntry(
                bot_id=msg.sender_bot_id,
                timestamp=msg.timestamp,
                context_type=p.get("context_type", "combat"),
                map_name=p.get("map_name", ""),
                monster_name=p.get("monster_name", ""),
                action_taken=p.get("action_taken", ""),
                success=p.get("success", False),
                reward=float(p.get("reward", 0.0)),
            )
            self._experience_db.record(entry)

    def _process_hunting_zone(self, msg: KnowledgeMessage) -> None:
        p = msg.payload
        map_name = p.get("map_name", "")
        if map_name:
            with self._lock:
                key = f"{map_name}:{p.get('monster_name', '')}"
                if key not in self._shared_hunting_zones:
                    self._shared_hunting_zones[key] = dict(p)
                    self._shared_hunting_zones[key]["discovered_by"] = msg.sender_bot_id
                    self._shared_hunting_zones[key]["discovered_at"] = msg.timestamp
                    logger.info("p2p_hunting_zone: bot=%s map=%s", msg.sender_bot_id, map_name)

    def _process_npc_location(self, msg: KnowledgeMessage) -> None:
        p = msg.payload
        map_name = p.get("map_name", "")
        service = p.get("service", "")
        if map_name and service:
            with self._lock:
                key = f"{map_name}:{service}"
                if key not in self._shared_npc_locations:
                    self._shared_npc_locations[key] = dict(p)
                    self._shared_npc_locations[key]["discovered_by"] = msg.sender_bot_id

    def _process_server_rate(self, msg: KnowledgeMessage) -> None:
        p = msg.payload
        rate_type = p.get("rate_type", "")
        value = p.get("value", 1.0)
        if rate_type:
            with self._lock:
                if rate_type not in self._shared_server_rates:
                    self._shared_server_rates[rate_type] = value
                else:
                    current = self._shared_server_rates[rate_type]
                    self._shared_server_rates[rate_type] = (current + value) / 2.0

    def _process_alert(self, msg: KnowledgeMessage) -> None:
        p = msg.payload
        alert = {
            "bot_id": msg.sender_bot_id,
            "alert_type": p.get("alert_type", "info"),
            "message": p.get("message", ""),
            "data": p.get("data", {}),
            "timestamp": msg.timestamp,
        }
        with self._lock:
            self._shared_alerts.append(alert)
            if len(self._shared_alerts) > 100:
                self._shared_alerts = self._shared_alerts[-100:]

    def _process_item_drop(self, msg: KnowledgeMessage) -> None:
        """Process an item drop observation from a peer.
        
        Records the observation and recomputes drop rate patterns.
        When multiple bots report drops of the same item from the same
        monster, the estimated drop rate becomes more accurate.
        """
        p = msg.payload
        monster = p.get("monster_name", "")
        item = p.get("item_name", "")
        dropped = p.get("dropped", False)
        is_card = p.get("is_card", False)
        key = f"{monster}:{item}"
        
        with self._lock:
            if key not in self._shared_item_drops:
                self._shared_item_drops[key] = {
                    "monster": monster,
                    "item": item,
                    "is_card": is_card,
                    "observations": [],
                    "total_kills": 0,
                    "total_drops": 0,
                    "observed_rate": 0.0,
                    "expected_rate": 0.0,
                    "variance": 0.0,
                    "last_observed": 0.0,
                    "reported_by": set(),
                }
            record = self._shared_item_drops[key]
            record["observations"].append(dropped)
            record["total_kills"] += 1
            if dropped:
                record["total_drops"] += 1
            record["observed_rate"] = record["total_drops"] / max(record["total_kills"], 1)
            record["last_observed"] = msg.timestamp
            record["reported_by"].add(msg.sender_bot_id)
            
            # Check expected rate from knowledge DB
            if self._server_adaptation is not None:
                profile = self._server_adaptation.get_profile()
                record["expected_rate"] = profile.drop_rate * 0.01  # 1% base
            else:
                record["expected_rate"] = 0.01
            
            # Compute variance: |observed - expected| / expected
            if record["expected_rate"] > 0:
                record["variance"] = abs(record["observed_rate"] - record["expected_rate"]) / record["expected_rate"]
            
            # PATTERN DETECTION: if variance > 3x, this server has custom rates
            if record["variance"] > 3.0 and record["total_kills"] >= 10:
                logger.info("p2p_drop_pattern: monster=%s item=%s observed=%.4f expected=%.4f variance=%.1fx (CUSTOM RATE)",
                           monster, item, record["observed_rate"], record["expected_rate"], record["variance"])
                # Broadcast alert about custom drop rate
                self.broadcast_alert("custom_drop_rate", 
                    f"Custom drop rate detected: {item} from {monster} ({record['observed_rate']:.2%} vs {record['expected_rate']:.2%} expected)",
                    {"monster": monster, "item": item, "observed_rate": record["observed_rate"]})
            
            # PATTERN DETECTION: if item drops 0 times in 50+ kills, it's disabled
            if record["total_kills"] >= 50 and record["total_drops"] == 0:
                logger.info("p2p_drop_pattern: monster=%s item=%s NEVER DROPS (50+ kills, 0 drops) — DISABLED ON SERVER",
                           monster, item)
                self.broadcast_alert("disabled_item",
                    f"Item disabled on server: {item} from {monster} (0 drops in {record['total_kills']} kills)",
                    {"monster": monster, "item": item, "total_kills": record["total_kills"]})

    def _process_item_valuation(self, msg: KnowledgeMessage) -> None:
        """Process an item valuation from a peer.
        
        Builds consensus on what items are worth keeping vs selling.
        If 3+ bots independently value the same item as "sell", it's junk.
        If 3+ bots value it as "keep", it's valuable.
        """
        p = msg.payload
        item = p.get("item_name", "")
        value = p.get("value", "medium")
        action = p.get("action", "sell")
        reason = p.get("reason", "")
        
        with self._lock:
            if item not in self._shared_item_valuations:
                self._shared_item_valuations[item] = {
                    "valuations": [],
                    "consensus_value": "unknown",
                    "consensus_action": "sell",
                    "confidence": 0.0,
                    "reported_by": set(),
                }
            record = self._shared_item_valuations[item]
            record["valuations"].append({"value": value, "action": action, "reason": reason, "bot": msg.sender_bot_id})
            record["reported_by"].add(msg.sender_bot_id)
            
            # Compute consensus: majority vote
            values = [v["value"] for v in record["valuations"]]
            high_count = sum(1 for v in values if v == "high")
            low_count = sum(1 for v in values if v == "low")
            if high_count > low_count and high_count >= 2:
                record["consensus_value"] = "high"
                record["consensus_action"] = "keep"
                record["confidence"] = min(1.0, high_count / 5.0)
            elif low_count > high_count and low_count >= 2:
                record["consensus_value"] = "low"
                record["consensus_action"] = "sell"
                record["confidence"] = min(1.0, low_count / 5.0)
            
            # PATTERN DETECTION: item with high consensus is worth farming
            if record["consensus_value"] == "high" and record["confidence"] >= 0.6:
                logger.info("p2p_valuation_pattern: item=%s value=high confidence=%.0f%% bots=%d",
                           item, record["confidence"] * 100, len(record["reported_by"]))

    def _process_item_price(self, msg: KnowledgeMessage) -> None:
        """Process an NPC price observation from a peer.
        
        Tracks price ranges across the network. When 3+ bots report
        the same item price, that price is confirmed.
        """
        p = msg.payload
        item = p.get("item_name", "")
        buy_price = p.get("npc_buy_price", 0)
        sell_price = p.get("npc_sell_price", 0)
        
        with self._lock:
            if item not in self._shared_item_prices:
                self._shared_item_prices[item] = {
                    "buy_prices": [],
                    "sell_prices": [],
                    "confirmed_buy": 0,
                    "confirmed_sell": 0,
                    "reported_by": set(),
                }
            record = self._shared_item_prices[item]
            record["buy_prices"].append(buy_price)
            record["sell_prices"].append(sell_price)
            record["reported_by"].add(msg.sender_bot_id)
            
            # Use median for confirmed price
            if len(record["buy_prices"]) >= 3:
                sorted_buys = sorted(record["buy_prices"])
                record["confirmed_buy"] = sorted_buys[len(sorted_buys) // 2]
            if len(record["sell_prices"]) >= 3:
                sorted_sells = sorted(record["sell_prices"])
                record["confirmed_sell"] = sorted_sells[len(sorted_sells) // 2]

    def _process_card_drop(self, msg: KnowledgeMessage) -> None:
        """Process a card drop observation from a peer.
        
        Cards are the most server-variable items. Dedicated tracking
        with pattern detection for custom card rates.
        """
        p = msg.payload
        monster = p.get("monster_name", "")
        card = p.get("card_name", "")
        dropped = p.get("dropped", False)
        key = f"{monster}:{card}"
        
        with self._lock:
            if key not in self._shared_card_drops:
                self._shared_card_drops[key] = {
                    "monster": monster,
                    "card": card,
                    "observations": [],
                    "total_kills": 0,
                    "total_drops": 0,
                    "observed_rate": 0.0,
                    "expected_rate": 0.0001,  # 0.01% base card rate
                    "variance": 0.0,
                    "last_observed": 0.0,
                    "reported_by": set(),
                }
            record = self._shared_card_drops[key]
            record["observations"].append(dropped)
            record["total_kills"] += 1
            if dropped:
                record["total_drops"] += 1
            record["observed_rate"] = record["total_drops"] / max(record["total_kills"], 1)
            record["last_observed"] = msg.timestamp
            record["reported_by"].add(msg.sender_bot_id)
            
            # PATTERN DETECTION: high card rate = server feature
            if record["total_kills"] >= 100 and record["observed_rate"] > 0.001:
                # More than 0.1% = high rate server
                logger.info("p2p_card_pattern: monster=%s card=%s rate=%.4f (HIGH CARD RATE)",
                           monster, card, record["observed_rate"])
                self.broadcast_alert("high_card_rate",
                    f"High card rate: {card} from {monster} ({record['observed_rate']:.2%})",
                    {"monster": monster, "card": card, "rate": record["observed_rate"]})
            
            # PATTERN DETECTION: card drop confirmed
            if record["total_drops"] >= 1:
                logger.info("p2p_card_confirmed: monster=%s card=%s confirmed by %s (kills=%d)",
                           monster, card, msg.sender_bot_id, record["total_kills"])

    def _process_market_price(self, msg: KnowledgeMessage) -> None:
        """Process a market price observation from a peer.
        
        Builds a real-time price map. When 5+ bots report similar prices
        for the same item, that price is confirmed as the market rate.
        Detects trends: if prices are dropping, sell now. If rising, hold.
        """
        p = msg.payload
        item = p.get("item_name", "")
        price = p.get("price", 0)
        listing_count = p.get("listing_count", 1)
        trend = p.get("trend", "stable")
        
        with self._lock:
            if item not in self._shared_market_prices:
                self._shared_market_prices[item] = {
                    "prices": [],
                    "confirmed_price": 0,
                    "min_price": 0,
                    "max_price": 0,
                    "avg_price": 0,
                    "trend": "unknown",
                    "listings_seen": 0,
                    "reported_by": set(),
                    "last_updated": 0.0,
                }
            record = self._shared_market_prices[item]
            record["prices"].append(price)
            record["listings_seen"] += listing_count
            record["reported_by"].add(msg.sender_bot_id)
            record["last_updated"] = msg.timestamp
            
            # Compute stats
            if len(record["prices"]) >= 3:
                sorted_p = sorted(record["prices"])
                record["confirmed_price"] = sorted_p[len(sorted_p) // 2]
                record["min_price"] = sorted_p[0]
                record["max_price"] = sorted_p[-1]
                record["avg_price"] = sum(sorted_p) / len(sorted_p)
            
            # Trend detection: compare recent vs older prices
            if len(record["prices"]) >= 5:
                recent = record["prices"][-3:]
                older = record["prices"][:3]
                if sum(recent) / 3 < sum(older) / 3 * 0.9:
                    record["trend"] = "falling"
                elif sum(recent) / 3 > sum(older) / 3 * 1.1:
                    record["trend"] = "rising"
                else:
                    record["trend"] = "stable"
            
            # PATTERN: if price is falling, alert bots to sell now
            if record["trend"] == "falling" and len(record["prices"]) >= 5:
                logger.info("p2p_market_pattern: item=%s price=%d trend=%s (SELL NOW)",
                           item, record["confirmed_price"], record["trend"])
                self.broadcast_alert("sell_now",
                    f"Price dropping: {item} at {record['confirmed_price']}z ({record['trend']})",
                    {"item": item, "price": record["confirmed_price"], "trend": record["trend"]})

    def _process_supply_demand(self, msg: KnowledgeMessage) -> None:
        """Process a supply/demand observation from a peer.
        
        Builds consensus on market supply and demand. When 3+ bots agree
        an item is "scarce" with "high" demand, that's a farming opportunity.
        """
        p = msg.payload
        item = p.get("item_name", "")
        supply = p.get("supply", "common")
        demand = p.get("demand", "medium")
        confidence = p.get("confidence", 0.5)
        
        with self._lock:
            if item not in self._shared_supply_demand:
                self._shared_supply_demand[item] = {
                    "observations": [],
                    "consensus_supply": "unknown",
                    "consensus_demand": "unknown",
                    "confidence": 0.0,
                    "reported_by": set(),
                    "farming_recommendation": None,
                }
            record = self._shared_supply_demand[item]
            record["observations"].append({"supply": supply, "demand": demand, "confidence": confidence})
            record["reported_by"].add(msg.sender_bot_id)
            
            # Compute consensus
            supplies = [o["supply"] for o in record["observations"]]
            demands = [o["demand"] for o in record["observations"]]
            
            supply_score = 0
            for s in supplies:
                if s == "abundant": supply_score += 1
                elif s == "common": supply_score += 2
                elif s == "scarce": supply_score += 3
                elif s == "rare": supply_score += 4
            avg_supply = supply_score / max(len(supplies), 1)
            if avg_supply >= 3.5: record["consensus_supply"] = "rare"
            elif avg_supply >= 2.5: record["consensus_supply"] = "scarce"
            elif avg_supply >= 1.5: record["consensus_supply"] = "common"
            else: record["consensus_supply"] = "abundant"
            
            demand_score = 0
            for d in demands:
                if d == "low": demand_score += 1
                elif d == "medium": demand_score += 2
                elif d == "high": demand_score += 3
                elif d == "critical": demand_score += 4
            avg_demand = demand_score / max(len(demands), 1)
            if avg_demand >= 3.5: record["consensus_demand"] = "critical"
            elif avg_demand >= 2.5: record["consensus_demand"] = "high"
            elif avg_demand >= 1.5: record["consensus_demand"] = "medium"
            else: record["consensus_demand"] = "low"
            
            record["confidence"] = min(1.0, len(record["reported_by"]) / 5.0)
            
            # PATTERN: scarce + high demand = farm this
            if record["consensus_supply"] in ("scarce", "rare") and record["consensus_demand"] in ("high", "critical"):
                record["farming_recommendation"] = "farm"
                if record["confidence"] >= 0.4:
                    logger.info("p2p_market_opportunity: item=%s supply=%s demand=%s (FARM THIS)",
                               item, record["consensus_supply"], record["consensus_demand"])
                    self.broadcast_alert("farming_opportunity",
                        f"Market opportunity: {item} ({record['consensus_supply']}/{record['consensus_demand']})",
                        {"item": item, "supply": record["consensus_supply"], "demand": record["consensus_demand"]})
            elif record["consensus_supply"] == "abundant" and record["consensus_demand"] == "low":
                record["farming_recommendation"] = "avoid"

    def get_shared_hunting_zones(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._shared_hunting_zones)

    def get_shared_npc_locations(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._shared_npc_locations)

    def get_shared_server_rates(self) -> dict[str, float]:
        with self._lock:
            return dict(self._shared_server_rates)

    def get_shared_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return self._shared_alerts[-limit:]

    def get_shared_item_drops(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._shared_item_drops)

    def get_shared_item_valuations(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._shared_item_valuations)

    def get_shared_item_prices(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._shared_item_prices)

    def get_shared_card_drops(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._shared_card_drops)

    def get_shared_item_valuation(self, item_name: str) -> dict[str, Any]:
        """Get the consensus valuation for a specific item."""
        with self._lock:
            return dict(self._shared_item_valuations.get(item_name, {}))

    def get_shared_item_price(self, item_name: str) -> dict[str, Any]:
        """Get the confirmed price for a specific item."""
        with self._lock:
            return dict(self._shared_item_prices.get(item_name, {}))

    def get_shared_card_drop(self, monster_name: str, card_name: str) -> dict[str, Any]:
        with self._lock:
            key = f"{monster_name}:{card_name}"
            return dict(self._shared_card_drops.get(key, {}))

    def get_shared_market_prices(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._shared_market_prices)

    def get_shared_supply_demand(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._shared_supply_demand)

    def get_shared_market_price(self, item_name: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._shared_market_prices.get(item_name, {}))

    def get_shared_supply_demand_for(self, item_name: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._shared_supply_demand.get(item_name, {}))

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "bot_id": self._bot_id,
                "server_id": self._server_id,
                "listen_port": self._listen_port,
                "server_running": self._server is not None,
                "peers": len(self._peers),
                "peer_list": list(self._peers),
                "messages_seen": len(self._seen_message_ids),
                "messages_stored": len(self._messages),
                "shared_hunting_zones": len(self._shared_hunting_zones),
                "shared_npc_locations": len(self._shared_npc_locations),
                "shared_alerts": len(self._shared_alerts),
                "shared_item_drops": len(self._shared_item_drops),
                "shared_item_valuations": len(self._shared_item_valuations),
                "shared_item_prices": len(self._shared_item_prices),
                "shared_card_drops": len(self._shared_card_drops),
                "reported_cards": len(self._shared_card_drops),
                "reported_items": len(self._shared_item_drops),
                "shared_market_prices": len(self._shared_market_prices),
                "shared_supply_demand": len(self._shared_supply_demand),
                "farming_opportunities": sum(1 for v in self._shared_supply_demand.values() if v.get("farming_recommendation") == "farm"),
            }


class P2PNetworkManager:
    """Manages the P2P knowledge network across all bots.

    Coordinates peer discovery, gossip, and knowledge consolidation.
    Each bot's P2PKnowledgeNode is registered here.
    """

    def __init__(self):
        self._nodes: dict[str, P2PKnowledgeNode] = {}
        self._node_ports: dict[str, int] = {}

    def register_node(self, bot_id: str, node: P2PKnowledgeNode) -> None:
        self._nodes[bot_id] = node
        self._node_ports[bot_id] = node._listen_port

    def unregister_node(self, bot_id: str) -> None:
        self._nodes.pop(bot_id, None)
        self._node_ports.pop(bot_id, None)

    def connect_all(self) -> None:
        """Connect all registered nodes to each other."""
        bot_ids = list(self._nodes.keys())
        for i, bot_id in enumerate(bot_ids):
            node = self._nodes[bot_id]
            for j, other_id in enumerate(bot_ids):
                if i != j:
                    other_port = self._node_ports.get(other_id, 0)
                    if other_port:
                        node.add_peer(f"127.0.0.1:{other_port}")

    def start_all_servers(self) -> None:
        """Start HTTP servers for all registered nodes."""
        for bot_id, node in self._nodes.items():
            node.start_server()

    def stop_all_servers(self) -> None:
        for node in self._nodes.values():
            node.stop_server()

    def get_network_stats(self) -> dict[str, Any]:
        total_peers = sum(len(n.get_peers()) for n in self._nodes.values())
        total_messages = sum(n.get_stats()["messages_seen"] for n in self._nodes.values())
        total_zones = sum(len(n.get_shared_hunting_zones()) for n in self._nodes.values())
        total_npcs = sum(len(n.get_shared_npc_locations()) for n in self._nodes.values())
        return {
            "bots": len(self._nodes),
            "total_peers": total_peers,
            "total_messages": total_messages,
            "shared_hunting_zones": total_zones,
            "shared_npc_locations": total_npcs,
            "nodes": {bid: n.get_stats() for bid, n in self._nodes.items()},
        }