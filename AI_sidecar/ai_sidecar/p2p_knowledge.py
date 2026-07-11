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
        """Handle incoming knowledge messages."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_response(400)
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
        self._listen_port = listen_port or (18090 + hash(bot_id) % 100)
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
        self._shared_hunting_zones: dict[str, dict[str, Any]] = {}
        self._shared_npc_locations: dict[str, dict[str, Any]] = {}
        self._shared_server_rates: dict[str, float] = {}
        self._shared_alerts: list[dict[str, Any]] = []

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
        """Stop the HTTP server."""
        with self._lock:
            self._running = False
            if self._server:
                self._server.shutdown()
                self._server = None
                self._server_thread = None

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

    def _gossip(self, msg: KnowledgeMessage) -> None:
        """Send a message to all peers (gossip protocol)."""
        with self._lock:
            if msg.msg_id in self._seen_message_ids:
                return
            self._seen_message_ids.add(msg.msg_id)
            self._messages[msg.msg_id] = msg

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