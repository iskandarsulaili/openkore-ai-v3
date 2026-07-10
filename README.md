<p align="center">
  <img alt="openkore AI" src="https://upload.wikimedia.org/wikipedia/commons/b/b5/Kore_2g_logo.png" width="200">
</p>

# openkore **AI**

> Ragnarok Online bot powered by LLM decision-making — not just macros.
> **AI**, not *bypass*.

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
[![Sponsor](https://img.shields.io/badge/Sponsor-donate-EA4AAA?logo=githubsponsors)](https://github.com/sponsors/iskandarsulaili)
[![GitHub stars](https://img.shields.io/github/stars/iskandarsulaili/openkore-ai-v3)](https://github.com/iskandarsulaili/openkore-ai-v3/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/iskandarsulaili/openkore-ai-v3)](https://github.com/iskandarsulaili/openkore-ai-v3/issues)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![Perl](https://img.shields.io/badge/Perl-5-blue?logo=perl)](https://perl.org)

🌐 [Português](i18n/README-pt-BR.md) | [Tagalog](i18n/README-tl.md) | [日本語](i18n/README-ja.md) | [한국어](i18n/README-ko.md) | [ไทย](i18n/README-th.md) | [Indonesia](i18n/README-id.md) | [简体中文](i18n/README-zh-CN.md)

---

## What is this?

A modified OpenKore that adds an **AI decision engine** (FastAPI + Python) alongside the classic Perl client. Bots use real-time game state + optional LLM calls to make decisions instead of following hardcoded macros.

**Compared to vanilla OpenKore:**

| Vanilla OpenKore | openkore **AI** |
|-----------------|----------------|
| Macro-based decision making | Reflex → Heuristic → LLM (3-tier) |
| Fixed behavior per config | Self-adapts from outcomes |
| Single bot per instance | Multi-bot swarm coordination |
| No cost control | Per-bot daily budget, graduated tiers |
| Community macros only | Cross-bot shared learning database |
| Reacts to conditions | Plans ahead via PDCA loop |

---

## Features

- **3-tier decision engine** — Fast reflex rules (17 built-in) → Heuristic scoring → LLM (DeepSeek, OpenAI, or local Ollama)
- **Multi-bot fleet** — Bots share state, auto-assign roles (tank/healer/dps/crafter), relay messages
- **Self-learning** — Tracks outcomes per action/map/monster. Improves over time. Cross-bot experience sharing via fleet coordinator.
- **Cost control** — Set daily token budget, hourly call limit, or disable LLM entirely. Configurable per environment via env vars.
- **RO mechanics** — Party, PVP, GVG, MVP, refine, cards, quests, crafting, job change, stat/skill allocation
- **NPC interaction** — Auto-talk for warps, quests, job changes, storage, vending, Kafra. LLM-powered conversation engine for complex NPCs.
- **LLM-powered NPC dialog** — Heuristic response sequences for common NPCs (warp/Kafra/shops), LLM fallback for quest/job/refine NPCs
- **Live console** — `./start.sh` streams all bots + sidecar logs in one terminal, color-coded
- **API server** — FastAPI with **112+ endpoints** across 19 routers, auth middleware, fleet coordination, PDCA autonomy loop
- **CrewAI agent framework** — 18 specialized agents (combat, economy, navigation, questing, trading, etc.) managed by orchestrator
- **ML Subconscious** — Self-hosted behavior learning via shadow-mode training, macro distillation, and model promotion pipeline
- **Works with vanilla OpenKore configs** — Uses standard `control/config.txt` format

---

## Required

- **Python 3.11+** — For the AI sidecar
- **Perl 5** — For the OpenKore client (bundled)
- **Ragnarok Online account** — On any server compatible with OpenKore
- **LLM provider** (optional) — DeepSeek API key, OpenAI API key, or local Ollama instance

---

## Quick Start (Single Bot)

Set up credentials in `.env` (at the repo root):

```bash
# format: BOT_<name>_PASS=<password>
cat >> .env << 'EOF'
BOT_yourname_PASS=your_password
EOF
```

```bash
# 1. Set up Python environment
cd AI_sidecar
python3 -m venv venv
source venv/bin/activate
pip install -e .
cd ..

# 2. Create a bot profile
mkdir -p .bot_profiles/yourname/control
cp control/config.txt .bot_profiles/yourname/control/
# Edit .bot_profiles/yourname/control/config.txt with your server and account details

# 3. Start sidecar + bot
./start.sh sidecar &   # Start sidecar in background
./start.sh bot yourname &   # Start one bot
```

Or start everything at once (for preset bots):

```bash
# Set bot credentials in .env, then:
./start.sh all
```

## Quick Start (Multi-Bot Fleet)

```bash
# Set up one profile per account:
mkdir -p .bot_profiles/char1/control .bot_profiles/char2/control
cp control/config.txt .bot_profiles/char1/control/
cp control/config.txt .bot_profiles/char2/control/
# Edit each profile's config.txt with different credentials

# Set passwords in .env
cat >> .env << 'EOF'
BOT_char1_PASS=pass1
BOT_char2_PASS=pass2
EOF

# Start everything at once:
./start.sh all
```

The console shows all bots + sidecar logs in one view. `Ctrl+C` stops everything cleanly.

---

## Cost Tiers

Set via environment variable `OPENKORE_AI_LLM_COST_TIER` in `AI_sidecar/.env`:

| Setting | Effect |
|---------|--------|
| `OPENKORE_AI_LLM_COST_TIER=off` | Reflex + heuristic only. Zero LLM cost. |
| `OPENKORE_AI_LLM_COST_TIER=economy` | 512 token context, minimal LLM calls |
| `OPENKORE_AI_LLM_COST_TIER=standard` | 2K context, normal LLM usage (default) |
| `OPENKORE_AI_LLM_COST_TIER=premium` | 8K context, full LLM reasoning |

Add `OPENKORE_AI_LLM_DAILY_BUDGET_TOKENS=50000` or `OPENKORE_AI_LLM_MAX_CALLS_PER_HOUR=20` to cap usage.

See [AI_sidecar/.env.example](AI_sidecar/.env.example) for all available settings.

---

## Commands

| Command | What it does |
|---------|-------------|
| `./start.sh all` | Start sidecar + all bots + console view (reads `.env` for passwords) |
| `./start.sh sidecar` | Start sidecar only + tail logs |
| `./start.sh bot <name>` | Start one bot by profile name |
| `./start.sh stop` | Kill all processes |
| `./start.sh status` | Show running bots with connection status |
| `./start.sh tail` | Re-attach console view (color-coded multi-tail) |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    start.sh                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Bot 1    │  │ Bot 2    │  │ Bot 3    │           │
│  │ (Perl)   │  │ (Perl)   │  │ (Perl)   │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │ bridge HTTP  │              │                 │
│       ▼              ▼              ▼                 │
│  ┌──────────────────────────────────────────┐        │
│  │       AI Sidecar (FastAPI + Python)       │        │
│  │                                           │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │        │
│  │  │ Reflex   │ │Heuristic │ │  LLM     │  │        │
│  │  │ Rules    │ │ Service  │ │Provider  │  │        │
│  │  │ (17)     │ │(5 domains)│ │Router   │  │        │
│  │  └──────────┘ └──────────┘ └──────────┘  │        │
│  │                                           │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │        │
│  │  │  PDCA    │ │  CrewAI  │ │  Fleet   │  │        │
│  │  │  Loop    │ │  Agents  │ │Coordinator│  │        │
│  │  │(5/30/120s)│ │  (18)   │ │          │  │        │
│  │  └──────────┘ └──────────┘ └──────────┘  │        │
│  │                                           │        │
│  │  SQLite ← → OpenMemory ← → Ollama API    │        │
│  └──────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

## Links

- Vanilla OpenKore: [github.com/OpenKore/openkore](https://github.com/OpenKore/openkore)
- OpenKore wiki: [openkore.com](https://openkore.com)
- DeepSeek API: [platform.deepseek.com](https://platform.deepseek.com)
- OpenAI API: [platform.openai.com](https://platform.openai.com)
- Ollama: [ollama.com](https://ollama.com)
- Discord (support): [discord.gg/zHCKr3rbM](https://discord.gg/zHCKr3rbM)

---

## License

GNU General Public License v2.0 — same as vanilla OpenKore.
