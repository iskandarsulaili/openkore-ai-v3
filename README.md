<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/iskandarsulaili/openkore-ai-v3/master/assets/logo-dark.png">
  <img alt="openkore AI" src="https://upload.wikimedia.org/wikipedia/commons/b/b5/Kore_2g_logo.png" width="200">
</picture>

# openkore **AI**

> Ragnarok Online bot powered by LLM decision-making — not just macros.
> **AI**, not *bypass*.

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
[![GitHub stars](https://img.shields.io/github/stars/iskandarsulaili/openkore-ai-v3)](https://github.com/iskandarsulaili/openkore-ai-v3/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/iskandarsulaili/openkore-ai-v3)](https://github.com/iskandarsulaili/openkore-ai-v3/issues)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![Perl](https://img.shields.io/badge/Perl-5.38-blue?logo=perl)](https://perl.org)

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

- **3-tier decision engine** — Fast reflex rules (25 built-in) → Heuristic scoring (17 profiles) → LLM (DeepSeek optional)
- **Multi-bot fleet** — Bots share state, auto-assign roles (tank/healer/dps/crafter), relay messages
- **Self-learning** — Tracks outcomes per action/map/monster. Improves over time. Cross-bot experience sharing.
- **Cost control** — Set daily token budget, hourly call limit, or disable LLM entirely. Configurable per environment.
- **30+ RO mechanics** — Party, PVP, GVG, MVP, refine, cards, quests, crafting, job change, stat/skill allocation
- **NPC interaction** — Auto-talk for warps, quests, job changes, storage, vending, Kafra
- **Live console** — `start.sh` streams all bots + sidecar logs in one terminal, color-coded
- **API server** — FastAPI with 24 endpoints, auth middleware, fleet coordination
- **Works with vanilla OpenKore configs** — Uses standard `control/config.txt` format

---

## Required

- **Python 3.11+** — For the AI sidecar
- **Perl 5** — For the OpenKore client (bundled)
- **Ragnarok Online account** — On any server compatible with OpenKore
- **DeepSeek API key** (optional) — For LLM-powered decisions. Get one at [platform.deepseek.com](https://platform.deepseek.com)

---

## Quick Start (Single Bot)

```bash
# 1. Clone
git clone https://github.com/iskandarsulaili/openkore-ai-v3.git
cd openkore-ai-v3

# 2. Set up credentials
cat >> .env << 'EOF'
BOT_kicapmasin2_PASS=your_password
EOF

# 3. Install Python sidecar
cd AI_sidecar
python3 -m venv venv
source venv/bin/activate
pip install -e .
cd ..

# 4. Set server details in control/config.txt
#    master YourServer
#    username your_username
#    password your_password
#    char 0

# 5. Start
source AI_sidecar/venv/bin/activate
nohup python -m ai_sidecar.app > logs/sidecar.log 2>&1 &

perl -I src openkore.pl
```

## Quick Start (Multi-Bot Fleet)

```bash
# Same as above, but set up one profile per account:

mkdir -p profiles/char1/control profiles/char2/control
cp control/config.txt profiles/char1/control/
cp control/config.txt profiles/char2/control/
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

Set in `AI_sidecar/.env`:

| Setting | Effect |
|---------|--------|
| `llm_cost_tier=off` | Reflex + heuristic only. $0. |
| `llm_cost_tier=economy` | 512 token context, minimal LLM |
| `llm_cost_tier=standard` | 2K context, normal LLM (default) |
| `llm_cost_tier=premium` | 8K context, full LLM reasoning |

Add `llm_daily_budget_tokens=50000` or `llm_max_calls_per_hour=20` to cap usage.

---

## Commands

| Command | What it does |
|---------|-------------|
| `./start.sh all` | Start sidecar + all bots + console view |
| `./start.sh status` | Show running bots |
| `./start.sh stop` | Stop everything |
| `./start.sh tail` | Re-attach console view |
| `perl -I src openkore.pl --control=profiles/<name>/control` | Start one bot with profile |

---

## Links

- Vanilla OpenKore: [github.com/OpenKore/openkore](https://github.com/OpenKore/openkore)
- OpenKore wiki: [openkore.com](https://openkore.com)
- DeepSeek API: [platform.deepseek.com](https://platform.deepseek.com)
- Discord (support): [discord.gg/zHCKr3rbM](https://discord.gg/zHCKr3rbM)

---

## License

GNU General Public License v2.0 — same as vanilla OpenKore.
