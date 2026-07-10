<p align="center">
  <img alt="openkore AI" src="https://upload.wikimedia.org/wikipedia/commons/b/b5/Kore_2g_logo.png" width="200">
</p>

# openkore **AI**

> Ragnarok Online bot na pinalakas ng LLM decision-making — hindi lang macros. **AI**, hindi *bypass*.

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
[![Sponsor](https://img.shields.io/badge/Sponsor-donate-EA4AAA?logo=githubsponsors)](https://github.com/sponsors/iskandarsulaili)
[![GitHub stars](https://img.shields.io/github/stars/iskandarsulaili/openkore-ai-v3)](https://github.com/iskandarsulaili/openkore-ai-v3/stargazers)

🌐 [English](../README.md) | [Português](README-pt-BR.md) | [Tagalog](README-tl.md) | [日本語](README-ja.md) | [한국어](README-ko.md) | [ไทย](README-th.md) | [Indonesia](README-id.md) | [简体中文](README-zh-CN.md)

---

## Ano Ito?

Isang modified OpenKore na may **AI decision engine** (FastAPI + Python) kasama ng classic Perl client. Gumagamit ang mga bot ng real-time game state + optional LLM calls para magdesisyon.

**Kumpara sa vanilla OpenKore:**

| Vanilla OpenKore | openkore **AI** |
|-----------------|----------------|
| Macro-based decisions | Reflex → Heuristic → LLM (3-tier) |
| Fixed behavior | Self-adapts sa outcomes |
| Single bot | Multi-bot swarm coordination |
| Walang cost control | Per-bot budget, graduated tiers |
| Community macros lang | Cross-bot shared learning |
| Nagre-react sa conditions | Nagpaplano gamit PDCA loop |

---

## Mga Tampok

- **3-tier engine** — 17 reflex rules → Heuristic scoring → LLM (DeepSeek, OpenAI, o local Ollama)
- **Multi-bot fleet** — Auto-role assignment (tank/healer/dps/crafter), message relay
- **Self-learning** — Nag-iimprove per action/map/monster. Cross-bot experience sharing.
- **Cost control** — Set daily token budget, hourly limit, o i-off ang LLM.
- **RO mechanics** — Party, PVP, GVG, MVP, refine, cards, quests, crafting, job change
- **NPC dialog powered ng LLM** — Heuristic sequences para sa common NPCs, LLM para sa quest/refine NPCs
- **Live console** — `./start.sh` stream ng lahat ng bot + sidecar logs sa isang terminal, color-coded
- **FastAPI server** — **112+ endpoints** sa 19 routers, auth middleware, fleet coordination
- **CrewAI** — 18 specialized agents na managed ng orchestrator
- **ML Subconscious** — Self-hosted behavior learning
- **Compatible** — Gumagamit ng standard `control/config.txt` format

---

## Mga Kailangan

- **Python 3.11+** — Para sa AI sidecar
- **Perl 5** — Para sa OpenKore client (kasama na)
- **Ragnarok Online account** — Sa kahit anong server na compatible sa OpenKore
- **LLM provider** (optional) — DeepSeek API key, OpenAI API key, o local Ollama instance

---

## Mabilis na Pagsisimula (Isang Bot)

Mag-set up ng credentials sa `.env` (sa root ng repository):

```bash
# format: BOT_<pangalan>_PASS=<password>
cat >> .env << 'EOF'
BOT_akingbot_PASS=aking_password
EOF
```

```bash
# 1. I-set up ang Python environment
cd AI_sidecar
python3 -m venv venv
source venv/bin/activate
pip install -e .
cd ..

# 2. Gumawa ng bot profile
mkdir -p .bot_profiles/akingbot/control
cp control/config.txt .bot_profiles/akingbot/control/
# I-edit ang .bot_profiles/akingbot/control/config.txt

# 3. Simulan ang sidecar + bot
./start.sh sidecar &
./start.sh bot akingbot &
```

O simulan lahat nang sabay:

```bash
./start.sh all
```

## Mabilis na Pagsisimula (Maraming Bot)

```bash
mkdir -p .bot_profiles/char1/control .bot_profiles/char2/control
cp control/config.txt .bot_profiles/char1/control/
cp control/config.txt .bot_profiles/char2/control/

# I-set ang passwords sa .env
cat >> .env << 'EOF'
BOT_char1_PASS=pass1
BOT_char2_PASS=pass2
EOF

# Simulan lahat:
./start.sh all
```

## Antas ng Gastos

I-set gamit ang environment variable `OPENKORE_AI_LLM_COST_TIER` sa `AI_sidecar/.env`:

| Setting | Epekto |
|---------|--------|
| `OPENKORE_AI_LLM_COST_TIER=off` | Reflex + heuristic lang. Zero cost. |
| `OPENKORE_AI_LLM_COST_TIER=economy` | 512 token context, minimal LLM |
| `OPENKORE_AI_LLM_COST_TIER=standard` | 2K context, normal LLM (default) |
| `OPENKORE_AI_LLM_COST_TIER=premium` | 8K context, full LLM reasoning |

Tingnan ang [AI_sidecar/.env.example](../AI_sidecar/.env.example) para sa lahat ng available na setting.

## Mga Utos

| Utos | Gamit |
|------|-------|
| `./start.sh all` | Simulan sidecar + lahat ng bot + console |
| `./start.sh sidecar` | Simulan sidecar lang |
| `./start.sh bot <pangalan>` | Simulan isang bot sa profile name |
| `./start.sh stop` | Ihinto lahat ng proseso |
| `./start.sh status` | Ipakita ang status ng mga bot |
| `./start.sh tail` | Muling kumonekta sa console |

## Mga Link

- OpenKore original: [github.com/OpenKore/openkore](https://github.com/OpenKore/openkore)
- OpenKore documentation: [openkore.com](https://openkore.com)
- DeepSeek API: [platform.deepseek.com](https://platform.deepseek.com)
- Discord: [discord.gg/zHCKr3rbM](https://discord.gg/zHCKr3rbM)

---

## Lisensya

GNU General Public License v2.0 — katulad ng OpenKore original.
