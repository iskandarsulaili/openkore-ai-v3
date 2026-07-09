# openkore **AI**

> Ragnarok Online bot na pinalakas ng LLM decision-making — hindi lang macros. **AI**, hindi *bypass*.

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
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
| Single bot | Multi-bot swarm |
| Walang cost control | Per-bot budget, graduated tiers |
| Community macros lang | Cross-bot shared learning |
| Nagre-react sa conditions | Nagpaplano gamit PDCA |

---

## Fitur

- 3-tier engine: 25 reflex rules → 17 heuristic profiles → LLM
- Multi-bot fleet na may auto-role assignment
- Self-learning: nag-iimprove sa pagdaan ng panahon
- Cost control: off/economy/standard/premium
- 30+ RO mechanics: party, PVP, GVG, MVP, refine, cards
- NPC dialog na powered ng LLM
- Live console sa `start.sh`
- 25 FastAPI endpoints
- Compatible sa original OpenKore configs

---

## Persyaratan

- **Python 3.11+** — Untuk sidecar AI
- **Perl 5** — Untuk klien OpenKore (sudah termasuk)
- **Akun Ragnarok Online** — Di server mana pun yang kompatibel dengan OpenKore
- **Kunci API DeepSeek** (opsional) — Untuk keputusan via LLM. Dapatkan di [platform.deepseek.com](https://platform.deepseek.com)

---

## Mulai Cepat (Bot Tunggal)

```bash
git clone https://github.com/iskandarsulaili/openkore-ai-v3.git
cd openkore-ai-v3

# Atur kredensial
cat >> .env << 'EOF'
BOT_karaktermu_PASS=sandi_anda
EOF

# Pasang sidecar Python
cd AI_sidecar
python3 -m venv venv
source venv/bin/activate
pip install -e .
cd ..

# Atur control/config.txt dengan server dan akun Anda

# Mulai
source AI_sidecar/venv/bin/activate
nohup python -m ai_sidecar.app > logs/sidecar.log 2>&1 &
perl -I src openkore.pl
```

## Mulai Cepat (Banyak Bot)

```bash
mkdir -p profiles/karakter1/control profiles/karakter2/control
cp control/config.txt profiles/karakter1/control/
cp control/config.txt profiles/karakter2/control/

# Atur sandi di .env
cat >> .env << 'EOF'
BOT_karakter1_PASS=sandi1
BOT_karakter2_PASS=sandi2
EOF

# Mulai semuanya:
./start.sh all
```

## Tingkat Biaya

Atur di `AI_sidecar/.env`:

| Pengaturan | Efek |
|------------|------|
| `llm_cost_tier=off` | Hanya refleks + heuristik. $0. |
| `llm_cost_tier=economy` | 512 token konteks, LLM minimal |
| `llm_cost_tier=standard` | 2K konteks, LLM normal (bawaan) |
| `llm_cost_tier=premium` | 8K konteks, LLM penuh |

## Perintah

| Perintah | Kegunaan |
|----------|----------|
| `./start.sh all` | Mulai sidecar + semua bot + konsol |
| `./start.sh status` | Lihat bot yang berjalan |
| `./start.sh stop` | Hentikan semua |
| `./start.sh tail` | Sambungkan ulang konsol |
| `perl -I src openkore.pl --control=.bot_profiles/<nama>/control` | Jalankan satu bot dengan profil |


## Links

- OpenKore original: [github.com/OpenKore/openkore](https://github.com/OpenKore/openkore)
- Dokumentasi OpenKore: [openkore.com](https://openkore.com)
- API DeepSeek: [platform.deepseek.com](https://platform.deepseek.com)
- Discord: [discord.gg/zHCKr3rbM](https://discord.gg/zHCKr3rbM)

---

## Lisensi

GNU General Public License v2.0 — sama dengan OpenKore original.
