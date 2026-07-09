# openkore **IA**

> Bot de Ragnarok Online com motor de decisão por IA — não apenas macros. **IA**, não *bypass*.

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
[![GitHub stars](https://img.shields.io/github/stars/iskandarsulaili/openkore-ai-v3)](https://github.com/iskandarsulaili/openkore-ai-v3/stargazers)

🌐 [English](../README.md) | [Português](README-pt-BR.md) | [Tagalog](README-tl.md) | [日本語](README-ja.md) | [한국어](README-ko.md) | [ไทย](README-th.md) | [Indonesia](README-id.md) | [简体中文](README-zh-CN.md)

---

## O que é?

Uma versão modificada do OpenKore que adiciona um **motor de decisão por IA** (FastAPI + Python) junto ao cliente Perl clássico. Os bots usam estado do jogo em tempo real + chamadas opcionais de LLM para tomar decisões.

**Comparado ao OpenKore original:**

| OpenKore Original | openkore **IA** |
|-----------------|----------------|
| Decisões por macros | Reflexo → Heurística → LLM (3 níveis) |
| Comportamento fixo | Auto-adaptação |
| Um bot por vez | Enxame multi-bot |
| Sem controle de custos | Orçamento diário por bot |
| Macros da comunidade | Aprendizado compartilhado |
| Reage a condições | Planeja com PDCA |

---

## Fitur

- Motor 3 níveis: 25 regras de reflexo → 17 perfis heurísticos → LLM
- Enxame multi-bot com coordenação automática
- Auto-aprendizado por ação/mapa/monstro
- Controle de custos: off/economy/standard/premium
- 30+ mecânicas: party, PVP, GVG, MVP, refine, cartas
- Diálogo com NPCs via LLM
- Console ao vivo `start.sh`
- 25 endpoints FastAPI
- Compatível com configs originais

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
