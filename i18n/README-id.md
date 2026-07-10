<p align="center">
  <img alt="openkore AI" src="https://upload.wikimedia.org/wikipedia/commons/b/b5/Kore_2g_logo.png" width="200">
</p>

# openkore **AI**

> Bot Ragnarok Online bertenaga pengambilan keputusan LLM — bukan sekadar makro. **AI**, bukan *bypass*.

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
[![Sponsor](https://img.shields.io/badge/Sponsor-donate-EA4AAA?logo=githubsponsors)](https://github.com/sponsors/iskandarsulaili)
[![GitHub stars](https://img.shields.io/github/stars/iskandarsulaili/openkore-ai-v3)](https://github.com/iskandarsulaili/openkore-ai-v3/stargazers)

🌐 [English](../README.md) | [Português](README-pt-BR.md) | [Tagalog](README-tl.md) | [日本語](README-ja.md) | [한국어](README-ko.md) | [ไทย](README-th.md) | [Indonesia](README-id.md) | [简体中文](README-zh-CN.md)

---

## Apa Ini?

OpenKore yang dimodifikasi dengan **mesin pengambilan keputusan AI** (FastAPI + Python) di samping klien Perl klasik. Bot menggunakan status game real-time + panggilan LLM opsional untuk membuat keputusan.

**Dibandingkan dengan OpenKore asli:**

| OpenKore Asli | openkore **AI** |
|-----------------|----------------|
| Keputusan berbasis makro | Refleks → Heuristik → LLM (3 tingkat) |
| Perilaku tetap | Beradaptasi dari hasil |
| Satu bot | Kawanan multi-bot terkoordinasi |
| Tanpa kontrol biaya | Anggaran per bot, tingkat bertahap |
| Makro komunitas saja | Pembelajaran bersama antar bot |
| Bereaksi terhadap kondisi | Merencanakan dengan siklus PDCA |

---

## Fitur

- **3 tingkat** — 17 aturan refleks → Skor heuristik → LLM (DeepSeek, OpenAI, atau Ollama lokal)
- **Kawanan multi-bot** — Penetapan peran otomatis (tank/healer/dps/crafter)
- **Belajar mandiri** — Meningkat per aksi/peta/monster. Berbagi pengalaman antar bot.
- **Kontrol biaya** — Atur anggaran token harian, batas panggilan per jam, atau nonaktifkan LLM.
- **Mekanik RO** — Party, PVP, GVG, MVP, refine, kartu, quest, crafting, job change
- **Percakapan NPC bertenaga LLM** — Urutan heuristik untuk NPC umum, LLM untuk NPC quest/refine
- **Konsol langsung** — `./start.sh` menampilkan log semua bot + sidecar dalam satu terminal, berwarna
- **Server FastAPI** — **112+ endpoint** di 19 router, middleware auth, koordinasi armada
- **CrewAI** — 18 agen khusus dikelola oleh orkestrator
- **ML Bawah Sadar** — Pembelajaran perilaku yang dihosting sendiri
- **Kompatibel** — Menggunakan format `control/config.txt` standar

---

## Persyaratan

- **Python 3.11+** — Untuk sidecar AI
- **Perl 5** — Untuk klien OpenKore (sudah termasuk)
- **Akun Ragnarok Online** — Di server mana pun yang kompatibel dengan OpenKore
- **Penyedia LLM** (opsional) — Kunci API DeepSeek, kunci API OpenAI, atau instance Ollama lokal

---

## Mulai Cepat (Bot Tunggal)

Atur kredensial di `.env` (di root repositori):

```bash
# format: BOT_<nama>_PASS=<kata_sandi>
cat >> .env << 'EOF'
BOT_botku_PASS=kata_sandiku
EOF
```

```bash
# 1. Siapkan lingkungan Python
cd AI_sidecar
python3 -m venv venv
source venv/bin/activate
pip install -e .
cd ..

# 2. Buat profil bot
mkdir -p .bot_profiles/botku/control
cp control/config.txt .bot_profiles/botku/control/
# Edit .bot_profiles/botku/control/config.txt

# 3. Mulai sidecar + bot
./start.sh sidecar &
./start.sh bot botku &
```

Atau mulai semuanya sekaligus:

```bash
./start.sh all
```

## Mulai Cepat (Banyak Bot)

```bash
mkdir -p .bot_profiles/karakter1/control .bot_profiles/karakter2/control
cp control/config.txt .bot_profiles/karakter1/control/
cp control/config.txt .bot_profiles/karakter2/control/

# Atur kata sandi di .env
cat >> .env << 'EOF'
BOT_karakter1_PASS=sandi1
BOT_karakter2_PASS=sandi2
EOF

# Mulai semuanya:
./start.sh all
```

## Tingkat Biaya

Atur melalui variabel lingkungan `OPENKORE_AI_LLM_COST_TIER` di `AI_sidecar/.env`:

| Pengaturan | Efek |
|------------|------|
| `OPENKORE_AI_LLM_COST_TIER=off` | Hanya refleks + heuristik. Biaya $0. |
| `OPENKORE_AI_LLM_COST_TIER=economy` | 512 token konteks, LLM minimal |
| `OPENKORE_AI_LLM_COST_TIER=standard` | 2K konteks, LLM normal (bawaan) |
| `OPENKORE_AI_LLM_COST_TIER=premium` | 8K konteks, LLM penuh |

Lihat [AI_sidecar/.env.example](../AI_sidecar/.env.example) untuk semua pengaturan yang tersedia.

## Perintah

| Perintah | Kegunaan |
|----------|----------|
| `./start.sh all` | Mulai sidecar + semua bot + konsol |
| `./start.sh sidecar` | Mulai sidecar saja |
| `./start.sh bot <nama>` | Mulai satu bot berdasarkan nama profil |
| `./start.sh stop` | Hentikan semua proses |
| `./start.sh status` | Tampilkan status bot |
| `./start.sh tail` | Sambungkan ulang konsol |

## Links

- OpenKore original: [github.com/OpenKore/openkore](https://github.com/OpenKore/openkore)
- Dokumentasi OpenKore: [openkore.com](https://openkore.com)
- DeepSeek API: [platform.deepseek.com](https://platform.deepseek.com)
- Discord: [discord.gg/zHCKr3rbM](https://discord.gg/zHCKr3rbM)

---

## Lisensi

GNU General Public License v2.0 — sama dengan OpenKore original.
