# openkore **AI**

> บอท Ragnarok Online ที่ขับเคลื่อนด้วย LLM — ไม่ใช่แค่มาโคร **AI** ไม่ใช่ *bypass*

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
[![GitHub stars](https://img.shields.io/github/stars/iskandarsulaili/openkore-ai-v3)](https://github.com/iskandarsulaili/openkore-ai-v3/stargazers)

🌐 [English](../README.md) | [Português](README-pt-BR.md) | [Tagalog](README-tl.md) | [日本語](README-ja.md) | [한국어](README-ko.md) | [ไทย](README-th.md) | [Indonesia](README-id.md) | [简体中文](README-zh-CN.md)

---

## นี่คืออะไร?

OpenKore ที่ถูกดัดแปลงเพื่อเพิ่ม **ระบบตัดสินใจด้วย AI** (FastAPI + Python) ควบคู่ไปกับ Perl client แบบดั้งเดิม บอทใช้สถานะเกมแบบเรียลไทม์ + การเรียก LLM เพื่อตัดสินใจแทนการทำตามมาโครแบบตายตัว

**เปรียบเทียบกับ OpenKore ดั้งเดิม:**

| OpenKore ดั้งเดิม | openkore **AI** |
|-----------------|----------------|
| ตัดสินใจด้วยมาโคร | รีเฟล็กซ์ → ฮิวริสติก → LLM (3 ระดับ) |
| พฤติกรรมคงที่ | ปรับเปลี่ยนตามผลลัพธ์ |
| บอทเดียว | ฝูงบอทหลายตัว |
| ไม่มีการควบคุมค่าใช้จ่าย | งบประมาณต่อบอท |
| เฉพาะมาโครชุมชน | เรียนรู้ร่วมกันข้ามบอท |
| ตอบสนองต่อเงื่อนไข | วางแผนล่วงหน้าด้วย PDCA |

---

## Fitur

- 3 ระดับ: 25 กฎรีเฟล็กซ์ → 17 โปรไฟล์ฮิวริสติก → LLM
- ฝูงบอทหลายตัว: กำหนดบทบาทอัตโนมัติ
- เรียนรู้ด้วยตนเอง: ปรับปรุงตามการกระทำ/แผนที่/มอนสเตอร์
- ควบคุมค่าใช้จ่าย: off/economy/standard/premium
- รองรับ 30+ กลไก RO
- สนทนากับ NPC ด้วย LLM
- คอนโซลสดด้วย start.sh
- 25 ปลายทาง FastAPI
- เข้ากันได้กับการตั้งค่า OpenKore ดั้งเดิม

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
