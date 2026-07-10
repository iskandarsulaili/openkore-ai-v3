<p align="center">
  <img alt="openkore AI" src="https://upload.wikimedia.org/wikipedia/commons/b/b5/Kore_2g_logo.png" width="200">
</p>

# openkore **AI**

> บอท Ragnarok Online ที่ขับเคลื่อนด้วย LLM — ไม่ใช่แค่มาโคร **AI** ไม่ใช่ *bypass*

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
[![Sponsor](https://img.shields.io/badge/Sponsor-donate-EA4AAA?logo=githubsponsors)](https://github.com/sponsors/iskandarsulaili)
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
| บอทเดียว | ฝูงบอทหลายตัวประสานงานกัน |
| ไม่มีการควบคุมค่าใช้จ่าย | งบประมาณต่อบอท ระดับลดหลั่น |
| เฉพาะมาโครชุมชน | เรียนรู้ร่วมกันข้ามบอท |
| ตอบสนองต่อเงื่อนไข | วางแผนล่วงหน้าด้วย PDCA |

---

## คุณสมบัติ

- **3 ระดับ** — 17 กฎรีเฟล็กซ์ → คะแนนฮิวริสติก → LLM (DeepSeek, OpenAI หรือ Ollama ในเครื่อง)
- **ฝูงบอทหลายตัว** — กำหนดบทบาทอัตโนมัติ (แทงค์/ฮีลเลอร์/DPS/คราฟเตอร์)
- **เรียนรู้ด้วยตนเอง** — ปรับปรุงตามการกระทำ/แผนที่/มอนสเตอร์ แบ่งปันประสบการณ์ข้ามบอท
- **ควบคุมค่าใช้จ่าย** — กำหนดงบประมาณโทเค็นรายวัน จำกัดการเรียกต่อชั่วโมง หรือปิด LLM
- **กลไก RO** — ปาร์ตี้, PVP, GVG, MVP, หลอม, การ์ด, เควสต์, คราฟต์, เปลี่ยนอาชีพ
- **สนทนากับ NPC ด้วย LLM** — ลำดับฮิวริสติกสำหรับ NPC ทั่วไป, LLM สำหรับ NPC เควสต์/หลอม
- **คอนโซลสด** — `./start.sh` แสดง logs ของบอททั้งหมด + sidecar ในเทอร์มินัลเดียว สีสันต่างกัน
- **เซิร์ฟเวอร์ FastAPI** — **112+ ปลายทาง** ใน 19 เราเตอร์, มิดเดิลแวร์ตรวจสอบสิทธิ์, การประสานงานฝูงบิน
- **CrewAI** — 18 เอเจนต์เฉพาะทางจัดการโดยออร์เคสเตรเตอร์
- **ML จิตใต้สำนึก** — การเรียนรู้พฤติกรรมโฮสต์เอง
- **เข้ากันได้** — ใช้รูปแบบ `control/config.txt` มาตรฐาน

---

## ข้อกำหนด

- **Python 3.11+** — สำหรับ AI sidecar
- **Perl 5** — สำหรับ OpenKore client (มาพร้อมแล้ว)
- **บัญชี Ragnarok Online** — บนเซิร์ฟเวอร์ที่เข้ากันได้กับ OpenKore
- **ผู้ให้บริการ LLM** (ไม่จำเป็น) — คีย์ API DeepSeek, คีย์ API OpenAI หรือ Ollama ในเครื่อง

---

## เริ่มต้นใช้งาน (บอทเดียว)

ตั้งค่าข้อมูลรับรองใน `.env` (ที่รากของ repository):

```bash
# รูปแบบ: BOT_<ชื่อ>_PASS=<รหัสผ่าน>
cat >> .env << 'EOF'
BOT_บอทฉัน_PASS=รหัสผ่านฉัน
EOF
```

```bash
# 1. ติดตั้งสภาพแวดล้อม Python
cd AI_sidecar
python3 -m venv venv
source venv/bin/activate
pip install -e .
cd ..

# 2. สร้างโปรไฟล์บอท
mkdir -p .bot_profiles/mybot/control
cp control/config.txt .bot_profiles/mybot/control/
# แก้ไข .bot_profiles/mybot/control/config.txt

# 3. เริ่ม sidecar + บอท
./start.sh sidecar &
./start.sh bot mybot &
```

หรือเริ่มทุกอย่างพร้อมกัน:

```bash
./start.sh all
```

## เริ่มต้นใช้งาน (หลายบอท)

```bash
mkdir -p .bot_profiles/char1/control .bot_profiles/char2/control
cp control/config.txt .bot_profiles/char1/control/
cp control/config.txt .bot_profiles/char2/control/

# ตั้งรหัสผ่านใน .env
cat >> .env << 'EOF'
BOT_char1_PASS=pass1
BOT_char2_PASS=pass2
EOF

# เริ่มทุกอย่าง:
./start.sh all
```

## ระดับค่าใช้จ่าย

ตั้งค่าผ่านตัวแปรสภาพแวดล้อม `OPENKORE_AI_LLM_COST_TIER` ใน `AI_sidecar/.env`:

| การตั้งค่า | ผล |
|------------|------|
| `OPENKORE_AI_LLM_COST_TIER=off` | เฉพาะรีเฟล็กซ์ + ฮิวริสติก ไม่มีค่าใช้จ่าย |
| `OPENKORE_AI_LLM_COST_TIER=economy` | 512 โทเค็นบริบท, LLM น้อยที่สุด |
| `OPENKORE_AI_LLM_COST_TIER=standard` | 2K บริบท, LLM ปกติ (ค่าเริ่มต้น) |
| `OPENKORE_AI_LLM_COST_TIER=premium` | 8K บริบท, LLM เต็มรูปแบบ |

ดู [AI_sidecar/.env.example](../AI_sidecar/.env.example) สำหรับการตั้งค่าทั้งหมด

## คำสั่ง

| คำสั่ง | การทำงาน |
|----------|--------|
| `./start.sh all` | เริ่ม sidecar + บอททั้งหมด + คอนโซล |
| `./start.sh sidecar` | เริ่มเฉพาะ sidecar |
| `./start.sh bot <ชื่อ>` | เริ่มบอทตามชื่อโปรไฟล์ |
| `./start.sh stop` | หยุดกระบวนการทั้งหมด |
| `./start.sh status` | แสดงสถานะของบอท |
| `./start.sh tail` | เชื่อมต่อคอนโซลอีกครั้ง |

## ลิงก์

- OpenKore ดั้งเดิม: [github.com/OpenKore/openkore](https://github.com/OpenKore/openkore)
- เอกสาร OpenKore: [openkore.com](https://openkore.com)
- DeepSeek API: [platform.deepseek.com](https://platform.deepseek.com)
- Discord: [discord.gg/zHCKr3rbM](https://discord.gg/zHCKr3rbM)

---

## สัญญาอนุญาต

GNU General Public License v2.0 — เช่นเดียวกับ OpenKore ดั้งเดิม
