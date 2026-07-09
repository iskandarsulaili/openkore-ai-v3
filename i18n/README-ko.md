# openkore **AI**

> LLM 기반 의사결정을 탑재한 Ragnarok Online 봇 — 단순한 매크로가 아닙니다. **AI**, *bypass*가 아닙니다.

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
[![GitHub stars](https://img.shields.io/github/stars/iskandarsulaili/openkore-ai-v3)](https://github.com/iskandarsulaili/openkore-ai-v3/stargazers)

🌐 [English](../README.md) | [Português](README-pt-BR.md) | [Tagalog](README-tl.md) | [日本語](README-ja.md) | [한국어](README-ko.md) | [ไทย](README-th.md) | [Indonesia](README-id.md) | [简体中文](README-zh-CN.md)

---

## 이게 뭔가요?

클래식 Perl 클라이언트와 함께 **AI 의사결정 엔진**(FastAPI + Python)을 추가한 수정된 OpenKore입니다. 봇은 하드코딩된 매크로 대신 실시간 게임 상태와 선택적 LLM 호출을 사용하여 결정을 내립니다.

**기존 OpenKore와 비교:**

| 기존 OpenKore | openkore **AI** |
|-----------------|----------------|
| 매크로 기반 결정 | 반사 → 휴리스틱 → LLM (3단계) |
| 고정된 동작 | 결과에서 자기 적응 |
| 단일 봇 | 멀티봇 떼 |
| 비용 관리 없음 | 봇별 예산, 단계별 티어 |
| 커뮤니티 매크로만 | 봇 간 공유 학습 |
| 조건에 반응 | PDCA로 선제 계획 |

---

## Fitur

- 3단계 엔진: 25개 반사 규칙 → 17개 휴리스틱 → LLM
- 멀티봇 함대: 자동 역할 할당
- 자기 학습: 행동/맵/몬스터별 개선
- 비용 제어: off/economy/standard/premium
- 30+ RO 메카닉 지원
- LLM 기반 NPC 대화
- start.sh 라이브 콘솔
- 25개 FastAPI 엔드포인트
- 기존 OpenKore 설정 호환

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
