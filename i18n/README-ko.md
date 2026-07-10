<p align="center">
  <img alt="openkore AI" src="https://upload.wikimedia.org/wikipedia/commons/b/b5/Kore_2g_logo.png" width="200">
</p>

# openkore **AI**

> LLM 기반 의사결정을 탑재한 Ragnarok Online 봇 — 단순한 매크로가 아닙니다. **AI**, *bypass*가 아닙니다.

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
[![Sponsor](https://img.shields.io/badge/Sponsor-donate-EA4AAA?logo=githubsponsors)](https://github.com/sponsors/iskandarsulaili)
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
| 단일 봇 | 멀티봇 떼 협력 |
| 비용 관리 없음 | 봇별 예산, 단계별 티어 |
| 커뮤니티 매크로만 | 봇 간 공유 학습 |
| 조건에 반응 | PDCA로 선제 계획 |

---

## 기능

- **3단계 엔진** — 17개 반사 규칙 → 휴리스틱 점수 → LLM (DeepSeek, OpenAI 또는 로컬 Ollama)
- **멀티봇 함대** — 자동 역할 할당 (탱크/힐러/DPS/크래프터)
- **자기 학습** — 행동/맵/몬스터별 개선. 봇 간 경험 공유.
- **비용 제어** — 일일 토큰 예산, 시간당 호출 제한, 또는 LLM 비활성화.
- **RO 메카닉** — 파티, PVP, GVG, MVP, 정제, 카드, 퀘스트, 크래프팅, 전직
- **LLM 기반 NPC 대화** — 일반 NPC는 휴리스틱, 퀘스트/정제 NPC는 LLM
- **라이브 콘솔** — `./start.sh`가 모든 봇 + 사이드카 로그를 색상별로 하나의 터미널에 표시
- **FastAPI 서버** — 19개 라우터에 **112+개 엔드포인트**, 인증 미들웨어, 함대 조정
- **CrewAI** — 18개 전문 에이전트를 오케스트레이터가 관리
- **ML 잠재의식** — 자체 호스팅 행동 학습
- **호환성** — 표준 `control/config.txt` 형식 사용

---

## 요구 사항

- **Python 3.11+** — AI 사이드카용
- **Perl 5** — OpenKore 클라이언트용 (번들됨)
- **Ragnarok Online 계정** — OpenKore 호환 서버
- **LLM 제공자** (선택사항) — DeepSeek API 키, OpenAI API 키 또는 로컬 Ollama

---

## 빠른 시작 (단일 봇)

`.env` 파일에 인증 정보 설정 (저장소 루트):

```bash
# 형식: BOT_<이름>_PASS=<비밀번호>
cat >> .env << 'EOF'
BOT_내봇_PASS=내_비밀번호
EOF
```

```bash
# 1. Python 환경 설정
cd AI_sidecar
python3 -m venv venv
source venv/bin/activate
pip install -e .
cd ..

# 2. 봇 프로필 생성
mkdir -p .bot_profiles/mybot/control
cp control/config.txt .bot_profiles/mybot/control/
# .bot_profiles/mybot/control/config.txt 편집

# 3. 사이드카 + 봇 시작
./start.sh sidecar &
./start.sh bot mybot &
```

또는 한 번에 모두 시작:

```bash
./start.sh all
```

## 빠른 시작 (다중 봇)

```bash
mkdir -p .bot_profiles/char1/control .bot_profiles/char2/control
cp control/config.txt .bot_profiles/char1/control/
cp control/config.txt .bot_profiles/char2/control/

# .env에 비밀번호 설정
cat >> .env << 'EOF'
BOT_char1_PASS=pass1
BOT_char2_PASS=pass2
EOF

# 모두 시작:
./start.sh all
```

## 요금 등급

환경 변수 `OPENKORE_AI_LLM_COST_TIER`로 설정 (`AI_sidecar/.env`):

| 설정 | 효과 |
|------|-------|
| `OPENKORE_AI_LLM_COST_TIER=off` | 반사 + 휴리스틱만. 비용 0. |
| `OPENKORE_AI_LLM_COST_TIER=economy` | 512 토큰 컨텍스트, 최소 LLM |
| `OPENKORE_AI_LLM_COST_TIER=standard` | 2K 컨텍스트, 일반 LLM (기본값) |
| `OPENKORE_AI_LLM_COST_TIER=premium` | 8K 컨텍스트, 전체 LLM 추론 |

모든 설정은 [AI_sidecar/.env.example](../AI_sidecar/.env.example) 참조.

## 명령어

| 명령어 | 설명 |
|--------|-------|
| `./start.sh all` | 사이드카 + 모든 봇 + 콘솔 시작 |
| `./start.sh sidecar` | 사이드카만 시작 |
| `./start.sh bot <이름>` | 프로필 이름으로 봇 시작 |
| `./start.sh stop` | 모든 프로세스 중지 |
| `./start.sh status` | 봇 상태 표시 |
| `./start.sh tail` | 콘솔 다시 연결 |

## 링크

- OpenKore original: [github.com/OpenKore/openkore](https://github.com/OpenKore/openkore)
- OpenKore 문서: [openkore.com](https://openkore.com)
- DeepSeek API: [platform.deepseek.com](https://platform.deepseek.com)
- Discord: [discord.gg/zHCKr3rbM](https://discord.gg/zHCKr3rbM)

---

## 라이선스

GNU General Public License v2.0 — OpenKore original과 동일.
