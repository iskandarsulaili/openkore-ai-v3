<p align="center">
  <img alt="openkore IA" src="https://upload.wikimedia.org/wikipedia/commons/b/b5/Kore_2g_logo.png" width="200">
</p>

# openkore **IA**

> Bot de Ragnarok Online com motor de decisão por IA — não apenas macros. **IA**, não *bypass*.

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
[![Sponsor](https://img.shields.io/badge/Sponsor-donate-EA4AAA?logo=githubsponsors)](https://github.com/sponsors/iskandarsulaili)
[![GitHub stars](https://img.shields.io/github/stars/iskandarsulaili/openkore-ai-v3)](https://github.com/iskandarsulaili/openkore-ai-v3/stargazers)

🌐 [English](../README.md) | [Português](README-pt-BR.md) | [Tagalog](README-tl.md) | [日本語](README-ja.md) | [한국어](README-ko.md) | [ไทย](README-th.md) | [Indonesia](README-id.md) | [简体中文](README-zh-CN.md)

---

## O que é?

Uma versão modificada do OpenKore que adiciona um **motor de decisão por IA** (FastAPI + Python) junto ao cliente Perl clássico. Os bots usam estado do jogo em tempo real + chamadas opcionais de LLM para tomar decisões.

**Comparado ao OpenKore original:**

| OpenKore Original | openkore **IA** |
|-----------------|----------------|
| Decisões por macros | Reflexo → Heurística → LLM (3 níveis) |
| Comportamento fixo | Auto-adaptação baseada em resultados |
| Um bot por vez | Enxame multi-bot coordenado |
| Sem controle de custos | Orçamento diário por bot, níveis graduais |
| Macros da comunidade | Aprendizado compartilhado entre bots |
| Reage a condições | Planeja com ciclo PDCA |

---

## Recursos

- **Motor 3 níveis** — 17 regras de reflexo → Pontuação heurística → LLM (DeepSeek, OpenAI ou Ollama local)
- **Enxame multi-bot** — Coordenação automática, atribuição de funções (tank/healer/dps/crafter)
- **Auto-aprendizado** — Melhora por ação/mapa/monstro ao longo do tempo. Experiência compartilhada entre bots.
- **Controle de custos** — Configure orçamento diário de tokens, limite por hora, ou desative LLM.
- **30+ mecânicas RO** — Party, PVP, GVG, MVP, refine, cartas, quests, crafting, troca de classe
- **Diálogo com NPCs via LLM** — Sequências heurísticas para NPCs comuns, LLM para quests/refino
- **Console ao vivo** — `./start.sh` exibe logs de todos os bots + sidecar em um terminal, colorido
- **API FastAPI** — **112+ endpoints** em 19 roteadores, middleware de autenticação, coordenação de frota
- **CrewAI** — 18 agentes especializados gerenciados por orquestrador
- **ML Subconsciente** — Aprendizado de comportamento auto-hospedado
- **Compatível** — Usa formato padrão `control/config.txt`

---

## Requisitos

- **Python 3.11+** — Para o sidecar AI
- **Perl 5** — Para o cliente OpenKore (incluído)
- **Conta Ragnarok Online** — Em qualquer servidor compatível com OpenKore
- **Provedor LLM** (opcional) — Chave API DeepSeek, OpenAI ou instância Ollama local

---

## Início Rápido (Bot Único)

Configure as credenciais em `.env` (na raiz do repositório):

```bash
# formato: BOT_<nome>_PASS=<senha>
cat >> .env << 'EOF'
BOT_meubot_PASS=minha_senha
EOF
```

```bash
# 1. Configure o ambiente Python
cd AI_sidecar
python3 -m venv venv
source venv/bin/activate
pip install -e .
cd ..

# 2. Crie um perfil de bot
mkdir -p .bot_profiles/meubot/control
cp control/config.txt .bot_profiles/meubot/control/
# Edite .bot_profiles/meubot/control/config.txt com servidor e conta

# 3. Inicie sidecar + bot
./start.sh sidecar &
./start.sh bot meubot &
```

Ou inicie tudo de uma vez:

```bash
./start.sh all
```

## Início Rápido (Vários Bots)

```bash
mkdir -p .bot_profiles/char1/control .bot_profiles/char2/control
cp control/config.txt .bot_profiles/char1/control/
cp control/config.txt .bot_profiles/char2/control/

# Configure senhas no .env
cat >> .env << 'EOF'
BOT_char1_PASS=senha1
BOT_char2_PASS=senha2
EOF

# Inicie tudo:
./start.sh all
```

## Níveis de Custo

Configure via variável de ambiente `OPENKORE_AI_LLM_COST_TIER` em `AI_sidecar/.env`:

| Configuração | Efeito |
|------------|------|
| `OPENKORE_AI_LLM_COST_TIER=off` | Apenas reflexo + heurística. Custo zero. |
| `OPENKORE_AI_LLM_COST_TIER=economy` | 512 tokens de contexto, LLM mínimo |
| `OPENKORE_AI_LLM_COST_TIER=standard` | 2K contexto, LLM normal (padrão) |
| `OPENKORE_AI_LLM_COST_TIER=premium` | 8K contexto, LLM completo |

Veja [AI_sidecar/.env.example](../AI_sidecar/.env.example) para todas as configurações disponíveis.

## Comandos

| Comando | Função |
|----------|--------|
| `./start.sh all` | Inicia sidecar + todos os bots + console |
| `./start.sh sidecar` | Inicia apenas o sidecar |
| `./start.sh bot <nome>` | Inicia um bot pelo nome do perfil |
| `./start.sh stop` | Para todos os processos |
| `./start.sh status` | Mostra status dos bots |
| `./start.sh tail` | Reconecta ao console |

## Links

- OpenKore original: [github.com/OpenKore/openkore](https://github.com/OpenKore/openkore)
- Documentação OpenKore: [openkore.com](https://openkore.com)
- DeepSeek API: [platform.deepseek.com](https://platform.deepseek.com)
- Discord: [discord.gg/zHCKr3rbM](https://discord.gg/zHCKr3rbM)

---

## Licença

GNU General Public License v2.0 — mesma do OpenKore original.
