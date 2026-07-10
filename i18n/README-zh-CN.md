<p align="center">
  <img alt="openkore AI" src="https://upload.wikimedia.org/wikipedia/commons/b/b5/Kore_2g_logo.png" width="200">
</p>

# openkore **AI**

> 由LLM决策驱动的Ragnarok Online机器人 — 不仅仅是宏。**AI**，而不是*bypass*。

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
[![Sponsor](https://img.shields.io/badge/Sponsor-donate-EA4AAA?logo=githubsponsors)](https://github.com/sponsors/iskandarsulaili)
[![GitHub stars](https://img.shields.io/github/stars/iskandarsulaili/openkore-ai-v3)](https://github.com/iskandarsulaili/openkore-ai-v3/stargazers)

🌐 [English](../README.md) | [Português](README-pt-BR.md) | [Tagalog](README-tl.md) | [日本語](README-ja.md) | [한국어](README-ko.md) | [ไทย](README-th.md) | [Indonesia](README-id.md) | [简体中文](README-zh-CN.md)

---

## 这是什么？

在经典Perl客户端之外，增加了**AI决策引擎**（FastAPI + Python）的修改版OpenKore。机器人使用实时游戏状态 + 可选的LLM调用来做出决策，而不是遵循硬编码的宏。

**与原始OpenKore的比较:**

| 原始OpenKore | openkore **AI** |
|-----------------|----------------|
| 基于宏的决策 | 反射 → 启发式 → LLM（3层） |
| 固定配置行为 | 从结果中自适应 |
| 单机器人 | 多机器人蜂群协调 |
| 无成本控制 | 每机器人预算，分级 |
| 仅社区宏 | 跨机器人共享学习 |
| 对条件做出反应 | 通过PDCA循环提前规划 |

---

## 功能

- **3层引擎** — 17条反射规则 → 启发式评分 → LLM（DeepSeek、OpenAI或本地Ollama）
- **多机器人舰队** — 自动角色分配（坦克/治疗/输出/工匠）
- **自我学习** — 按行动/地图/怪物改善。跨机器人经验共享。
- **成本控制** — 设置每日令牌预算、每小时调用限制或禁用LLM。
- **RO机制** — 组队、PVP、GVG、MVP、精炼、卡片、任务、制作、转职
- **LLM驱动的NPC对话** — 通用NPC使用启发式，任务/精炼NPC使用LLM
- **实时控制台** — `./start.sh`在一个终端中彩色显示所有机器人和sidecar日志
- **FastAPI服务器** — 19个路由器上的**112+个端点**，认证中间件，舰队协调
- **CrewAI** — 18个由编排器管理的专业代理
- **ML潜意识** — 自托管行为学习
- **兼容** — 使用标准`control/config.txt`格式

---

## 要求

- **Python 3.11+** — 用于AI sidecar
- **Perl 5** — 用于OpenKore客户端（已包含）
- **Ragnarok Online账户** — 在任何兼容OpenKore的服务器上
- **LLM提供商**（可选）— DeepSeek API密钥、OpenAI API密钥或本地Ollama实例

---

## 快速开始（单机器人）

在`.env`中设置凭据（仓库根目录）:

```bash
# 格式: BOT_<名称>_PASS=<密码>
cat >> .env << 'EOF'
BOT_我的机器人_PASS=我的密码
EOF
```

```bash
# 1. 设置Python环境
cd AI_sidecar
python3 -m venv venv
source venv/bin/activate
pip install -e .
cd ..

# 2. 创建机器人配置文件
mkdir -p .bot_profiles/mybot/control
cp control/config.txt .bot_profiles/mybot/control/
# 编辑 .bot_profiles/mybot/control/config.txt

# 3. 启动sidecar + 机器人
./start.sh sidecar &
./start.sh bot mybot &
```

或一次性启动所有:

```bash
./start.sh all
```

## 快速开始（多机器人）

```bash
mkdir -p .bot_profiles/char1/control .bot_profiles/char2/control
cp control/config.txt .bot_profiles/char1/control/
cp control/config.txt .bot_profiles/char2/control/

# 在.env中设置密码
cat >> .env << 'EOF'
BOT_char1_PASS=pass1
BOT_char2_PASS=pass2
EOF

# 启动所有:
./start.sh all
```

## 成本等级

通过环境变量`OPENKORE_AI_LLM_COST_TIER`设置（在`AI_sidecar/.env`中）:

| 设置 | 效果 |
|------|------|
| `OPENKORE_AI_LLM_COST_TIER=off` | 仅反射 + 启发式。零成本。 |
| `OPENKORE_AI_LLM_COST_TIER=economy` | 512令牌上下文，最少LLM |
| `OPENKORE_AI_LLM_COST_TIER=standard` | 2K上下文，正常LLM（默认） |
| `OPENKORE_AI_LLM_COST_TIER=premium` | 8K上下文，完整LLM推理 |

查看[AI_sidecar/.env.example](../AI_sidecar/.env.example)获取所有可用设置。

## 命令

| 命令 | 作用 |
|--------|------|
| `./start.sh all` | 启动sidecar + 所有机器人 + 控制台 |
| `./start.sh sidecar` | 仅启动sidecar |
| `./start.sh bot <名称>` | 按配置文件名启动一个机器人 |
| `./start.sh stop` | 停止所有进程 |
| `./start.sh status` | 显示机器人状态 |
| `./start.sh tail` | 重新连接控制台 |

## 链接

- 原始OpenKore: [github.com/OpenKore/openkore](https://github.com/OpenKore/openkore)
- OpenKore文档: [openkore.com](https://openkore.com)
- DeepSeek API: [platform.deepseek.com](https://platform.deepseek.com)
- Discord: [discord.gg/zHCKr3rbM](https://discord.gg/zHCKr3rbM)

---

## 许可证

GNU General Public License v2.0 — 与原始OpenKore相同。
