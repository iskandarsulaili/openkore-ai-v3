<p align="center">
  <img alt="openkore AI" src="https://upload.wikimedia.org/wikipedia/commons/b/b5/Kore_2g_logo.png" width="200">
</p>

# openkore **AI**

> LLMによる意思決定を搭載したRagnarok Onlineボット — 単なるマクロではありません。**AI**、*bypass*ではありません。

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord)](https://discord.gg/zHCKr3rbM)
[![Sponsor](https://img.shields.io/badge/Sponsor-donate-EA4AAA?logo=githubsponsors)](https://github.com/sponsors/iskandarsulaili)
[![GitHub stars](https://img.shields.io/github/stars/iskandarsulaili/openkore-ai-v3)](https://github.com/iskandarsulaili/openkore-ai-v3/stargazers)

🌐 [English](../README.md) | [Português](README-pt-BR.md) | [Tagalog](README-tl.md) | [日本語](README-ja.md) | [한국어](README-ko.md) | [ไทย](README-th.md) | [Indonesia](README-id.md) | [简体中文](README-zh-CN.md)

---

## これは何？

従来のPerlクライアントに加えて、**AI意思決定エンジン**（FastAPI + Python）を追加した改造版OpenKoreです。ボットはハードコードされたマクロではなく、リアルタイムのゲーム状態とオプションのLLM呼び出しを使用して意思決定を行います。

**オリジナルのOpenKoreとの比較:**

| オリジナルOpenKore | openkore **AI** |
|-----------------|----------------|
| マクロベースの判断 | 反射→ヒューリスティック→LLM（3段階） |
| 固定設定 | 結果から自己適応 |
| 単一ボット | マルチボット群の協調 |
| コスト管理なし | ボット別予算、段階的レベル |
| コミュニティマクロのみ | ボット間共有学習 |
| 条件に反応 | PDCA計画で先読み |

---

## 機能

- **3段階エンジン** — 17の反射ルール → ヒューリスティックスコアリング → LLM（DeepSeek、OpenAI、またはローカルOllama）
- **マルチボット艦隊** — 自動ロール割り当て（タンク/ヒーラー/DPS/クラフター）
- **自己学習** — アクション・マップ・モンスターごとに改善。ボット間経験共有。
- **コスト制御** — 1日あたりのトークン予算、時間あたりの呼び出し制限、またはLLM無効化。
- **ROメカニクス** — パーティー、PVP、GVG、MVP、精錬、カード、クエスト、クラフト、転職
- **LLM搭載NPC会話** — 一般的なNPCはヒューリスティック、クエスト/精錬NPCはLLM
- **ライブコンソール** — `./start.sh` ですべてのボット + サイドカーのログを色分け表示
- **FastAPIサーバー** — 19ルーターに **112+エンドポイント**、認証ミドルウェア、フリート調整
- **CrewAI** — 18の専門エージェントをオーケストレーターが管理
- **ML潜在学習** — セルフホスト型行動学習
- **互換性** — 標準の `control/config.txt` 形式を使用

---

## 必要条件

- **Python 3.11+** — AIサイドカー用
- **Perl 5** — OpenKoreクライアント用（バンドル済み）
- **Ragnarok Onlineアカウント** — OpenKore互換サーバー
- **LLMプロバイダー**（オプション） — DeepSeek APIキー、OpenAI APIキー、またはローカルOllama

---

## クイックスタート（単一ボット）

`.env`ファイルに認証情報を設定（リポジトリルート）:

```bash
# 形式: BOT_<名前>_PASS=<パスワード>
cat >> .env << 'EOF'
BOT_あなたのボット_PASS=あなたのパスワード
EOF
```

```bash
# 1. Python環境のセットアップ
cd AI_sidecar
python3 -m venv venv
source venv/bin/activate
pip install -e .
cd ..

# 2. ボットプロファイルの作成
mkdir -p .bot_profiles/mybot/control
cp control/config.txt .bot_profiles/mybot/control/
# .bot_profiles/mybot/control/config.txt を編集

# 3. サイドカー + ボットの起動
./start.sh sidecar &
./start.sh bot mybot &
```

または一度にすべて起動:

```bash
./start.sh all
```

## クイックスタート（複数ボット）

```bash
mkdir -p .bot_profiles/char1/control .bot_profiles/char2/control
cp control/config.txt .bot_profiles/char1/control/
cp control/config.txt .bot_profiles/char2/control/

# .envにパスワードを設定
cat >> .env << 'EOF'
BOT_char1_PASS=pass1
BOT_char2_PASS=pass2
EOF

# すべて起動:
./start.sh all
```

## 料金プラン

環境変数 `OPENKORE_AI_LLM_COST_TIER` で設定（`AI_sidecar/.env`）:

| 設定 | 効果 |
|------|--------|
| `OPENKORE_AI_LLM_COST_TIER=off` | 反射 + ヒューリスティックのみ。コスト0。 |
| `OPENKORE_AI_LLM_COST_TIER=economy` | 512トークンコンテキスト、最小LLM |
| `OPENKORE_AI_LLM_COST_TIER=standard` | 2Kコンテキスト、通常LLM（デフォルト） |
| `OPENKORE_AI_LLM_COST_TIER=premium` | 8Kコンテキスト、完全LLM推論 |

すべての設定は [AI_sidecar/.env.example](../AI_sidecar/.env.example) を参照。

## コマンド

| コマンド | 説明 |
|----------|------|
| `./start.sh all` | サイドカー + 全ボット + コンソールを起動 |
| `./start.sh sidecar` | サイドカーのみ起動 |
| `./start.sh bot <名前>` | 指定したボットを起動 |
| `./start.sh stop` | 全プロセスを停止 |
| `./start.sh status` | ボットの状態を表示 |
| `./start.sh tail` | コンソールに再接続 |

## リンク

- OpenKore original: [github.com/OpenKore/openkore](https://github.com/OpenKore/openkore)
- OpenKore ドキュメント: [openkore.com](https://openkore.com)
- DeepSeek API: [platform.deepseek.com](https://platform.deepseek.com)
- Discord: [discord.gg/zHCKr3rbM](https://discord.gg/zHCKr3rbM)

---

## ライセンス

GNU General Public License v2.0 — OpenKore originalと同じ。
