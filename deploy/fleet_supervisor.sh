#!/usr/bin/env bash
# Fleet supervisor: launches the openkore-ai-v3 sidecar + all bots fully detached
# (setsid + nohup) so they leave this unit's cgroup and survive session/systemd
# reaping. Then sleeps to keep the systemd unit in "active" (Type=simple) state.
set -u
SCRIPT_DIR="/home/lot399/openkore-ai-v3"
SIDECAR_DIR="$SCRIPT_DIR/AI_sidecar"
SIDECAR_LOG="$SCRIPT_DIR/logs/sidecar.log"
BOT_LOGS="$SCRIPT_DIR/logs"
SIDECAR_PORT="${OPENKORE_AI_PORT:-18081}"

# ── LLM gateway wiring (conscious tier) ──
# Point an OpenAI-compatible provider at the local LLM gateway so the LLMManager has a
# configured provider and the conscious tier (LLM gear/sustain advisory, agents) can
# actually reason. The gateway serves deepseek-v4-flash over an OpenAI-compatible API.
# Without these the LLMManager has no providers -> is_available()=False -> the conscious
# tier silently no-ops (no LLM-driven decisions).
export LLM_OPENAI_BASE_URL="${LLM_OPENAI_BASE_URL:-http://192.168.0.100:20128/v1}"
export LLM_OPENAI_API_KEY="${LLM_OPENAI_API_KEY:-local}"
export LLM_OPENAI_MODEL="${LLM_OPENAI_MODEL:-deepseek-v4-flash}"

# --- Sidecar (fleet manager, keep-alive) ---
if ! curl -sf "http://127.0.0.1:${SIDECAR_PORT}/health/live" > /dev/null 2>&1; then
    cd "$SIDECAR_DIR" || exit 1
    setsid nohup "$SIDECAR_DIR/venv/bin/python" -m ai_sidecar.app --keep-alive --keep-alive-poll 10 \
        > "$SIDECAR_LOG" 2>&1 < /dev/null &
    cd "$SCRIPT_DIR" || exit 1
fi

# --- Bots (discover from .bot_profiles) ---
for profile_dir in "$SCRIPT_DIR"/.bot_profiles/*/; do
    [ -d "$profile_dir" ] || continue
    name="$(basename "$profile_dir")"
    # Skip if already running
    if pgrep -f "openkore\.pl.*\.bot_profiles/$name/" > /dev/null 2>&1; then
        continue
    fi
    cd "$SCRIPT_DIR" || exit 1
    setsid nohup perl -I src openkore.pl --plugins=plugins --control=".bot_profiles/$name/control" \
        < /dev/null > "$BOT_LOGS/$name.log" 2>&1 &
    sleep 12   # stagger to avoid char-server SEGV on simultaneous char-select
done

cd "$SCRIPT_DIR" || exit 1
# Keep the unit active (Type=simple). On each tick, relaunch the sidecar if it
# died AND relaunch any dead bot, so the whole fleet self-heals without a restart.
# Tick is 15s (was 60s) to shrink the window where the sidecar is down and the
# bots run disconnected — 60s was slow enough to leave a noticeable gap.
while true; do
    sleep 15
    # Sidecar self-heal
    if ! curl -sf "http://127.0.0.1:${SIDECAR_PORT}/health/live" > /dev/null 2>&1 \
       && ! pgrep -f "ai_sidecar\.app" > /dev/null 2>&1; then
        cd "$SIDECAR_DIR"
        setsid nohup "$SIDECAR_DIR/venv/bin/python" -m ai_sidecar.app --keep-alive --keep-alive-poll 10 \
            > "$SIDECAR_LOG" 2>&1 < /dev/null &
        cd "$SCRIPT_DIR"
    fi
    # Bot self-heal
    for profile_dir in "$SCRIPT_DIR"/.bot_profiles/*/; do
        [ -d "$profile_dir" ] || continue
        name="$(basename "$profile_dir")"
        if ! pgrep -f "openkore\.pl.*\.bot_profiles/$name/" > /dev/null 2>&1; then
            cd "$SCRIPT_DIR"
            setsid nohup perl -I src openkore.pl --plugins=plugins --control=".bot_profiles/$name/control" \
                < /dev/null > "$BOT_LOGS/$name.log" 2>&1 &
            sleep 12
        fi
    done
done
