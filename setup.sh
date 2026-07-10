#!/usr/bin/env bash
# ============================================================================
# openkore AI — First-run setup for fresh machines
# Installs all dependencies: Python, Perl, OpenKore, Qwen model,
# Cloakbrowser, SearXNG, and all Python packages.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC} $*"; }

info "=== openkore AI — Setup ===\n"

# ── 1. System dependencies ────────────────────────────────────────────────
info "Checking system dependencies..."
for cmd in git curl wget python3 perl make gcc; do
    if ! command -v $cmd &>/dev/null; then
        err "$cmd not found. Run: sudo apt install -y git curl wget python3 perl make gcc"
        exit 1
    fi
done
info "  System deps: OK"

# ── 2. Python venv + deps ─────────────────────────────────────────────────
info "Setting up Python virtual environment..."
cd AI_sidecar
if [ ! -d venv ]; then
    python3 -m venv venv
    info "  Virtual env created"
fi
source venv/bin/activate
pip install --upgrade pip -q
pip install -e . -q
info "  Python deps: OK"
cd ..

# ── 3. Environment config ──────────────────────────────────────────────────
info "Setting up environment..."
if [ ! -f AI_sidecar/.env ]; then
    cp AI_sidecar/.env.example AI_sidecar/.env
    warn "  Edit AI_sidecar/.env with your settings"
fi
if [ ! -f .env ]; then
    echo "# Bot credentials" > .env
    warn "  Add bot passwords to .env (see README)"
fi
info "  Environment: OK"

# ── 4. Qwen model download ─────────────────────────────────────────────────
MODEL_DIR="${MODEL_DIR:-/home/lot399/models}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/bartowski/Qwen3.6-27B-UD-Q4_K_XL-GGUF/resolve/main/qwen3.6-27b-ud-q4_k_xl.gguf}"
MODEL_FILE="$MODEL_DIR/Qwen3.6-27B-UD-Q4_K_XL.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    info "Downloading Qwen3.6-27B model (~17GB)..."
    mkdir -p "$MODEL_DIR"
    wget -O "$MODEL_FILE" "$MODEL_URL"
    info "  Model downloaded"
else
    info "  Model exists: $MODEL_FILE"
fi

# ── 5. Cloakbrowser ────────────────────────────────────────────────────────
if [ ! -d "$SCRIPT_DIR/cloakbrowser" ]; then
    info "Installing Cloakbrowser..."
    git clone https://github.com/nousresearch/cloakbrowser.git
    cd cloakbrowser && npm install && cd ..
    info "  Cloakbrowser: OK"
else
    info "  Cloakbrowser: OK"
fi

# ── 6. SearXNG ─────────────────────────────────────────────────────────────
if ! command -v searxng &>/dev/null && [ ! -d "/usr/local/searxng" ]; then
    info "Installing SearXNG..."
    # Install via pip
    pip install searxng 2>/dev/null || pip install searx 2>/dev/null || true
    info "  SearXNG: Check docs if install fails (may need Docker)"
else
    info "  SearXNG: OK"
fi

# ── 7. Llama-server (llama.cpp) ────────────────────────────────────────────
if ! command -v llama-server &>/dev/null; then
    info "Installing llama.cpp..."
    if command -v nvidia-smi &>/dev/null; then
        CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir 2>/dev/null || \
        warn "  llama-cpp-python install failed. Build from source: https://github.com/ggerganov/llama.cpp"
    else
        pip install llama-cpp-python 2>/dev/null || \
        warn "  llama-cpp-python install failed (CPU mode)"
    fi
else
    info "  llama-server: OK"
fi

# ── 8. Start services ──────────────────────────────────────────────────────
info "\n=== Starting services ==="

# Kill any existing
pkill -f "llama-server.*8012" 2>/dev/null || true
pkill -f "cdp-server" 2>/dev/null || true

# Start cloakbrowser
if [ -f "$SCRIPT_DIR/cloakbrowser/cdp-server.mjs" ]; then
    cd "$SCRIPT_DIR/cloakbrowser"
    nohup node cdp-server.mjs > /dev/null 2>&1 &
    cd "$SCRIPT_DIR"
    info "  Cloakbrowser: started"
fi

# Start SearXNG (if docker available)
if command -v docker &>/dev/null; then
    docker run -d --name searxng \
        --restart unless-stopped \
        -p 127.0.0.1:8080:8080 \
        -e SEARXNG_BASE_URL=http://localhost:8080 \
        searxng/searxng 2>/dev/null && \
        info "  SearXNG: started via Docker" || true
fi

# ── 9. Start sidecar ───────────────────────────────────────────────────────
info "\nStarting sidecar..."
cd AI_sidecar
source venv/bin/activate
nohup python -m ai_sidecar.app >> ../logs/sidecar.log 2>&1 &
cd ..
info "  Sidecar: starting on port 18081"

# ── 10. Done ───────────────────────────────────────────────────────────────
info "\n=== Setup complete ==="
info "Next steps:"
info "  1. Edit .env with bot passwords"
info "  2. Start Qwen model:"
info "     CUDA_VISIBLE_DEVICES=1 llama-server --model $MODEL_FILE --host 127.0.0.1 --port 8012 --n-gpu-layers 99 &"
info "  3. Start bots:"
info "     perl -I src openkore.pl --control=.bot_profiles/<name>/control"
info ""
echo -e "${YELLOW}Run: bash setup.sh${NC}"
