#!/usr/bin/env bash
# ============================================================================
# openkore-ai-v3 — First-Run Setup Wizard
# ============================================================================
# Guides a fresh machine from zero to running:
#   1. System packages (git, curl, build tools, Perl modules)
#   2. Python virtual environment + sidecar dependencies
#   3. Environment configuration
#   4. Cloakbrowser (optional, for anti-detection)
#   5. Verification
#
# Usage:
#   bash setup.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC} $*"; }
header() { echo -e "\n${BOLD}════════════════════════════════════════════════════════════${NC}"; }
step()  { echo -e "\n${CYAN}${BOLD}▶ $1${NC}"; }

# ────────────────────────────────────────────────────────────────────────────
# Pre-flight: root check
# ────────────────────────────────────────────────────────────────────────────
if [ "$(id -u)" -eq 0 ]; then
    err "Do not run as root. Run as your normal user (sudo will be prompted where needed)."
    exit 1
fi

header
echo -e "  ${BOLD}openkore AI v3 — Setup Wizard${NC}"
echo ""
echo "  This will install all dependencies needed to run"
echo "  the openkore multi-bot system with AI sidecar."
echo ""
header

# ────────────────────────────────────────────────────────────────────────────
# 1. System dependencies
# ────────────────────────────────────────────────────────────────────────────
step "1/6 — System packages"

SYSTEM_DEPS=(git curl wget python3 python3-venv python3-pip perl make gcc sqlite3 libsqlite3-dev)
PERL_DEPS=(libcarp-assert-perl libjson-pp-perl)

info "Updating package lists..."
sudo apt update -qq || warn "apt update failed, continuing..."

info "Installing system packages..."
sudo apt install -y -qq "${SYSTEM_DEPS[@]}" "${PERL_DEPS[@]}" || {
    err "Failed to install system packages."
    err "Try manually: sudo apt install -y ${SYSTEM_DEPS[*]} ${PERL_DEPS[*]}"
    exit 1
}

for cmd in git curl wget python3 perl make gcc; do
    if ! command -v $cmd &>/dev/null; then
        err "$cmd still not found after install — check your package manager"
        exit 1
    fi
done

# Verify Perl modules
for mod in Carp::Assert JSON::PP; do
    if perl -e "use $mod; print qq{OK\n}" 2>/dev/null; then
        info "  Perl module $mod: OK"
    else
        warn "  Perl module $mod: missing — install with: sudo cpan $mod"
    fi
done

info "System packages: done."

# ────────────────────────────────────────────────────────────────────────────
# 2. NVIDIA / CUDA check (informational)
# ────────────────────────────────────────────────────────────────────────────
step "2/6 — GPU detection"

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    info "NVIDIA GPU detected: $GPU_INFO"
else
    warn "nvidia-smi not found — running in CPU-only mode."
    warn "For GPU acceleration, install NVIDIA drivers + CUDA toolkit:"
    warn "  https://developer.nvidia.com/cuda-downloads"
fi

# ────────────────────────────────────────────────────────────────────────────
# 3. Python virtual environment
# ────────────────────────────────────────────────────────────────────────────
step "3/6 — Python virtual environment"

cd "$SCRIPT_DIR/AI_sidecar"

if [ ! -d venv ]; then
    info "Creating Python virtual environment..."
    python3 -m venv venv
    info "  Virtual env created"
else
    info "  Virtual env exists"
fi

info "Installing Python packages..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -e . -q
deactivate
info "Python dependencies: done."

cd "$SCRIPT_DIR"

# ────────────────────────────────────────────────────────────────────────────
# 4. Environment configuration
# ────────────────────────────────────────────────────────────────────────────
step "4/6 — Environment configuration"

if [ ! -f AI_sidecar/.env ]; then
    cp AI_sidecar/.env.example AI_sidecar/.env
    info "Created AI_sidecar/.env from example"
    warn "  → Edit AI_sidecar/.env with your API keys and settings"
else
    info "  AI_sidecar/.env exists"
fi

if [ ! -f .env ]; then
    cat > .env << 'EOF'
# ============================================================================
# openkore AI v3 — Bot Credentials
# ============================================================================
# Add passwords for each bot profile.
# Naming convention: BOT_<PROFILE_NAME_UPPERCASE>_PASS
#
# Examples:
#   BOT_MYCHAR_PASS=mysecret
#   BOT_ALTCHAR_PASS=othersecret
# ============================================================================
EOF
    info "Created root .env (add your bot passwords here)"
    info "  → Add entries like: BOT_<NAME>_PASS=your_password"
else
    info "  Root .env exists"
fi

# ────────────────────────────────────────────────────────────────────────────
# 5. Cloakbrowser (optional, for anti-detection)
# ────────────────────────────────────────────────────────────────────────────
step "5/6 — Cloakbrowser (optional)"

if command -v node &>/dev/null; then
    if [ ! -d "$SCRIPT_DIR/cloakbrowser" ]; then
        info "Installing Cloakbrowser..."
        git clone https://github.com/nousresearch/cloakbrowser.git "$SCRIPT_DIR/cloakbrowser" 2>/dev/null || {
            warn "  git clone failed — skip or install manually"
        }
        if [ -d "$SCRIPT_DIR/cloakbrowser" ]; then
            cd "$SCRIPT_DIR/cloakbrowser" && npm install && cd "$SCRIPT_DIR"
            info "  Cloakbrowser: installed"
        fi
    else
        info "  Cloakbrowser: already present"
    fi
else
    warn "  Node.js not found — skip Cloakbrowser (anti-detection won't work)"
    warn "  Install Node.js: sudo apt install nodejs npm"
fi

# ────────────────────────────────────────────────────────────────────────────
# 6. Verification
# ────────────────────────────────────────────────────────────────────────────
step "6/6 — Verification"

PASS=0
FAIL=0

check() {
    local desc="$1"
    shift
    if "$@" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $desc"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗${NC} $desc"
        FAIL=$((FAIL + 1))
    fi
}

check "Python 3.11+"          python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"
check "Perl 5.32+"            perl -e "exit($] >= 5.032 ? 0 : 1)"
check "Git"                   command -v git
check "Perl: Carp::Assert"    perl -e "use Carp::Assert; print qq{}" 2>/dev/null
check "Perl: JSON::PP"        perl -e "use JSON::PP; print qq{}" 2>/dev/null
check "Python venv exists"    [ -f "$SCRIPT_DIR/AI_sidecar/venv/bin/python" ]
check "Sidecar package"       "$SCRIPT_DIR/AI_sidecar/venv/bin/python" -c "import fastapi, uvicorn, httpx, pydantic, yaml" 2>/dev/null
check "Bot profiles exist"    ls "$SCRIPT_DIR/.bot_profiles/"/*/control/config.txt &>/dev/null

echo ""
if [ $FAIL -eq 0 ]; then
    info "All checks passed! ($PASS/$((PASS+FAIL)))"
else
    warn "$FAIL check(s) failed — review above"
fi

# ────────────────────────────────────────────────────────────────────────────
# Done
# ────────────────────────────────────────────────────────────────────────────
header
echo -e "  ${BOLD}Setup complete!${NC}"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Edit .env — add bot passwords:"
echo "     nano .env"
echo ""
echo "  2. Edit AI_sidecar/.env — set API keys:"
echo "     nano AI_sidecar/.env"
echo ""
if [ -d "$SCRIPT_DIR/.bot_profiles" ]; then
    first_profile=$(ls -d "$SCRIPT_DIR/.bot_profiles/"*/ 2>/dev/null | head -1)
    if [ -n "$first_profile" ]; then
        echo "  3. Start the system:"
        echo "     ./start.sh"
        echo ""
    fi
else
    echo "  3. Create at least one bot profile:"
    echo "     mkdir -p .bot_profiles/myaccount/control"
    echo "     cp -r control/* .bot_profiles/myaccount/control/"
    echo ""
    echo "  4. Then start:"
    echo "     ./start.sh"
    echo ""
fi
echo "  (Optional) Start Cloakbrowser for anti-detection:"
echo "    cd cloakbrowser && node cdp-server.mjs &"
echo ""
header
