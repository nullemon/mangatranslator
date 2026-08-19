#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# MangaTranslator — one-step Mac setup
#
# Double-click this file in Finder, or run it in Terminal:
#     chmod +x setup_mac.command && ./setup_mac.command
#
# It creates a private virtual environment inside this folder, installs
# everything (base app + the accuracy models), downloads the model weights,
# and finishes with a report. Safe to re-run — it only installs what's
# missing.
#
# Options:
#     ./setup_mac.command --basic       skip the AI model stack (small + fast,
#                                       uses the built-in CV fallbacks)
#     ./setup_mac.command --mangajanai  also fetch the manga HD upscaler
#
# When it's done, start the app by double-clicking  start_mac.command
# ═══════════════════════════════════════════════════════════════════════════
set -u

# .command files launch with the home folder as the working directory —
# always work relative to where this script actually lives.
cd "$(dirname "$0")" || exit 1

BASIC=0
WANT_MANGAJANAI=0
for arg in "$@"; do
    [ "$arg" = "--basic" ] && BASIC=1
    [ "$arg" = "--mangajanai" ] && WANT_MANGAJANAI=1
done

echo "═══════════════════════════════════════════════════════"
echo "  MangaTranslator — Mac setup"
echo "═══════════════════════════════════════════════════════"
echo "  folder: $(pwd)"
echo "  mac   : $(sw_vers -productVersion 2>/dev/null || echo '?') on $(uname -m)"
echo ""

if [ "$(uname -m)" = "arm64" ]; then
    echo "[i] Apple Silicon detected — the models will use the Mac's GPU (Metal)."
else
    echo "[i] Intel Mac — everything works, running on CPU (slower)."
fi

# ── 1. Xcode command line tools (needed to build a few wheels) ───────────
if ! xcode-select -p >/dev/null 2>&1; then
    echo ""
    echo "==> Command Line Tools are required. A system dialog will open —"
    echo "    click Install, wait for it to finish, then run this again."
    xcode-select --install 2>/dev/null
    exit 1
fi

# ── 2. Find a usable Python (3.10 – 3.12) ────────────────────────────────
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        v="$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
        case "$v" in
            3.10|3.11|3.12) PY="$cand"; break;;
        esac
    fi
done

if [ -z "$PY" ]; then
    echo ""
    echo "[x] No suitable Python found (need 3.10, 3.11 or 3.12)."
    echo "    macOS ships an older one, and 3.13 isn't supported by the AI"
    echo "    libraries yet. Install 3.12 either way:"
    echo ""
    echo "      • Download the 3.12 installer from  python.org/downloads/macos"
    echo "      • or with Homebrew:   brew install python@3.12"
    echo ""
    echo "    Then run this script again."
    exit 1
fi
echo "[✓] Using $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

# ── 3. Private virtual environment (keeps your system Python untouched) ──
if [ ! -d venv ]; then
    echo ""
    echo "==> Creating the virtual environment..."
    "$PY" -m venv venv || { echo "[x] Could not create venv"; exit 1; }
fi
# shellcheck disable=SC1091
. venv/bin/activate
python -m pip install --upgrade pip >/dev/null 2>&1
echo "[✓] Virtual environment ready"

# ── 4. Base app dependencies ─────────────────────────────────────────────
echo ""
echo "==> [1/3] Base dependencies (web app, image handling)..."
pip install -r requirements.txt || { echo "[x] Base install failed"; exit 1; }

# ── 5. The AI model stack ────────────────────────────────────────────────
if [ "$BASIC" = "1" ]; then
    echo ""
    echo "==> [2/3] Skipping the model stack (--basic)."
    echo "    Detection and erasure will use the built-in CV fallbacks."
else
    echo ""
    echo "==> [2/3] AI models (torch+Metal, balloon model, OCR, LaMa, upscaler)..."
    echo "    First run downloads roughly 2-3 GB. Grab a coffee."
    pip install -r requirements-mac.txt || \
        echo "[!] Some model packages failed — the app still runs on fallbacks."
    # CRAFT pins an ancient opencv in its setup; install it without deps.
    pip install --no-deps craft-text-detector >/dev/null 2>&1 || \
        echo "[!] CRAFT unavailable — the built-in free-text finder will be used."

    echo ""
    echo "==> [3/3] Downloading model weights..."
    python setup_models.py || \
        echo "[!] Some weights failed — they retry automatically on first use."

    if [ "$WANT_MANGAJANAI" = "1" ]; then
        echo ""
        echo "==> Manga HD upscaler (MangaJaNai)..."
        mkdir -p models/mangajanai
        curl -fL --progress-bar -o models/mangajanai/2x_MangaJaNai_1500p_V1_ESRGAN_90k.pth \
          "https://huggingface.co/Kim2091/MangaJaNai/resolve/main/2x_MangaJaNai_1500p_V1_ESRGAN_90k.pth" \
          || echo "    (download failed — the standard upscaler still works)"
    fi
fi

# ── 6. Report ────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
python check_setup.py 2>/dev/null || true
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Setup finished."
echo ""
echo "  To start:  double-click  start_mac.command"
echo "             (or run:  source venv/bin/activate && python app.py)"
echo ""
echo "  Then open  http://localhost:8000  and paste your API key"
echo "  into Settings."
echo ""
