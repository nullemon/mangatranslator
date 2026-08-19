#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# MangaTranslator — start the app on a Mac
#
# Double-click this file in Finder. It starts the server and opens your
# browser. Leave the Terminal window open while you work; close it (or press
# Control-C) to stop.
# ═══════════════════════════════════════════════════════════════════════════
set -u
cd "$(dirname "$0")" || exit 1

if [ ! -d venv ]; then
    echo "[x] Not set up yet — run setup_mac.command first."
    echo ""
    read -r -p "Press Return to close..." _
    exit 1
fi

# shellcheck disable=SC1091
. venv/bin/activate

PORT="${PORT:-8000}"
echo "Starting MangaTranslator on http://localhost:$PORT"
echo "(leave this window open; press Control-C to stop)"
echo ""

# Open the browser once the server is actually accepting connections.
(
  for _ in $(seq 1 60); do
      sleep 1
      if curl -s -o /dev/null "http://localhost:$PORT"; then
          open "http://localhost:$PORT"
          break
      fi
  done
) &

PORT="$PORT" python app.py

echo ""
echo "Server stopped."
read -r -p "Press Return to close..." _
