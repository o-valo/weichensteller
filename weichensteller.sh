#!/usr/bin/env bash
# Weichensteller starten (Ollama nach aussen, Gleise aus zwei Quellen innen).
#
#   ./weichensteller.sh                  startet mit der .env im Ordner
#   ./weichensteller.sh --pruefen        zeigt die Konfiguration und beide Gleis-Gruppen
#   ./weichensteller.sh --env /pfad/.env mit anderer Konfiguration
set -u

HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HIER" || exit 1

if [ ! -x ./venv/bin/python3 ]; then
    echo "Kein venv gefunden. Einmalig:"
    echo "  cd $HIER && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

exec ./venv/bin/python3 weichensteller.py "$@"
