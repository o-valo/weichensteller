#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kleinigkeiten am Weichensteller: "~" im Pfad aufloesen, .env.example ergaenzen."""

from __future__ import annotations

import sys
from pathlib import Path

ORDNER = Path(__file__).resolve().parent.parent
MODUL = ORDNER / "weichensteller.py"
VORLAGE = ORDNER / ".env.example"

ALT = '    e["llm_env"] = hole("LLM_BAHNHOF_ENV", STILL["llm_env"])'
NEU = '    # "~" aufloesen - sonst findet os.path.exists die Datei nicht.\n' \
      '    e["llm_env"] = os.path.expanduser(hole("LLM_BAHNHOF_ENV", STILL["llm_env"]))'

ZUSATZ = '''
# ============================================================================
#  Verbindung zum LLM-Bahnhof (das Neue gegenueber dem Ollama-Bahnhof)
# ============================================================================
#  Nach aussen ist dieser Weichensteller ein vollstaendiger Ollama-Server. Innen holt
#  er sich die Gleise aus ZWEI Quellen, in fester Reihenfolge:
#    1. die Routenliste des LLM-Bahnhofs (LLM_BAHNHOF_ENV)
#    2. die eigenen ROUTE_xx dieser Datei
#
#  Die beiden werden zu EINER Kette verbunden - die Reihenfolge entscheidet,
#  welches Gleis zuerst probiert wird.
#
#  Konfigurationsdatei des LLM-Bahnhofs. "~" wird aufgeloest. Fehlt die Datei,
#  gelten nur die eigenen Gleise (kein Fehler, nur ein Hinweis im Protokoll).
LLM_BAHNHOF_ENV=~/llm-bahnhof/.env

#  Feste Reihenfolge der Quellen:
#    llm-zuerst     erst LLM-Bahnhof, dann eigene   (Vorgabe)
#    eigene-zuerst  erst eigene, dann LLM-Bahnhof
#    nur-llm        nur die Gleise des LLM-Bahnhofs
#    nur-eigene     nur die eigenen Gleise
GLEIS_REIHENFOLGE=llm-zuerst

#  1 = erfolgreiches Gleis merken (Sticky-Fallback wie im LLM-Bahnhof): die
#      naechste Anfrage beginnt dort und laeuft im Kreis weiter. Das verhindert,
#      dass jede Anfrage wieder auf ein eben ausgefallenes Gleis springt.
#  0 = jede Anfrage beginnt wieder beim ersten Gleis (wie im Ollama-Bahnhof).
STICKY_FALLBACK=1

#  Hinweis: aus ROUTE_01 der fremden Datei wird llm-ROUTE_01 - die eigenen
#  Namen bleiben unveraendert, damit sich beide Quellen nie ueberschreiben.
# ============================================================================
'''


def main() -> int:
    text = MODUL.read_text(encoding="utf-8")
    if NEU not in text:
        if ALT not in text:
            print("  ✗ Ankerstelle fuer den Pfad nicht gefunden")
            return 1
        MODUL.write_text(text.replace(ALT, NEU, 1), encoding="utf-8")
        print("  ✓ weichensteller.py: \"~\" wird jetzt aufgeloest")

    vorlage = VORLAGE.read_text(encoding="utf-8")
    if "LLM_BAHNHOF_ENV" not in vorlage:
        VORLAGE.write_text(vorlage.rstrip("\n") + "\n" + ZUSATZ, encoding="utf-8")
        print("  ✓ .env.example: Verbindungs-Sektion angehaengt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
