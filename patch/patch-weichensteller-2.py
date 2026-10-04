#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch 2: Gleis-Zeilen robust lesen + Pruefungen isolieren."""

from __future__ import annotations

import sys
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
MODUL = WURZEL / "weichensteller.py"
TESTS = WURZEL / "tests" / "test_weichensteller.py"

MODUL_PATCHES = [
    # Zeitangabe-Erkennung vor _routen_lesen einfuegen
    (
        'def _routen_lesen(konf):\n'
        '    """ROUTE_01=URL|Schluessel|Modell|Timeout - Reihenfolge = Gleisreihenfolge."""',
        'def _ist_zeitangabe(text):\n'
        '    """Sieht der Wert nach einer Dauer aus ("45s", "10m", "1h")?\n\n'
        '    Noetig, weil fremde Konfigurationen (z. B. die des llm-bahnhofs) das\n'
        '    Modellfeld gern weglassen: URL|Schluessel|Timeout. Ohne diese Pruefung\n'
        '    waere "45s" der Modellname. Eine Dauer ohne Einheit zaehlt bewusst\n'
        '    NICHT als Zeitangabe - ein Modellname soll nie still verschwinden.\n'
        '    """\n'
        '    return bool(re.fullmatch(r"\\d+(\\.\\d+)?\\s*[smh]", (text or "").strip()))\n\n\n'
        'def _routen_lesen(konf):\n'
        '    """ROUTE_01=URL|Schluessel|Modell|Timeout - Reihenfolge = Gleisreihenfolge."""',
    ),
    # Feldauswertung: Zeitangabe in Feld 3 als Timeout deuten
    (
        '        routen.append({\n'
        '            "name": f"ROUTE_{nummer:02d}",\n'
        '            "url": _normalisiere_chat_url(url),\n'
        '            "embed_url": _normalisiere_embed_url(url),\n'
        '            "key": teile[1] if len(teile) > 1 else "",\n'
        '            "modell": teile[2] if len(teile) > 2 else "",\n'
        '            "timeout": _sekunden(teile[3] if len(teile) > 3 else "", 600.0),\n'
        '        })',
        '        modell = teile[2] if len(teile) > 2 else ""\n'
        '        zeit = teile[3] if len(teile) > 3 else ""\n'
        '        if _ist_zeitangabe(modell) and not zeit:\n'
        '            zeit, modell = modell, ""\n'
        '        routen.append({\n'
        '            "name": f"ROUTE_{nummer:02d}",\n'
        '            "url": _normalisiere_chat_url(url),\n'
        '            "embed_url": _normalisiere_embed_url(url),\n'
        '            "key": teile[1] if len(teile) > 1 else "",\n'
        '            "modell": modell,\n'
        '            "timeout": _sekunden(zeit, 600.0),\n'
        '        })',
    ),
]

TEST_PATCHES = [
    (
        'os.environ["WEICHENSTELLER_LOG"] = os.path.join(TESTORDNER, "weichensteller.log")\n',
        'os.environ["WEICHENSTELLER_LOG"] = os.path.join(TESTORDNER, "weichensteller.log")\n'
        '# Die Gleisliste des LLM-Bahnhofs NICHT mitlesen: die Pruefungen sollen\n'
        '# nicht davon abhaengen, was gerade in ~/llm-bahnhof/.env steht.\n'
        'os.environ["LLM_BAHNHOF_ENV"] = os.path.join(TESTORDNER, "kein-llm-bahnhof.env")\n',
    ),
]


def anwenden(datei: Path, patches: list[tuple[str, str]], name: str) -> int:
    text = datei.read_text(encoding="utf-8")
    fehlend = [alt for alt, _ in patches if alt not in text]
    if fehlend:
        for a in fehlend:
            print(f"  ✗ {name}: Anker nicht gefunden: {a.splitlines()[0][:60]}")
        return 1
    for alt, neu in patches:
        text = text.replace(alt, neu, 1)
    datei.write_text(text, encoding="utf-8")
    print(f"  ✓ {name}: {len(patches)} Stelle(n) gepatcht")
    return 0


def main() -> int:
    if anwenden(MODUL, MODUL_PATCHES, "weichensteller.py"):
        return 1
    return anwenden(TESTS, TEST_PATCHES, "tests/")


if __name__ == "__main__":
    sys.exit(main())
