#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""README des Weichenstellers um die Erkenntnisse aus dem Rauchtest ergaenzen."""

from __future__ import annotations

import sys
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

AUSTAUSCH: list[tuple[str, str]] = [
    # 1) Einbettungs-Falle (im Rauchtest aufgefallen)
    (
        "- `pull`/`push` laden und senden nichts – sie verwalten nur den eigenen Katalog.",
        "- **Einbettungen brauchen ein Gleis, das sie kann.** Die Gleisliste des\n"
        "  LLM-Bahnhofs liefert **keine** Einbettungen (der LLM-Bahnhof kennt\n"
        "  `/v1/embeddings` nicht – im Rauchtest mit 404 belegt). Im Modus\n"
        "  `durchreichen` scheitert `/api/embed` dann mit **503**. Mit `MODUS=auto`\n"
        "  (Vorgabe) oder `SIM_FALLBACK=1` antwortet stattdessen die Simulation –\n"
        "  deterministisch, aber bedeutungslos. Wer echte Einbettungen braucht,\n"
        "  trägt ein Gleis mit `/embeddings` ein (z. B. ein echtes Ollama).\n"
        "- `pull`/`push` laden und senden nichts – sie verwalten nur den eigenen Katalog.",
    ),
    # 2) Rauchtest nennen
    (
        "Alles läuft **ohne Netz** und ohne ein echtes Modell.",
        "Alles läuft **ohne Netz** und ohne ein echtes Modell.\n\n"
        "Dazu der **Rauchtest durch die ganze Kette** – startet den Weichensteller und\n"
        "einen Stub-Anbieter, lässt das erste Gleis ausfallen und prüft die\n"
        "Ollama-Oberfläche, die feste Gleis-Reihenfolge, den Fallback, das\n"
        "Sticky-Verhalten, `/v1/completions`, Einbettungen und den NDJSON-Strom\n"
        "(19 Prüfungen):\n\n"
        "```bash\n./venv/bin/python3 rauchtest.py\n```",
    ),
    # 3) Patch-Ordner erklaeren (gegen das Auseinanderlaufen)
    (
        "Verbesserungen an der Ollama-Oberfläche gehören in `~/ollama-bahnhof` und\n"
        "müssen hier nachgezogen werden.",
        "Verbesserungen an der Ollama-Oberfläche gehören in `~/ollama-bahnhof` und\n"
        "müssen hier nachgezogen werden.\n\n"
        "Damit das nachvollziehbar bleibt, liegt der Unterschied als **Patch** bei\n"
        "(`patch/`): `patch-weichensteller.py` und `patch-weichensteller-2.py` machen\n"
        "aus einer frischen Kopie von `ollama_bahnhof.py` diesen Weichensteller,\n"
        "`setup-weichensteller.py` ergänzt den `~`-Pfad und die `.env.example`.\n"
        "Zum Nachziehen: Kopie des neuen Ollama-Bahnhofs nach `weichensteller.py`,\n"
        "dann die drei Skripte der Reihe nach laufen lassen. Passt eine Ankerstelle\n"
        "nicht mehr, meldet das Skript es und ändert **nichts**.",
    ),
]


def main() -> int:
    text = README.read_text(encoding="utf-8")
    fehlend = [alt for alt, _ in AUSTAUSCH if alt not in text]
    if fehlend:
        for a in fehlend:
            print(f"  ✗ Anker nicht gefunden: {a.splitlines()[0][:60]}")
        return 1
    for alt, neu in AUSTAUSCH:
        text = text.replace(alt, neu, 1)
    README.write_text(text, encoding="utf-8")
    print(f"  ✓ README ergänzt ({len(AUSTAUSCH)} Stellen)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
