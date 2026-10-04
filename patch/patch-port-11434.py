#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weichensteller auf den Ollama-Port 11434 umstellen.

Der Ollama-Bahnhof (der bis 21.09.2026 auf 11434 lauschte) ist gestoppt, der
Weichensteller nimmt den Platz ein. Jede Ersetzung wird vorher geprueft.
"""
import pathlib
import sys

Z = pathlib.Path(__file__).resolve().parent.parent

AENDERUNGEN = [
    (
        Z / ".env",
        [
            (
                "# 11435, weil der Ollama-Bahnhof bereits auf 11434 lauscht (Ollama-Port).\n"
                "# Frei ist 11434, sobald der Ollama-Bahnhof gestoppt ist - dann hier umstellen.\n"
                "WEICHENSTELLER_PORT=11435",
                "# 11434 = der echte Ollama-Port. Der Ollama-Bahnhof, der bis 21.09. hier\n"
                "# lauschte, ist gestoppt - der Weichensteller nimmt seinen Platz ein und\n"
                "# geht damit ohne Umstellung bei Clients als Ollama durch.\n"
                "WEICHENSTELLER_PORT=11434",
            )
        ],
    ),
    (
        Z / "README.md",
        [
            (
                "./venv/bin/python3 weichensteller.py             # startet (Vorgabe-Port 11435)",
                "./venv/bin/python3 weichensteller.py             # startet auf dem Ollama-Port 11434",
            ),
            (
                "**Port:** Vorgabe **11435**, denn der Ollama-Bahnhof lauscht bereits auf dem\n"
                "Ollama-Port 11434. Ist 11434 frei, in der `.env` auf 11434 umstellen – dann ist\n"
                "der Weichensteller ohne Umstellung anstelle eines echten Ollama einsetzbar.",
                "**Port:** Vorgabe **11434** – der echte Ollama-Port. Der Weichensteller tritt damit\n"
                "ohne Umstellung an die Stelle eines echten Ollama; ein Client, der auf\n"
                "`http://127.0.0.1:11434` zeigt, merkt den Unterschied nicht.\n"
                "\n"
                "> Nur **einer** darf auf 11434 lauschen. Der Ollama-Bahnhof wurde dafür am\n"
                "> 21.09.2026 gestoppt (`kill -TERM <PID>`); er lauscht sonst auf denselben Port.\n"
                "> Umgekehrt: soll der Ollama-Bahnhof wieder laufen, hier in der `.env` auf\n"
                "> 11435 zurückstellen.",
            ),
            ("export OLLAMA_HOST=http://127.0.0.1:11435", "export OLLAMA_HOST=http://127.0.0.1:11434"),
            ("http://127.0.0.1:11435/v1        # Schlüssel egal", "http://127.0.0.1:11434/v1        # Schlüssel egal"),
            (
                "curl -s http://127.0.0.1:11435/health | python3 -m json.tool",
                "curl -s http://127.0.0.1:11434/health | python3 -m json.tool",
            ),
        ],
    ),
]


def main() -> int:
    fehler = 0
    for pfad, ersetzungen in AENDERUNGEN:
        if not pfad.exists():
            print(f"FEHLT: {pfad}")
            fehler += 1
            continue
        text = pfad.read_text(encoding="utf-8")
        for alt, neu in ersetzungen:
            n = text.count(alt)
            if n != 1:
                print(f"ANKER {n}x (erwartet 1x) in {pfad.name}: {alt.splitlines()[0][:70]!r}")
                fehler += 1
                continue
            text = text.replace(alt, neu)
            print(f"ok {pfad.name}: {alt.splitlines()[0][:62]!r}")
        if fehler:
            print(f"-> {pfad.name} NICHT geschrieben (erst Anker klaeren)")
            continue
        pfad.write_text(text, encoding="utf-8")
    if fehler:
        print(f"\nABBRUCH: {fehler} Anker passen nicht.")
        return 1
    print("\nAlle Aenderungen geschrieben.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
