#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch: aus dem Ollama-Bahnhof den Weichensteller machen.

Der Weichensteller = Ollama-Bahnhof (Oberflaeche unveraendert) + das, was den
LLM-Bahnhof ausmacht:

  1. zweite Gleis-Quelle: ROUTE_xx aus der Konfiguration des LLM-Bahnhofs,
     in FESTER Reihenfolge (Vorgabe: llm-zuerst) mit den eigenen Gleisen
     zu einer Kette verbunden,
  2. Sticky-Fallback: das erfolgreiche Gleis wird gemerkt, die naechste
     Anfrage startet dort und laeuft im Kreis weiter,
  3. /health und --pruefen zeigen beide Gruppen und das aktive Gleis.

Alle Ankerstellen werden vorher geprueft; passt eine nicht, wird NICHTS
geschrieben.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Wurzel des Projekts, RELATIV zu diesem Skript - nicht ueber Path.home().
# Vorher stand hier ein fester Pfad im Heimatverzeichnis des Autors; damit
# zeigte das Skript bei jedem anderen Benutzer ins Leere.
WURZEL = Path(__file__).resolve().parent.parent
DATEI = WURZEL / "weichensteller.py"

NEUE_FUNKTIONEN = '''
def _routen_lesen_aus_pfad(pfad, kennung):
    """ROUTE_xx aus einer zweiten Konfiguration lesen (Gleise des LLM-Bahnhofs).

    Gleiche Zeilenform wie hier, aber eigener Namensraum: der Name bekommt die
    Kennung als Praefix, damit sich zwei Quellen nie ueberschreiben.
    """
    if not pfad or not os.path.exists(pfad):
        return []
    routen = _routen_lesen(_env_lesen(pfad))
    for route in routen:
        route["name"] = f"{kennung}-{route['name']}"
        route["gruppe"] = kennung
    return routen


def _routen_zusammenfuehren(konf, e):
    """Eigene Gleise und Gleise des LLM-Bahnhofs in FESTER Reihenfolge.

    Vorgabe 'llm-zuerst': erst die Routenliste des LLM-Bahnhofs (dort wird die
    Free-Tier-Liste gepflegt), dann die eigenen Gleise. So liegt die Pflege an
    einer Stelle, und trotzdem greift eine eigene Route, wenn der LLM-Bahnhof
    nichts liefert.
    """
    eigene = _routen_lesen(konf)
    for route in eigene:
        route["gruppe"] = "eigene"

    pfad = e.get("llm_env") or ""
    fremde = _routen_lesen_aus_pfad(pfad, "llm")
    if pfad and not os.path.exists(pfad):
        log.info(f"Gleise des LLM-Bahnhofs nicht gefunden ({pfad}) - "
                 "es gelten nur die eigenen Gleise.")

    reihenfolge = e.get("reihenfolge", "llm-zuerst")
    if reihenfolge == "nur-llm":
        return fremde
    if reihenfolge == "nur-eigene":
        return eigene
    if reihenfolge == "eigene-zuerst":
        return eigene + fremde
    return fremde + eigene


def _gleis_reihenfolge():
    """Gleise ab dem zuletzt erfolgreichen beginnen, dann im Kreis.

    Ohne Sticky beginnt jede Anfrage wieder beim ersten Gleis (Verhalten des
    Ollama-Bahnhofs). Mit Sticky - Vorgabe - startet die naechste Anfrage dort,
    wo die letzte erfolgreich war: kein vorzeitiges Zurueckspringen zum ersten
    Gleis. Faellt das gemerkte Gleis aus, laufen die uebrigen der Reihe nach
    durch, jedes fruehere also erst, wenn alle anderen dran waren.
    """
    routen = K["routen"]
    if not routen:
        return []
    gemerkt = STICKY.get("name")
    if not K.get("sticky") or not gemerkt:
        return list(routen)
    namen = [r["name"] for r in routen]
    if gemerkt not in namen:
        return list(routen)
    start = namen.index(gemerkt)
    return routen[start:] + routen[:start]

'''


PATCHES: list[tuple[str, str]] = [
    # --- 1) Kennung ---------------------------------------------------------
    # Nur die Kennung umhaengen, den Versionsstand NICHT: sonst haengt der Patch
    # an der Fassungsnummer des Ollama-Bahnhofs und bricht bei jedem Sprung
    # (passiert am 21.09.2026 beim Wechsel auf 1.2.0).
    (
        'KENNUNG = "Ollama-Bahnhof/" + FASSUNG',
        'KENNUNG = "Weichensteller (Ollama + LLM-Bahnhof)/" + FASSUNG',
    ),
    # --- 2) Vorgaben fuer die Verbindung ------------------------------------
    (
        '    "cors": True,\n}',
        '    "cors": True,\n'
        '    # --- Verbindung zum LLM-Bahnhof (das Neue im Weichensteller) ---\n'
        '    "llm_env": os.path.join(os.path.dirname(HIER), "llm-bahnhof", ".env"),\n'
        '    "reihenfolge": "llm-zuerst",  # llm-zuerst | eigene-zuerst | nur-llm | nur-eigene\n'
        '    "sticky": True,               # erfolgreiches Gleis merken (wie llm-bahnhof)\n'
        '}',
    ),
    # --- 3) Zaehler + Sticky-Speicher ---------------------------------------
    (
        'ZAEHLER = {"anfragen": 0, "simulationen": 0, "durchgereicht": 0, "fehlschlaege": 0}',
        'ZAEHLER = {"anfragen": 0, "simulationen": 0, "durchgereicht": 0, "fehlschlaege": 0,\n'
        '           "sticky_setzungen": 0}\n'
        '# Zuletzt erfolgreiches Gleis - der Kern des Sticky-Fallbacks.\n'
        'STICKY = {"name": None}',
    ),
    # --- 4) Gleise zusammenfuehren ------------------------------------------
    (
        '    e["routen"] = _routen_lesen(konf)',
        '    e["llm_env"] = hole("LLM_BAHNHOF_ENV", STILL["llm_env"])\n'
        '    e["reihenfolge"] = (hole("GLEIS_REIHENFOLGE", STILL["reihenfolge"])\n'
        '                       or "llm-zuerst").strip().lower()\n'
        '    e["sticky"] = _wahr(hole("STICKY_FALLBACK"), STILL["sticky"])\n'
        '    e["routen"] = _routen_zusammenfuehren(konf, e)',
    ),
    # --- 5) Startmeldung ----------------------------------------------------
    (
        "log.info(f\"Modus: {K['modus']} | Gleise: {len(K['routen'])} | Katalog: {len(K['katalog'])} Modell(e)\")",
        "log.info(f\"Modus: {K['modus']} | Gleise: {len(K['routen'])} | Katalog: {len(K['katalog'])} Modell(e)\")\n"
        "log.info(f\"Gleis-Reihenfolge: {K['reihenfolge']} | Sticky-Fallback: \"\n"
        "         f\"{'an' if K.get('sticky') else 'aus'} | LLM-Bahnhof-Konfiguration: {K.get('llm_env')}\")",
    ),
    # --- 6) Sticky im Gleis-Durchlauf ---------------------------------------
    (
        '    fehler = []\n    for route in K["routen"]:',
        '    fehler = []\n    for route in _gleis_reihenfolge():',
    ),
    (
        "        log.info(f\"Gleis {route['name']} antwortet (Modell {route['modell'] or intern['modell']}).\")",
        "        if STICKY.get(\"name\") != route[\"name\"]:\n"
        "            ZAEHLER[\"sticky_setzungen\"] += 1\n"
        "        STICKY[\"name\"] = route[\"name\"]\n"
        "        log.info(f\"Gleis {route['name']} antwortet (Modell {route['modell'] or intern['modell']}).\")",
    ),
    # --- 7) /health ---------------------------------------------------------
    (
        '        "gleise": [{"name": r["name"], "url": r["url"], "modell": r["modell"],\n'
        '                    "timeout": r["timeout"]} for r in K["routen"]],',
        '        "gleise": [{"name": r["name"], "gruppe": r.get("gruppe", ""),\n'
        '                    "url": r["url"], "modell": r["modell"],\n'
        '                    "timeout": r["timeout"]} for r in K["routen"]],\n'
        '        "gleis_reihenfolge": K["reihenfolge"],\n'
        '        "sticky_fallback": K.get("sticky", False),\n'
        '        "sticky_gleis": STICKY.get("name"),\n'
        '        "llm_bahnhof_konfiguration": K.get("llm_env"),\n'
        '        "gleis_start": [r["name"] for r in _gleis_reihenfolge()],',
    ),
    # --- 8) --pruefen -------------------------------------------------------
    (
        '    print(f"Ollama-Bahnhof {FASSUNG}")\n    print(f"Konfiguration      : {quelle}")',
        '    print(f"Weichensteller (Ollama + LLM-Bahnhof) {FASSUNG}")\n'
        '    print(f"Konfiguration      : {quelle}")',
    ),
    (
        '    print(f"Gleise             : {len(K[\'routen\'])}")\n',
        '    print(f"Gleise             : {len(K[\'routen\'])}   "\n'
        '          f"(Reihenfolge: {K[\'reihenfolge\']}, Sticky-Fallback: "\n'
        '          f"{\'an\' if K.get(\'sticky\') else \'aus\'})")\n'
        '    llm_pfad = K.get("llm_env") or ""\n'
        '    vorhanden = "vorhanden" if (llm_pfad and os.path.exists(llm_pfad)) else "nicht vorhanden"\n'
        '    print(f"LLM-Bahnhof-Konfig.: {llm_pfad} ({vorhanden})")\n',
    ),
    (
        '        print(f"  - {route[\'name\']} {route[\'url\']} | Modell {route[\'modell\'] or \'(vom Client)\'}"\n'
        '              f" | Timeout {route[\'timeout\']} | {schluessel}")',
        '        print(f"  - {route[\'name\']} [{route.get(\'gruppe\', \'-\')}] {route[\'url\']}"\n'
        '              f" | Modell {route[\'modell\'] or \'(vom Client)\'}"\n'
        '              f" | Timeout {route[\'timeout\']} | {schluessel}")',
    ),
    (
        '    print(f"Katalog            : {len(K[\'katalog\'])} Modell(e) aus der Konfiguration, "',
        '    start = [r["name"] for r in _gleis_reihenfolge()]\n'
        '    if start and K["routen"] and start[0] != K["routen"][0]["name"]:\n'
        '        print(f"  (Sticky: naechste Anfrage beginnt bei {start[0]})")\n'
        '    print(f"Katalog            : {len(K[\'katalog\'])} Modell(e) aus der Konfiguration, "',
    ),
]


def main() -> int:
    text = DATEI.read_text(encoding="utf-8")
    fehlend = [alt for alt, _ in PATCHES if alt not in text]
    if fehlend:
        print("  ✗ Ankerstelle(n) nicht gefunden:")
        for a in fehlend:
            print(f"      {a.splitlines()[0][:70]}")
        return 1
    for alt, neu in PATCHES:
        text = text.replace(alt, neu, 1)

    # Neue Funktionen vor _virtual_eintrag einsetzen (Whitespace-tolerant)
    muster = re.compile(r"\n\n+def _virtual_eintrag\(e\):")
    if not muster.search(text):
        print("  ✗ Einfuegestelle fuer die neuen Funktionen nicht gefunden")
        return 1
    text = muster.sub(lambda _m: NEUE_FUNKTIONEN + "\n\ndef _virtual_eintrag(e):", text, count=1)

    # Ueberschrift des Modulkopfs ebenfalls ohne Versionsstand - gleicher Grund
    # wie oben bei der Kennung.
    kopf = re.compile(r"Ollama-Bahnhof  \(v[\d.]+\)\n=+\n")
    if not kopf.search(text):
        print("  ✗ Ueberschrift im Modulkopf nicht gefunden")
        return 1
    text = kopf.sub("Weichensteller (Ollama + LLM-Bahnhof)\n" + "=" * 35 + "\n", text, count=1)

    DATEI.write_text(text, encoding="utf-8")
    print(f"  ✓ {len(PATCHES)} Stelle(n) gepatcht + neue Funktionen + Modulkopf: {DATEI}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
