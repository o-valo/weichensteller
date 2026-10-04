#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pruefungen fuer die Verbindung: zwei Gleis-Quellen + Sticky-Fallback.

Start:  ./venv/bin/python3 -m unittest discover -s tests -v

Diese Pruefungen brauchen kein Netz: die Gleisliste des LLM-Bahnhofs wird in
eine temporaere Datei geschrieben, die eigenen Gleise direkt uebergeben.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

HIER = os.path.dirname(os.path.abspath(__file__))
WURZEL = os.path.dirname(HIER)
sys.path.insert(0, WURZEL)

# Eigene Dateien fuer die Gleis-Tests (die Funktionen bekommen ihre Pfade
# ausdruecklich uebergeben - sie lesen nicht die globale Konfiguration).
TESTORDNER = tempfile.mkdtemp(prefix="weichensteller-verbindung-")

# Das Modul und seine Konfiguration kommen aus dem gemeinsamen Test-Setup.
#
# Wichtig: unittest entdeckt die Testdateien alphabetisch, und wer zuerst
# importiert wird, legt fuer ALLE die Konfiguration fest. Vorher setzte diese
# Datei hier eigene Werte und lud das Modul selbst - dadurch hing das Ergebnis
# vom Dateinamen ab. Der Umbau von "test_test_bahnhof.py" zu
# "test_weichensteller.py" hat die Reihenfolge gedreht (jetzt ist v vor w) und
# damit 11 Pruefungen sichtbar brechen lassen. Das Setup wird deshalb nur an
# EINER Stelle gebaut.
import test_weichensteller  # noqa: E402,F401  (setzt WEICHENSTELLER_ENV, laedt das Modul)
wst = test_weichensteller.wst


def llm_datei(inhalt: str) -> str:
    """Konfigurationsdatei eines 'LLM-Bahnhofs' anlegen und den Pfad liefern."""
    pfad = os.path.join(TESTORDNER, "llm-wst.env")
    with open(pfad, "w", encoding="utf-8") as f:
        f.write(inhalt)
    return pfad


EIGENE = {
    "ROUTE_01": "http://eigen.invalid:8000/v1|none|eigen-modell|30s",
}


class ZweiQuellenTest(unittest.TestCase):
    """Gleise aus zwei Quellen, in fester Reihenfolge."""

    def test_llm_zuerst_ist_die_vorgabe(self):
        pfad = llm_datei("ROUTE_01=https://llm.invalid/v1|key|llm-modell|20s\n"
                         "ROUTE_02=https://llm.invalid/v1|key|llm-modell-2|20s\n")
        routen = wst._routen_zusammenfuehren(
            EIGENE, {"llm_env": pfad, "reihenfolge": "llm-zuerst"})

        self.assertEqual([r["name"] for r in routen],
                         ["llm-ROUTE_01", "llm-ROUTE_02", "ROUTE_01"])
        self.assertEqual([r["gruppe"] for r in routen], ["llm", "llm", "eigene"])
        self.assertEqual([r["modell"] for r in routen],
                         ["llm-modell", "llm-modell-2", "eigen-modell"])

    def test_eigene_zuerst_umstellbar(self):
        pfad = llm_datei("ROUTE_01=https://llm.invalid/v1|key|llm-modell|20s\n")
        routen = wst._routen_zusammenfuehren(
            EIGENE, {"llm_env": pfad, "reihenfolge": "eigene-zuerst"})
        self.assertEqual([r["name"] for r in routen], ["ROUTE_01", "llm-ROUTE_01"])

    def test_nur_llm_und_nur_eigene(self):
        pfad = llm_datei("ROUTE_01=https://llm.invalid/v1|key|llm-modell|20s\n")
        nur_llm = wst._routen_zusammenfuehren(
            EIGENE, {"llm_env": pfad, "reihenfolge": "nur-llm"})
        nur_eigene = wst._routen_zusammenfuehren(
            EIGENE, {"llm_env": pfad, "reihenfolge": "nur-eigene"})
        self.assertEqual([r["name"] for r in nur_llm], ["llm-ROUTE_01"])
        self.assertEqual([r["name"] for r in nur_eigene], ["ROUTE_01"])

    def test_fehlende_llm_datei_stoert_nicht(self):
        routen = wst._routen_zusammenfuehren(
            EIGENE, {"llm_env": os.path.join(TESTORDNER, "gibtsnicht.env"),
                     "reihenfolge": "llm-zuerst"})
        self.assertEqual([r["name"] for r in routen], ["ROUTE_01"])

    def test_gleiches_modell_aus_beiden_quellen_bleibt_getrennt(self):
        """Zwei Quellen mit gleicher ROUTE-Nummer duerfen sich nicht ueberschreiben."""
        pfad = llm_datei("ROUTE_01=https://llm.invalid/v1|key|llm-modell|20s\n")
        routen = wst._routen_zusammenfuehren(
            EIGENE, {"llm_env": pfad, "reihenfolge": "llm-zuerst"})
        namen = [r["name"] for r in routen]
        self.assertEqual(len(namen), len(set(namen)))


class GleisZeileTest(unittest.TestCase):
    """Fremde Konfigurationen lesen: fehlendes Modellfeld."""

    def test_zeitangabe_im_modellfeld_wird_zum_timeout(self):
        routen = wst._routen_lesen(
            {"ROUTE_01": "https://api.invalid/v1|key|45s"})
        self.assertEqual(routen[0]["modell"], "")
        self.assertEqual(routen[0]["timeout"], 45.0)

    def test_vollstaendige_zeile_bleibt_unveraendert(self):
        routen = wst._routen_lesen(
            {"ROUTE_01": "https://api.invalid/v1|key|mein-modell|90s"})
        self.assertEqual(routen[0]["modell"], "mein-modell")
        self.assertEqual(routen[0]["timeout"], 90.0)

    def test_modellname_ohne_einheit_bleibt_modellname(self):
        """Ein Name wie 'llama3' darf nie als Zeitangabe verschwinden."""
        for name in ("llama3", "qwen3.6:8b", "gpt-oss:20b", "7b"):
            routen = wst._routen_lesen(
                {"ROUTE_01": f"https://api.invalid/v1|key|{name}"})
            self.assertEqual(routen[0]["modell"], name, f"{name} wurde verschluckt")


class StickyTest(unittest.TestCase):
    """Sticky-Fallback: am erfolgreichen Gleis haengen bleiben."""

    def setUp(self):
        self.original = wst.K["routen"]
        self.original_sticky = wst.K.get("sticky")
        self.original_name = wst.STICKY.get("name")
        wst.K["routen"] = [
            {"name": "ROUTE_01"},
            {"name": "ROUTE_02"},
            {"name": "ROUTE_03"},
        ]

    def tearDown(self):
        wst.K["routen"] = self.original
        wst.K["sticky"] = self.original_sticky
        wst.STICKY["name"] = self.original_name

    def test_ohne_merkmal_beginnen_alle_beim_ersten(self):
        wst.K["sticky"] = True
        wst.STICKY["name"] = None
        self.assertEqual([r["name"] for r in wst._gleis_reihenfolge()],
                         ["ROUTE_01", "ROUTE_02", "ROUTE_03"])

    def test_startet_am_gemerkten_gleis_und_laeuft_im_kreis(self):
        wst.K["sticky"] = True
        wst.STICKY["name"] = "ROUTE_02"
        self.assertEqual([r["name"] for r in wst._gleis_reihenfolge()],
                         ["ROUTE_02", "ROUTE_03", "ROUTE_01"])

    def test_abgeschaltet_beginnen_alle_beim_ersten(self):
        wst.K["sticky"] = False
        wst.STICKY["name"] = "ROUTE_03"
        self.assertEqual([r["name"] for r in wst._gleis_reihenfolge()],
                         ["ROUTE_01", "ROUTE_02", "ROUTE_03"])

    def test_unbekanntes_merkmal_faellt_zurueck(self):
        wst.K["sticky"] = True
        wst.STICKY["name"] = "ROUTE_99"
        self.assertEqual([r["name"] for r in wst._gleis_reihenfolge()],
                         ["ROUTE_01", "ROUTE_02", "ROUTE_03"])

    def test_ohne_gleise_leere_reihenfolge(self):
        wst.K["routen"] = []
        self.assertEqual(wst._gleis_reihenfolge(), [])


if __name__ == "__main__":
    unittest.main()
