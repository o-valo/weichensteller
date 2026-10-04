#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Weichensteller -- GNU Affero General Public License v3.0 oder spaeter.
# Die vollstaendige Lizenz steht in LICENSE.
#
# Abgeleitet aus dem Ollama-Bahnhof (v1.2.0), der seinerseits aus dem
# LLM-Bahnhof entstanden ist. An der Ollama-Oberflaeche wurde nichts
# geaendert; ergaenzt wurden zwei Gleis-Quellen mit fester Reihenfolge,
# Sticky-Fallback, die Responses-API und die erweiterte Diagnose.
#
"""
Weichensteller (Ollama + LLM-Bahnhof)
===================================

Ein LLM-Proxy, der nach aussen einen **kompletten Ollama-Server simuliert**.
Die Antworten kommen entweder von echten, OpenAI-kompatiblen Gleisen (Routen,
gleiches Format wie beim llm-bahnhof) oder -- wenn keine Route da ist bzw. alle
ausfallen -- aus der eingebauten Simulation.

Damit koennen Clients, die "Ollama koennen" (Open WebUI, Continue, Cline,
Aider, LangChain/Ollama, ollama-python, Claude Code ueber die Kompat-Schicht,
Home-Assistant-Integrationen, ...) gegen den Weichensteller laufen, ohne dass lokal
eine GPU steht.

Bediente Schnittstellen
-----------------------
Ollama nativ:
    GET    /                      "Ollama is running"
    GET    /api/version           Versionsnummer
    GET    /api/tags              Modellkatalog
    GET    /api/ps                geladene Modelle (keep_alive)
    POST   /api/show              Modelfile, Template, Parameter, Faehigkeiten
    POST   /api/chat              Chat (NDJSON-Stream, Denken, Werkzeuge, Bilder)
    POST   /api/generate          Textvervollstaendigung (NDJSON-Stream)
    POST   /api/embed             Einbettungen
    POST   /api/embeddings        Einbettungen (Alt-Fassung, ein Vektor)
    POST   /api/pull              Modell "ziehen" (Fortschritt, Katalog)
    POST   /api/push              Modell "schieben" (Fortschritt)
    POST   /api/create            Modell anlegen
    POST   /api/copy              Modell umbenennen
    DELETE /api/delete            Modell loeschen
    POST   /api/blobs/<digest>    Blob hochladen (Pruefsumme wird geprueft)
    HEAD   /api/blobs/<digest>    Blob vorhanden?

OpenAI-kompatibel (wie Ollama sie zusaetzlich anbietet):
    GET    /v1/models
    POST   /v1/chat/completions   SSE-Stream oder JSON
    POST   /v1/responses          SSE-Stream oder JSON (Responses-API)
    POST   /v1/completions
    POST   /v1/embeddings

Eigene Diagnose:
    GET    /health                Modus, Gleise, Katalog, Zaehler

Pseudo-Modell
-------------
VIRTUAL_MODEL aus der .env ist der Name, den ein Client anspricht - wie
VIRTUAL_MODEL beim llm-bahnhof. Es steht immer im Katalog. Ist ein Gleis
konfiguriert, geht die Anfrage mit dem Ziel-Modell des Gleises nach vorne;
ohne Gleis (oder wenn alle ausfallen) antwortet die Simulation.

Konfiguration: .env im Format des llm-bahnhofs (KEY=WERT, # Kommentar, eine
[ABSCHNITT]-Kopfzeile wird uebergangen). Pfad: WEICHENSTELLER_ENV bzw. DOTENV_PATH,
sonst ./.env. Vorrang: .env > Umgebungsvariable > Vorgabe.

Start:  ./venv/bin/python3 weichensteller.py
Pruefen: ./venv/bin/python3 weichensteller.py --pruefen
"""

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

import requests
from flask import Flask, Response, jsonify, request

FASSUNG = "1.2.0"
KENNUNG = "Weichensteller (Ollama + LLM-Bahnhof)/" + FASSUNG

HIER = os.path.dirname(os.path.abspath(__file__))
STANDARD_ENV = os.path.join(HIER, ".env")
STANDARD_LOG = os.path.join(HIER, "weichensteller.log")

# Feste Vorgaben, wenn in der Konfiguration nichts steht.
STILL = {
    "modus": "auto",            # auto | simulation | durchreichen
    "sim_fallback": False,      # alle Gleise ausgefallen -> simulieren?
    "sim_version": "0.34.2",
    "sim_marker": True,         # Hinweis "Simulation" im Text (Ehrlichkeit)
    "sim_delay_ms": 20,         # Pause je Streaming-Haeppchen
    "sim_embed_dim": 768,
    "sim_toolcall": "auto",     # auto | aus | <Werkzeugname>
    "sim_unbekannt": "annehmen",  # annehmen | ablehnen (ablehnen = echtes Ollama)
    "sim_tokens_pro_s": 25.0,   # angenommene Geschwindigkeit fuer die Zaehlwerte
    "sim_keep_alive": "5m",
    "virtual_model": "weichensteller:latest",  # Name des Pseudo-Modells
    "virtual_familie": "weichensteller",
    "virtual_faehigkeiten": "completion,tools,thinking",
    "port": 11434,
    "host": "0.0.0.0",
    "cors": True,
    # --- Verbindung zum LLM-Bahnhof (das Neue im Weichensteller) ---
    "llm_env": os.path.join(os.path.dirname(HIER), "llm-bahnhof", ".env"),
    "reihenfolge": "llm-zuerst",  # llm-zuerst | eigene-zuerst | nur-llm | nur-eigene
    "sticky": True,               # erfolgreiches Gleis merken (wie llm-bahnhof)
}

# Beispielkatalog, solange in der Konfiguration keine MODEL_xx stehen.
STANDARD_MODELLE = [
    ("gemma4:8b", "8.0B", "Q4_K_M", "gemma4", "completion,tools,thinking", 5.2),
    ("qwen3.6:8b", "8.2B", "Q4_K_M", "qwen3", "completion,tools,thinking", 5.4),
    ("granite4.1:8b", "8.2B", "Q4_K_M", "granite", "completion,tools", 5.1),
    ("embeddinggemma:300m", "0.3B", "F16", "embeddinggemma", "embedding", 0.6),
]

log = logging.getLogger("OLLAMA_BAHNHOF")


def _logger_setzen(ziel):
    """Konsole + Protokolldatei (wie bei der Heinzelmaennchen-Bruecke)."""
    log.setLevel(logging.INFO)
    log.handlers.clear()
    form = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    konsole = logging.StreamHandler(sys.stdout)
    konsole.setFormatter(form)
    log.addHandler(konsole)
    try:
        datei = logging.FileHandler(ziel, encoding="utf-8")
        datei.setFormatter(form)
        log.addHandler(datei)
    except OSError as exc:                                  # z. B. Rechte
        log.warning(f"Protokolldatei {ziel} nicht schreibbar: {exc}")


_logger_setzen(os.getenv("WEICHENSTELLER_LOG", STANDARD_LOG))


def _zeitstempel(zeitpunkt=None):
    """Ollama-Zeitstempel mit Nanosekunden und Zonenversatz."""
    stempel = time.time() if zeitpunkt is None else float(zeitpunkt)
    jetzt = datetime.fromtimestamp(stempel).astimezone()
    ns = int(stempel * 1e9) % 1_000_000_000
    zone = jetzt.strftime("%z") or "+0000"
    return jetzt.strftime("%Y-%m-%dT%H:%M:%S") + f".{ns:09d}" + zone[:3] + ":" + zone[3:]


def _iso(zeitpunkt):
    """Zeitstempel aus Sekunden seit Epoche oder aus einem datetime."""
    if isinstance(zeitpunkt, (int, float)):
        return _zeitstempel(zeitpunkt)
    return _zeitstempel(zeitpunkt.timestamp())


def _nanosekunden(sekunden):
    return int(max(0.0, sekunden) * 1e9)


# ---------------------------------------------------------------- Konfiguration

def _env_lesen(pfad):
    """KEY=WERT-Zeilen einer .env lesen (Format des llm-bahnhofs).

    Kommentare mit # oder ;, eine Kopfzeile wie [TEMPLATE] und ein fuehrendes
    'export' werden uebergangen; umschliessende Anfuehrungszeichen fallen weg.
    """
    konf = {}
    try:
        with open(pfad, "r", encoding="utf-8") as f:
            for zeile in f:
                zeile = zeile.strip()
                if not zeile or zeile[0] in "#;[":
                    continue
                if zeile.lower().startswith("export "):
                    zeile = zeile[7:].lstrip()
                if "=" in zeile:
                    schluessel, wert = zeile.split("=", 1)
                    wert = wert.strip()
                    if len(wert) > 1 and wert[0] == wert[-1] and wert[0] in "\"'":
                        wert = wert[1:-1]
                    konf[schluessel.strip().upper()] = wert
    except FileNotFoundError:
        log.info(f"Keine Konfigurationsdatei unter {pfad} - es gelten die Vorgaben.")
    except OSError as exc:
        log.warning(f"Konfiguration {pfad} nicht lesbar: {exc}")
    return konf


def _wahr(wert, standard=False):
    """Robuste Wahr-Auswertung (1/ja/true/on ...), auch als String."""
    if wert is None:
        return standard
    if isinstance(wert, bool):
        return wert
    return str(wert).strip().lower() in ("1", "ja", "yes", "true", "on", "an")


def _zahl(wert, standard, art=int):
    try:
        return art(str(wert).strip())
    except (TypeError, ValueError):
        log.warning(f"Ungueltiger Zahlenwert '{wert}' - verwende {standard}")
        return standard


def _sekunden(wert, standard=60.0):
    """'30', '90s', '15m', '2h' -> Sekunden; 0 -> None (= kein Timeout)."""
    roh = str(wert).strip().lower() if wert is not None else ""
    if not roh:
        roh = str(standard).strip().lower()
    if roh == "0":
        return None
    treffer = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smh]?)", roh)
    if not treffer:
        log.warning(f"Ungueltiger Timeout '{wert}' - verwende {standard}")
        return _sekunden(standard)
    anzahl = float(treffer.group(1))
    einheit = treffer.group(2)
    return {"": anzahl, "s": anzahl, "m": anzahl * 60, "h": anzahl * 3600}[einheit]


def _zeit_lesen(wert, standard=300.0):
    """keep_alive: '5m', '300', '-1' (nie entladen), 0 -> 0 Sekunden."""
    roh = str(wert).strip().lower() if wert is not None else ""
    if not roh:
        return standard
    if roh in ("-1", "-1s", "unbegrenzt", "nie"):
        return math.inf
    treffer = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*([smh]?)", roh)
    if not treffer:
        return standard
    anzahl = float(treffer.group(1))
    if anzahl < 0:
        return math.inf
    return anzahl * {"": 1, "s": 1, "m": 60, "h": 3600}[treffer.group(2)]


def _digest(text):
    """Ollama-artige Pruefsumme zu einem Modellnamen."""
    return "sha256:" + hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _normalisiere_chat_url(basis):
    b = str(basis).strip().rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    if re.search(r"/v\d+$", b):
        return b + "/chat/completions"
    return b + "/v1/chat/completions"


def _normalisiere_embed_url(basis):
    b = str(basis).strip().rstrip("/")
    if b.endswith("/embeddings"):
        return b
    return re.sub(r"/chat/completions$", "/embeddings", _normalisiere_chat_url(b))


def _ist_zeitangabe(text):
    """Sieht der Wert nach einer Dauer aus ("45s", "10m", "1h")?

    Noetig, weil fremde Konfigurationen (z. B. die des llm-bahnhofs) das
    Modellfeld gern weglassen: URL|Schluessel|Timeout. Ohne diese Pruefung
    waere "45s" der Modellname. Eine Dauer ohne Einheit zaehlt bewusst
    NICHT als Zeitangabe - ein Modellname soll nie still verschwinden.
    """
    return bool(re.fullmatch(r"\d+(\.\d+)?\s*[smh]", (text or "").strip()))


def _routen_lesen(konf):
    """ROUTE_01=URL|Schluessel|Modell|Timeout - Reihenfolge = Gleisreihenfolge."""
    eintraege = []
    for schluessel, wert in konf.items():
        treffer = re.fullmatch(r"ROUTE_(\d+)", schluessel)
        if treffer:
            eintraege.append((int(treffer.group(1)), schluessel, wert))
    eintraege.sort()

    routen = []
    for nummer, schluessel, roh in eintraege:
        teile = [t.strip() for t in roh.split("|")]
        url = teile[0] if teile else ""
        if not url or "example" in url:
            log.warning(f"{schluessel} uebersprungen (keine gueltige URL: '{url}')")
            continue
        modell = teile[2] if len(teile) > 2 else ""
        zeit = teile[3] if len(teile) > 3 else ""
        if _ist_zeitangabe(modell) and not zeit:
            zeit, modell = modell, ""
        routen.append({
            "name": f"ROUTE_{nummer:02d}",
            "url": _normalisiere_chat_url(url),
            "embed_url": _normalisiere_embed_url(url),
            "key": teile[1] if len(teile) > 1 else "",
            "modell": modell,
            "timeout": _sekunden(zeit, 600.0),
        })
    return routen
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



def _virtual_eintrag(e):
    """Der Katalogeintrag des Pseudo-Modells (Name aus VIRTUAL_MODEL).

    Er ist der Einstieg fuer Clients (wie VIRTUAL_MODEL beim llm-bahnhof):
    konfigurierte Gleise senden ihr eigenes Ziel-Modell, ohne Gleis antwortet
    die Simulation.
    """
    name = e["virtual_model"]
    return _modell_eintrag(name, parameter="8.0B", quantisierung="Q4_K_M",
                           familie=e["virtual_familie"] or name.split(":")[0].split("/")[-1],
                           faehigkeiten=e["virtual_faehigkeiten"], groesse=4.7)


def _katalog_lesen(konf):
    """MODEL_01=Name|Parameter|Quantisierung|Familie|Faehigkeiten|Groesse_GB"""
    eintraege = []
    for schluessel, wert in konf.items():
        treffer = re.fullmatch(r"MODEL_(\d+)", schluessel)
        if treffer:
            eintraege.append((int(treffer.group(1)), wert))
    eintraege.sort()

    katalog = []
    if eintraege:
        for _, roh in eintraege:
            teile = [t.strip() for t in roh.split("|")]
            if not teile or not teile[0]:
                continue
            katalog.append(_modell_eintrag(
                teile[0],
                parameter=teile[1] if len(teile) > 1 and teile[1] else "8.0B",
                quantisierung=teile[2] if len(teile) > 2 and teile[2] else "Q4_K_M",
                familie=teile[3] if len(teile) > 3 and teile[3] else "llama",
                faehigkeiten=teile[4] if len(teile) > 4 and teile[4] else "completion",
                groesse=_zahl(teile[5], 5.0, float) if len(teile) > 5 and teile[5] else 5.0,
            ))
    else:
        for name, parameter, quant, familie, faehig, groesse in STANDARD_MODELLE:
            katalog.append(_modell_eintrag(name, parameter=parameter, quantisierung=quant,
                                           familie=familie, faehigkeiten=faehig, groesse=groesse))
    return katalog


def _modell_eintrag(name, parameter="8.0B", quantisierung="Q4_K_M", familie="llama",
                    faehigkeiten="completion", groesse=5.0):
    return {
        "name": name,
        "model": name,
        "parameter_size": parameter,
        "quantization_level": quantisierung,
        "family": familie or name.split(":")[0].split("/")[-1],
        "capabilities": [c.strip() for c in str(faehigkeiten).split(",") if c.strip()],
        "size": int(float(groesse) * 1024 ** 3),
        "digest": _digest(name),
        "modified_at": _zeitstempel(),
    }


def _einstellungen(konf):
    """Konfigurationsdatei > Umgebungsvariable > Vorgabe."""
    def hole(schluessel, standard=None):
        if schluessel in konf and str(konf[schluessel]).strip() != "":
            return str(konf[schluessel]).strip()
        if os.getenv(schluessel):
            return os.getenv(schluessel)
        return standard

    e = {}
    e["modus"] = (hole("MODUS", STILL["modus"]) or "auto").strip().lower()
    if e["modus"] not in ("auto", "simulation", "durchreichen"):
        log.warning(f"Unbekannter MODUS '{e['modus']}' - verwende 'auto'")
        e["modus"] = "auto"
    e["sim_fallback"] = _wahr(hole("SIM_FALLBACK"), STILL["sim_fallback"])
    e["sim_version"] = hole("SIM_VERSION", STILL["sim_version"])
    e["sim_marker"] = _wahr(hole("SIM_MARKER"), STILL["sim_marker"])
    e["sim_delay_ms"] = _zahl(hole("SIM_DELAY_MS"), STILL["sim_delay_ms"])
    e["sim_embed_dim"] = _zahl(hole("SIM_EMBED_DIM"), STILL["sim_embed_dim"])
    e["sim_toolcall"] = (hole("SIM_TOOLCALL", STILL["sim_toolcall"]) or "auto").strip()
    e["sim_unbekannt"] = (hole("SIM_UNBEKANNT", STILL["sim_unbekannt"]) or "annehmen").strip().lower()
    e["sim_tokens_pro_s"] = _zahl(hole("SIM_TOKENS_PRO_S"), STILL["sim_tokens_pro_s"], float)
    e["sim_keep_alive"] = hole("SIM_KEEP_ALIVE", STILL["sim_keep_alive"])
    e["virtual_model"] = (hole("VIRTUAL_MODEL", STILL["virtual_model"])
                         or STILL["virtual_model"]).strip()
    e["virtual_familie"] = (hole("VIRTUAL_MODEL_FAMILIE", STILL["virtual_familie"]) or "").strip()
    e["virtual_faehigkeiten"] = hole("VIRTUAL_MODEL_FAEHIGKEITEN", STILL["virtual_faehigkeiten"])
    e["port"] = _zahl(hole("WEICHENSTELLER_PORT", hole("PORT")), STILL["port"])
    e["host"] = hole("WEICHENSTELLER_HOST", hole("HOST", STILL["host"])) or STILL["host"]
    e["cors"] = _wahr(hole("WEICHENSTELLER_CORS"), STILL["cors"])
    # "~" aufloesen - sonst findet os.path.exists die Datei nicht.
    e["llm_env"] = os.path.expanduser(hole("LLM_BAHNHOF_ENV", STILL["llm_env"]))
    e["reihenfolge"] = (hole("GLEIS_REIHENFOLGE", STILL["reihenfolge"])
                       or "llm-zuerst").strip().lower()
    e["sticky"] = _wahr(hole("STICKY_FALLBACK"), STILL["sticky"])
    e["routen"] = _routen_zusammenfuehren(konf, e)
    # Das Pseudo-Modell steht immer mit im Katalog. Definiert eine MODEL_xx-Zeile
    # denselben Namen, gilt die ausfuehrlichere Angabe aus der .env.
    katalog = _katalog_lesen(konf)
    e["virtual"] = next((m for m in katalog if m["name"] == e["virtual_model"]), None) \
        or _virtual_eintrag(e)
    e["katalog"] = [e["virtual"]] + [m for m in katalog if m is not e["virtual"]]
    e["state"] = hole("WEICHENSTELLER_STATE", os.path.join(HIER, "state"))
    e["log"] = hole("WEICHENSTELLER_LOG", STANDARD_LOG)
    return e


ENV_PFAD = os.getenv("WEICHENSTELLER_ENV") or os.getenv("DOTENV_PATH") or STANDARD_ENV
ENV = _env_lesen(ENV_PFAD)
K = _einstellungen(ENV)

if K["routen"]:
    log.info(f"Konfiguration: {ENV_PFAD}")
log.info(f"Modus: {K['modus']} | Gleise: {len(K['routen'])} | Katalog: {len(K['katalog'])} Modell(e)")
log.info(f"Gleis-Reihenfolge: {K['reihenfolge']} | Sticky-Fallback: "
         f"{'an' if K.get('sticky') else 'aus'} | LLM-Bahnhof-Konfiguration: {K.get('llm_env')}")
ziel = "-> Gleis-Zielmodelle werden benutzt" if K["routen"] else "-> wird simuliert"
log.info(f"Pseudo-Modell (Einstieg fuer Clients): {K['virtual_model']} {ziel}")
if K["virtual_model"] in {r["modell"] for r in K["routen"]}:
    log.warning("Ein Gleis traegt das Pseudo-Modell als Ziel-Modell ein - dort gibt es "
                "dieses Modell nicht.")

BLOB_PFAD = os.path.join(K["state"], "blobs")
MODELL_PFAD = os.path.join(K["state"], "modelle.json")

# ------------------------------------------------------------- Zustand (Speicher)

_ZUSTAND_SPERRE = threading.Lock()
MODELLE = {}          # Name -> Modelleintrag (Katalog + gezogene Modelle)
GELADEN = {}          # Name -> {"seit", "expires_at", ...} fuer /api/ps
ZAEHLER = {"anfragen": 0, "simulationen": 0, "durchgereicht": 0, "fehlschlaege": 0,
           "sticky_setzungen": 0}
# Zuletzt erfolgreiches Gleis - der Kern des Sticky-Fallbacks.
STICKY = {"name": None}


def _zustand_laden():
    """Katalog: Vorgaben aus der Konfiguration, dann gezogene Modelle ergaenzen."""
    MODELLE.clear()
    for eintrag in K["katalog"]:
        MODELLE[eintrag["name"]] = dict(eintrag)
    try:
        with open(MODELL_PFAD, "r", encoding="utf-8") as f:
            for eintrag in json.load(f):
                MODELLE[eintrag["name"]] = eintrag
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        log.warning(f"Zustandsdatei {MODELL_PFAD} nicht lesbar: {exc}")


def _zustand_speichern():
    """Nur die gezogenen/angelegten Modelle sichern (der Rest kommt aus der Konf)."""
    fest = {e["name"] for e in K["katalog"]}
    beweglich = [m for name, m in MODELLE.items() if name not in fest]
    try:
        os.makedirs(K["state"], exist_ok=True)
        with open(MODELL_PFAD, "w", encoding="utf-8") as f:
            json.dump(beweglich, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        log.warning(f"Zustand konnte nicht gesichert werden: {exc}")


def _modell_finden(name):
    """Name aufloesen; ohne Tag wird ':latest' ergaenzt (wie bei Ollama)."""
    if not name:
        return None
    if name in MODELLE:
        return MODELLE[name]
    with_tag = name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"
    return MODELLE.get(with_tag)


def _geladen_melden(name, keep_alive):
    """Modell als "geladen" fuehren (Grundlage fuer /api/ps)."""
    eintrag = _modell_finden(name) or MODELLE.get(name) or _modell_eintrag(name)
    dauer = _zeit_lesen(keep_alive, _zeit_lesen(K["sim_keep_alive"]))
    mit_sperre = {
        "name": eintrag["name"],
        "model": eintrag["name"],
        "size": eintrag["size"],
        "digest": eintrag["digest"],
        "details": _details(eintrag),
        "expires_at": None if dauer == math.inf else _iso(time.time() + dauer),
        "size_vram": eintrag["size"],
    }
    with _ZUSTAND_SPERRE:
        GELADEN[eintrag["name"]] = mit_sperre
    return mit_sperre


def _geladen_aufraeumen():
    jetzt = time.time()
    with _ZUSTAND_SPERRE:
        for name in list(GELADEN):
            ende = GELADEN[name]["expires_at"]
            if ende:
                try:
                    if datetime.fromisoformat(ende).timestamp() <= jetzt:
                        del GELADEN[name]
                except ValueError:
                    del GELADEN[name]


def _details(eintrag):
    if not eintrag:
        return {}
    return {
        "parent_model": "",
        "format": "gguf",
        "family": eintrag["family"],
        "families": [eintrag["family"]],
        "parameter_size": eintrag["parameter_size"],
        "quantization_level": eintrag["quantization_level"],
    }


def _faehigkeiten(eintrag):
    if eintrag and eintrag.get("capabilities"):
        return list(eintrag["capabilities"])
    return ["completion"]


def _umfang(name):
    return {"name": name, "model": name, "modified_at": _zeitstempel(),
            "size": 0, "digest": _digest(name), "details": _details(None)}


_zustand_laden()


# ------------------------------------------------------------------ Neutrale Form
# Ereignisse sind der gemeinsame Nenner aller Ausgabeformate:
#   {"typ": "text", "text": ...}        sichtbarer Text
#   {"typ": "denken", "text": ...}      Denkanteil (thinking)
#   {"typ": "werkzeug", "aufrufe": ...} Werkzeugaufrufe (Liste, Ollama-Form)
#   {"typ": "fehler", "meldung": ...}   Abbruch mitten im Strom
#   {"typ": "ende", "grund": ..., }     Abschluss
# Nebenher fuellt der Erzeuger ein meta-Dict (Zaehlwerte, Gleis, Werkzeuge).

class KeinGleis(Exception):
    """Kein Gleis lieferte eine verwertbare Antwort (vor dem ersten Haeppchen)."""

    def __init__(self, fehler):
        super().__init__("Kein Gleis lieferte eine Antwort")
        self.fehler = fehler


def _ist_stream(wert):
    """Nur echte Wahr-Werte aktivieren Streaming (String-'false' ist falsch)."""
    if isinstance(wert, bool):
        return wert
    if isinstance(wert, (int, float)):
        return wert != 0
    if isinstance(wert, str):
        return wert.strip().lower() in ("1", "true", "yes", "on", "ja")
    return False


def _prompt_aus_messages(messages):
    """Nutz-Text einer Nachrichtenliste (fuer Inhaltsverzeichnis/Simulation)."""
    teile = []
    for m in messages or []:
        inhalt = m.get("content")
        if isinstance(inhalt, str):
            teile.append(inhalt)
        elif isinstance(inhalt, list):
            for block in inhalt:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        teile.append(block.get("text", ""))
                    elif block.get("type") == "image_url":
                        teile.append("[Bild]")
        if m.get("tool_calls"):
            teile.append(json.dumps(m["tool_calls"], ensure_ascii=False))
    return "\n".join(teile)


def _anzahl_bilder(messages):
    anzahl = 0
    for m in messages or []:
        anzahl += len(m.get("images") or [])
        inhalt = m.get("content")
        if isinstance(inhalt, list):
            anzahl += sum(1 for b in inhalt if isinstance(b, dict) and b.get("type") == "image_url")
    return anzahl


def _intern_chat(rohdaten):
    """Ollama-native Anfrage auf die interne Form bringen."""
    rohdaten = rohdaten or {}
    return {
        "modell": rohdaten.get("model") or "",
        "messages": rohdaten.get("messages") or [],
        "options": rohdaten.get("options") or {},
        "tools": rohdaten.get("tools") or [],
        "format": rohdaten.get("format"),
        "denken": rohdaten.get("think", False),
        "keep_alive": rohdaten.get("keep_alive"),
        "logprobs": bool(rohdaten.get("logprobs")),
        "top_logprobs": rohdaten.get("top_logprobs"),
    }


def _intern_aus_openai(rohdaten):
    """OpenAI-Anfrage (/v1/...) auf die interne Form bringen."""
    rohdaten = rohdaten or {}
    messages = []
    for m in rohdaten.get("messages") or []:
        if not isinstance(m, dict):
            continue
        rolle = m.get("role", "user")
        if rolle == "developer":
            rolle = "system"
        inhalt = m.get("content")
        neu = {"role": rolle}
        if isinstance(inhalt, list):
            texte, bilder = [], []
            for block in inhalt:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    texte.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    url = (block.get("image_url") or {}).get("url", "")
                    if "," in url:
                        bilder.append(url.split(",", 1)[1])
                    else:
                        bilder.append(url)
            neu["content"] = "\n".join(texte)
            if bilder:
                neu["images"] = bilder
        else:
            neu["content"] = inhalt if inhalt is not None else ""
        if m.get("tool_calls"):
            neu["tool_calls"] = m["tool_calls"]
        if rolle == "tool":
            neu["tool_name"] = m.get("name") or m.get("tool_call_id") or ""
        messages.append(neu)

    optionen = {}
    for ollama_name, openai_name in (("temperature", "temperature"), ("top_p", "top_p"),
                                     ("seed", "seed"), ("frequency_penalty", "frequency_penalty"),
                                     ("presence_penalty", "presence_penalty")):
        if rohdaten.get(openai_name) is not None:
            optionen[ollama_name] = rohdaten[openai_name]
    if rohdaten.get("max_tokens") is not None:
        optionen["num_predict"] = rohdaten["max_tokens"]
    if rohdaten.get("stop") is not None:
        optionen["stop"] = rohdaten["stop"]
    if rohdaten.get("top_k") is not None:
        optionen["top_k"] = rohdaten["top_k"]

    format_wunsch = None
    if isinstance(rohdaten.get("response_format"), dict):
        art = rohdaten["response_format"].get("type")
        if art == "json_object":
            format_wunsch = "json"
        elif art == "json_schema":
            format_wunsch = (rohdaten["response_format"].get("json_schema") or {}).get("schema")

    return {
        "modell": rohdaten.get("model") or "",
        "messages": messages,
        "options": optionen,
        "tools": rohdaten.get("tools") or [],
        "format": format_wunsch,
        "denken": bool(rohdaten.get("reasoning_effort")) or bool(rohdaten.get("think")),
        "keep_alive": rohdaten.get("keep_alive"),
        "logprobs": bool(rohdaten.get("logprobs")),
        "top_logprobs": rohdaten.get("top_logprobs"),
    }


def _responses_nachrichten(eingabe):
    """'input' der Responses-API in Chat-Nachrichten uebersetzen.

    Erlaubt sind der blosse Text, einzelne Nachrichten-Eintraege und die
    Eintraege, die aus einer vorigen Antwort zurueckkommen: der Werkzeugaufruf
    ('function_call') und sein Ergebnis ('function_call_output'). So bleibt ein
    Werkzeug-Gespraech ueber mehrere Runden zusammenhaengend.
    """
    if isinstance(eingabe, str):
        return [{"role": "user", "content": eingabe}]
    nachrichten = []
    for eintrag in eingabe or []:
        if isinstance(eintrag, str):
            nachrichten.append({"role": "user", "content": eintrag})
            continue
        if not isinstance(eintrag, dict):
            continue
        art = eintrag.get("type")
        if art == "function_call_output":
            ausgabe = eintrag.get("output")
            nachrichten.append({
                "role": "tool",
                "name": eintrag.get("name") or eintrag.get("call_id") or "",
                "content": ausgabe if isinstance(ausgabe, str)
                           else json.dumps(ausgabe, ensure_ascii=False),
            })
            continue
        if art == "function_call":
            nachrichten.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": eintrag.get("call_id") or eintrag.get("id") or "",
                "type": "function",
                "function": {"name": eintrag.get("name") or "",
                             "arguments": eintrag.get("arguments") or "{}"}}]})
            continue
        rolle = eintrag.get("role") or "user"
        inhalt = eintrag.get("content")
        if isinstance(inhalt, list):
            teile = []
            for block in inhalt:
                if isinstance(block, str):
                    teile.append({"type": "text", "text": block})
                elif isinstance(block, dict) and block.get("type") in ("input_text", "output_text", "text"):
                    teile.append({"type": "text", "text": block.get("text", "")})
                elif isinstance(block, dict) and block.get("type") in ("input_image", "image_url"):
                    url = block.get("image_url") or ""
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    teile.append({"type": "image_url", "image_url": {"url": url}})
            inhalt = teile
        nachrichten.append({"role": rolle, "content": inhalt if inhalt is not None else ""})
    return nachrichten


def _responses_werkzeuge(rohe):
    """Werkzeuge der Responses-API (flache Form) in die Chat-Form bringen."""
    werkzeuge = []
    for werkzeug in rohe or []:
        if not isinstance(werkzeug, dict) or werkzeug.get("type") not in (None, "function"):
            continue
        if isinstance(werkzeug.get("function"), dict):
            werkzeuge.append(werkzeug)
        elif werkzeug.get("name"):
            werkzeuge.append({"type": "function", "function": {
                "name": werkzeug["name"],
                "description": werkzeug.get("description") or "",
                "parameters": werkzeug.get("parameters") or {"type": "object", "properties": {}}}})
    return werkzeuge


def _responses_format(text, ersatz):
    """'text.format' der Responses-API in 'response_format' uebersetzen."""
    if isinstance(text, dict) and isinstance(text.get("format"), dict):
        art = text["format"].get("type")
        if art == "json_object":
            return {"type": "json_object"}
        if art == "json_schema":
            return {"type": "json_schema", "json_schema": {
                "name": text["format"].get("name") or "antwort",
                "schema": text["format"].get("schema") or {}}}
    return ersatz


def _intern_aus_responses(rohdaten):
    """Anfrage der Responses-API (/v1/responses) auf die interne Form bringen.

    Die Responses-API sagt dasselbe anders: 'input' statt 'messages',
    'instructions' als Systemauftrag, Werkzeuge flach, 'max_output_tokens'
    statt 'max_tokens'. Uebersetzt wird deshalb auf die OpenAI-Chat-Form -
    ab dort geht die Anfrage den gemeinsamen Weg mit /v1/chat/completions.
    """
    rohdaten = rohdaten or {}
    nachrichten = []
    anweisungen = rohdaten.get("instructions")
    if isinstance(anweisungen, str) and anweisungen.strip():
        nachrichten.append({"role": "system", "content": anweisungen})
    nachrichten.extend(_responses_nachrichten(rohdaten.get("input")))

    umgerechnet = {
        "model": rohdaten.get("model") or "",
        "messages": nachrichten,
        "tools": _responses_werkzeuge(rohdaten.get("tools")),
        "temperature": rohdaten.get("temperature"),
        "top_p": rohdaten.get("top_p"),
        "max_tokens": rohdaten.get("max_output_tokens"),
        "response_format": _responses_format(rohdaten.get("text"), rohdaten.get("response_format")),
    }
    anstrengung = rohdaten.get("reasoning")
    if isinstance(anstrengung, dict) and anstrengung.get("effort"):
        umgerechnet["reasoning_effort"] = anstrengung["effort"]
    return _intern_aus_openai(umgerechnet)


def _openai_payload(intern, zielmodell, stream):
    """Interne Form -> OpenAI-Chat-Payload fuer ein Gleis."""
    messages = []
    for m in intern["messages"]:
        inhalt = m.get("content", "")
        bilder = m.get("images")
        if bilder:
            teile = [{"type": "text", "text": inhalt if isinstance(inhalt, str) else ""}]
            for bild in bilder:
                teile.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bild}"}})
            inhalt = teile
        neu = {"role": m.get("role", "user"), "content": inhalt}
        if m.get("tool_calls"):
            neu["tool_calls"] = m["tool_calls"]
        messages.append(neu)

    payload = {"model": zielmodell, "messages": messages, "stream": bool(stream)}
    opt = intern.get("options") or {}
    if opt.get("temperature") is not None:
        payload["temperature"] = opt["temperature"]
    if opt.get("top_p") is not None:
        payload["top_p"] = opt["top_p"]
    if opt.get("seed") is not None:
        payload["seed"] = opt["seed"]
    if opt.get("frequency_penalty") is not None:
        payload["frequency_penalty"] = opt["frequency_penalty"]
    if opt.get("presence_penalty") is not None:
        payload["presence_penalty"] = opt["presence_penalty"]
    if opt.get("num_predict"):
        payload["max_tokens"] = opt["num_predict"]
    if opt.get("stop"):
        stop = opt["stop"]
        payload["stop"] = stop if isinstance(stop, list) else [stop]
    if intern.get("tools"):
        payload["tools"] = intern["tools"]
    if intern.get("format") == "json":
        payload["response_format"] = {"type": "json_object"}
    elif isinstance(intern.get("format"), dict):
        payload["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "ollama_format", "schema": intern["format"]}}
    return payload


def _kopf(route, eingehender_schluessel, stream):
    kopf = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "User-Agent": KENNUNG,
    }
    schluessel = (route.get("key") or "").strip()
    if schluessel and schluessel.lower() not in ("none", "-", "ollama", "leer"):
        kopf["Authorization"] = schluessel if schluessel.lower().startswith("bearer ") else f"Bearer {schluessel}"
    elif eingehender_schluessel:
        kopf["Authorization"] = eingehender_schluessel
    return kopf


def _platzhalter_argumente(schema):
    """Beispiel-Argumente zu einem Werkzeug-Schema bauen (fuer die Simulation)."""
    if not isinstance(schema, dict):
        return {}
    eigenschaften = schema.get("properties") or {}
    argumente = {}
    for name, feld in eigenschaften.items():
        art = (feld or {}).get("type", "string")
        if art == "string":
            argumente[name] = (feld or {}).get("enum", ["beispiel"])[0]
        elif art in ("integer", "number"):
            argumente[name] = 0
        elif art == "boolean":
            argumente[name] = True
        elif art == "array":
            argumente[name] = []
        else:
            argumente[name] = {}
    return argumente


def _werkzeugname(tools):
    for t in tools or []:
        if isinstance(t, dict):
            funktion = t.get("function") or t
            if funktion.get("name"):
                return funktion["name"], funktion.get("parameters") or {}
    return None, None


def _werkzeug_wunsch(prompt):
    text = (prompt or "").lower()
    return any(w in text for w in ("werkzeug", "tool", "nutze", "benutze", "rufe auf", "führe aus",
                                   "fuehre aus"))


# -------------------------------------------------------------------- Simulation

def _sim_statistik(prompt, antwort, erster_lauf):
    """Plausible Zaehlwerte und Laufzeiten (ns), wie Ollama sie liefert."""
    tempo = max(1.0, K["sim_tokens_pro_s"])
    prompt_tokens = max(1, math.ceil(len(prompt or "") / 4))
    antwort_tokens = max(1, math.ceil(len(antwort or "") / 4))
    return {
        "total_duration": _nanosekunden(prompt_tokens * 0.005 + antwort_tokens / tempo),
        "load_duration": _nanosekunden(0.32 if erster_lauf else 0.02),
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_cached_count": 0,
        "prompt_eval_duration": _nanosekunden(prompt_tokens * 0.005),
        "eval_count": antwort_tokens,
        "eval_duration": _nanosekunden(antwort_tokens / tempo),
    }


def _sim_text(prompt, modell, werkzeug_text=""):
    """Deterministischer, als Simulation gekennzeichneter Antworttext."""
    fingerabdruck = hashlib.sha256((modell + "\x00" + (prompt or "")).encode("utf-8")).hexdigest()
    worte = re.findall(r"\w{4,}", (prompt or "").lower())[:8]
    stichworte = ", ".join(sorted(set(worte))) or "keine"
    kopf = ""
    if K["sim_marker"]:
        kopf = ("[Simulation des Ollama-Bahnhofs - keine echte Modell-Antwort]\n"
                "Dieser Text kommt nicht von einem geladenen Modell, sondern aus der "
                "eingebauten Simulation.\n\n")
    return (
        f"{kopf}Modell {modell} hat laut Weichensteller {len(prompt or '')} Zeichen Eingabe erhalten "
        f"(Anfrage-Kennung {fingerabdruck[:12]}).\n\n"
        "Inhaltlich gibt es hier nichts zu behaupten: ohne Gleis antwortet kein Sprachmodell, "
        "und dieser Weichensteller erfindet keine Fachantworten. Alles, was ein Ollama-Client prueft, "
        "ist trotzdem vorhanden: Haeppchen-Streaming als NDJSON, Denkanteil, Werkzeugaufrufe, "
        "Zeitstempel, Zaehlwerte und Laufzeiten in Nanosekunden, keep_alive und die Modellverwaltung.\n\n"
        f"Erkannte Stichworte aus deiner Nachricht: {stichworte}.\n"
        f"{werkzeug_text}"
    )


def _sim_denk_text(prompt):
    return ("Kurz abwaegen: die Anfrage kommt aus dem Modus Simulation, es liegt also kein Gleis "
            "und kein Sprachmodell vor. Deshalb bleibe ich bei einer kurzen, ehrlichen Antwort "
            "und kennzeichne sie als Simulation.\n")


def _sim_werkzeuge(prompt, modell):
    """Werkzeugaufruf in Ollama-Form, wenn die Simulation ihn erzeugen soll."""
    vorgabe = (K["sim_toolcall"] or "auto").strip()
    if vorgabe.lower() in ("aus", "off", "0", "nein", "no"):
        return []
    name, schema = _werkzeugname(_intern_aktuell.get("tools"))
    if not name:
        return []
    if vorgabe.lower() == "auto" and not _werkzeug_wunsch(prompt):
        return []
    wunsch = name if vorgabe.lower() in ("auto", "an", "on", "1", "ja", "yes") else vorgabe
    if vorgabe.lower() not in ("auto", "an", "on", "1", "ja", "yes"):
        gewaehlt, gewaehlt_schema = _werkzeugname([t for t in _intern_aktuell.get("tools") or []
                                                   if (t.get("function") or t).get("name") == vorgabe])
        if gewaehlt:
            name, schema = gewaehlt, gewaehlt_schema
    return [{"function": {"name": name, "arguments": _platzhalter_argumente(schema)}}]


# Kleiner Merker: die Simulation braucht Zugriff auf die laufende Anfrage
# (Werkzeugliste), ohne alles durchzuschleifen.
_intern_aktuell = {}
_intern_sperre = threading.Lock()


def _simulation_ereignisse(intern, meta, fehlerhinweis=None):
    """Rein simulierter Antwortstrom (neutral)."""
    modell = intern["modell"] or "gemma4:8b"
    prompt = _prompt_aus_messages(intern["messages"])
    bilder = _anzahl_bilder(intern["messages"])

    with _intern_sperre:
        _intern_aktuell.clear()
        _intern_aktuell.update(intern)

    werkzeuge = _sim_werkzeuge(prompt, modell)
    hinweis = ""
    if werkzeuge:
        hinweis = ("\nWerkzeugaufruf (simuliert): "
                   f"{werkzeuge[0]['function']['name']} mit "
                   f"{json.dumps(werkzeuge[0]['function']['arguments'], ensure_ascii=False)}\n")
    if bilder:
        hinweis += f"\nEs wurden {bilder} Bild(er) mitgeschickt; im Simulationsmodus werden sie nicht ausgewertet.\n"
    if fehlerhinweis:
        hinweis += f"\nAlle Gleise sind ausgefallen ({fehlerhinweis}); deshalb simuliert der Weichensteller.\n"

    grund = "tool_calls" if werkzeuge else "stop"
    denken_text = _sim_denk_text(prompt) if intern.get("denken") else ""

    if intern.get("format") == "json":
        antwort = json.dumps({
            "simulation": True,
            "modell": modell,
            "hinweis": ("Der Ollama-Bahnhof hat keine echte Modell-Antwort erzeugt." if K["sim_marker"]
                        else ""),
            "eingabe_zeichen": len(prompt),
            "bilder": bilder,
            "werkzeuge": werkzeuge,
        }, ensure_ascii=False, indent=2)
    else:
        antwort = _sim_text(prompt, modell, hinweis)

    meta["quelle"] = "simulation"
    meta["grund"] = grund
    meta["werkzeuge"] = werkzeuge
    meta["denken"] = denken_text
    meta["stats"] = _sim_statistik(prompt + denken_text, antwort, erster_lauf=meta.get("erster_lauf", True))
    meta["bilder"] = bilder

    pause = max(0, K["sim_delay_ms"]) / 1000.0

    def haeppchen(text, art):
        for stueck in re.findall(r"\S+\s*", text):
            if art == "denken":
                yield {"typ": "denken", "text": stueck}
            else:
                yield {"typ": "text", "text": stueck}
            if pause:
                time.sleep(pause)

    if denken_text:
        yield from haeppchen(denken_text, "denken")
    if intern.get("format") == "json":
        # JSON am Stueck, sonst zerfällt es unterwegs.
        yield {"typ": "text", "text": antwort}
    else:
        yield from haeppchen(antwort, "text")
    if werkzeuge:
        yield {"typ": "werkzeug", "aufrufe": werkzeuge}
    yield {"typ": "ende", "grund": grund}


# ---------------------------------------------------------------------- Durchreichen

def _route_chat_ereignisse(route, intern, stream, meta):
    """Ein Gleis abfragen und dessen Antwort in neutrale Ereignisse uebersetzen."""
    zielmodell = route["modell"] or intern["modell"]
    payload = _openai_payload(intern, zielmodell, stream)
    kopf = _kopf(route, request.headers.get("Authorization", ""), stream)
    antwort = requests.post(route["url"], json=payload, headers=kopf,
                            timeout=route["timeout"], stream=stream)
    meta["quelle"] = f"gleis:{route['name']}"

    if antwort.status_code != 200:
        text = (antwort.text or "")[:200]
        antwort.close()
        raise KeinGleis([f"{route['name']}: HTTP {antwort.status_code} {text}"])

    if not stream:
        try:
            daten = antwort.json()
        except ValueError:
            antwort.close()
            raise KeinGleis([f"{route['name']}: keine JSON-Antwort"])
        antwort.close()
        wahl = (daten.get("choices") or [{}])[0]
        nachricht = wahl.get("message") or {}
        nutzung = daten.get("usage") or {}
        denken = nachricht.get("reasoning_content") or nachricht.get("reasoning") or ""
        if denken:
            yield {"typ": "denken", "text": denken}
        inhalt = nachricht.get("content") or ""
        if inhalt:
            yield {"typ": "text", "text": inhalt}
        aufrufe = []
        for aufruf in nachricht.get("tool_calls") or []:
            funktion = aufruf.get("function") or {}
            argumente = funktion.get("arguments")
            if isinstance(argumente, str):
                try:
                    argumente = json.loads(argumente)
                except ValueError:
                    argumente = {"_roh": argumente}
            aufrufe.append({"function": {"name": funktion.get("name"), "arguments": argumente or {}}})
        if aufrufe:
            yield {"typ": "werkzeug", "aufrufe": aufrufe}
        meta["grund"] = _grund_aus_openai(wahl.get("finish_reason"), bool(aufrufe))
        meta["werkzeuge"] = aufrufe
        meta["denken"] = denken
        laufzeit = _nanosekunden(time.time() - meta.get("start", time.time()))
        meta["stats"] = {
            "total_duration": laufzeit,
            "load_duration": 0,
            "prompt_eval_count": nutzung.get("prompt_tokens", 0),
            "prompt_eval_cached_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": nutzung.get("completion_tokens", 0),
            "eval_duration": laufzeit,
        }
        yield {"typ": "ende", "grund": meta["grund"]}
        return

    # Streaming: SSE-Zeilen in neutrale Ereignisse uebersetzen.
    zaehler = {"text": 0, "denken": 0}
    werkzeuge = {}
    grund = "stop"
    fertig = False
    try:
        for zeile in antwort.iter_lines(decode_unicode=True):
            if not zeile:
                continue
            zeile = zeile.strip()
            if not zeile.startswith("data:"):
                continue
            daten = zeile[5:].strip()
            if daten == "[DONE]":
                fertig = True
                break
            try:
                paket = json.loads(daten)
            except ValueError:
                continue
            if paket.get("usage"):
                nutzung = paket["usage"]
                laufzeit = _nanosekunden(time.time() - meta.get("start", time.time()))
                meta["stats"] = {
                    "total_duration": laufzeit,
                    "load_duration": 0,
                    "prompt_eval_count": nutzung.get("prompt_tokens", 0),
                    "prompt_eval_cached_count": 0,
                    "prompt_eval_duration": 0,
                    "eval_count": nutzung.get("completion_tokens", 0),
                    "eval_duration": laufzeit,
                }
            for wahl in paket.get("choices") or []:
                delta = wahl.get("delta") or {}
                if wahl.get("finish_reason"):
                    grund = _grund_aus_openai(wahl["finish_reason"], grund == "tool_calls")
                denken = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if denken:
                    zaehler["denken"] += 1
                    yield {"typ": "denken", "text": denken}
                inhalt = delta.get("content")
                if inhalt:
                    zaehler["text"] += 1
                    yield {"typ": "text", "text": inhalt}
                for aufruf in delta.get("tool_calls") or []:
                    index = aufruf.get("index", 0)
                    eintrag = werkzeuge.setdefault(index, {"function": {"name": "", "arguments": ""}})
                    funktion = aufruf.get("function") or {}
                    if funktion.get("name"):
                        eintrag["function"]["name"] += funktion["name"]
                    if funktion.get("arguments"):
                        eintrag["function"]["arguments"] += funktion["arguments"]
                    grund = "tool_calls"
    finally:
        antwort.close()

    aufrufe = []
    for index in sorted(werkzeuge):
        eintrag = werkzeuge[index]
        argumente = eintrag["function"]["arguments"]
        try:
            argumente = json.loads(argumente) if argumente else {}
        except ValueError:
            argumente = {"_roh": argumente}
        aufrufe.append({"function": {"name": eintrag["function"]["name"], "arguments": argumente}})
    if aufrufe:
        yield {"typ": "werkzeug", "aufrufe": aufrufe}
    meta.setdefault("werkzeuge", aufrufe)
    meta.setdefault("denken", "")
    meta["grund"] = "tool_calls" if aufrufe else grund
    laufzeit = time.time() - meta.get("start", time.time())
    # Falls das Gleis keine Zaehlwerte liefert: grob schaetzen.
    if "stats" not in meta:
        meta["stats"] = {
            "total_duration": _nanosekunden(laufzeit),
            "load_duration": 0,
            "prompt_eval_count": max(1, math.ceil(len(_prompt_aus_messages(intern["messages"])) / 4)),
            "prompt_eval_cached_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": max(1, zaehler["text"] + zaehler["denken"]),
            "eval_duration": _nanosekunden(laufzeit),
        }
    if not fertig:
        log.info(f"{route['name']}: Strom ohne [DONE] beendet (kommt vor).")
    yield {"typ": "ende", "grund": meta["grund"]}


def _grund_aus_openai(grund, gibt_werkzeuge=False):
    zuordnung = {"stop": "stop", "length": "length", "tool_calls": "tool_calls",
                 "content_filter": "stop", "function_call": "tool_calls"}
    if grund is None:
        return "tool_calls" if gibt_werkzeuge else "stop"
    return zuordnung.get(str(grund), "stop")


def _ereignis_quelle(intern, stream, meta):
    """Generator ueber neutrale Ereignisse; waehlt Gleis oder Simulation.

    Der erste Zug am Generator prueft die Gleise: faellt vor dem ersten
    Haeppchen alles aus, wird KeinGleis geworfen (sauberer 503 statt
    halbem Strom).
    """
    meta.setdefault("start", time.time())
    meta.setdefault("erster_lauf", not GELADEN)

    if K["modus"] == "simulation":
        _simulation_ereignisse_und_meta(intern, meta)
        yield from _simulation_ereignisse(intern, meta)
        return

    if not K["routen"]:
        if K["modus"] == "auto":
            _simulation_ereignisse_und_meta(intern, meta)
            yield from _simulation_ereignisse(intern, meta)
            return
        raise KeinGleis(["keine Route konfiguriert (MODUS=durchreichen)"])

    fehler = []
    for route in _gleis_reihenfolge():
        quelle = _route_chat_ereignisse(route, intern, stream, meta)
        try:
            erstes = next(quelle)
        except StopIteration:
            fehler.append(f"{route['name']}: leere Antwort")
            meta.pop("stats", None)
            continue
        except KeinGleis as ausnahme:
            fehler.extend(ausnahme.fehler)
            meta.pop("stats", None)
            continue
        except requests.exceptions.RequestException as ausnahme:
            fehler.append(f"{route['name']}: {ausnahme}")
            meta.pop("stats", None)
            continue
        except Exception as ausnahme:                       # noqa: BLE001 - Gleis ist fremd
            fehler.append(f"{route['name']}: {type(ausnahme).__name__}: {ausnahme}")
            meta.pop("stats", None)
            continue

        if STICKY.get("name") != route["name"]:
            ZAEHLER["sticky_setzungen"] += 1
        STICKY["name"] = route["name"]
        log.info(f"Gleis {route['name']} antwortet (Modell {route['modell'] or intern['modell']}).")
        yield erstes
        try:
            yield from quelle
        except Exception as ausnahme:                       # noqa: BLE001
            log.error(f"Gleis {route['name']} brach mitten im Strom ab: {ausnahme}")
            fehler.append(f"{route['name']}: {ausnahme}")
            meta["grund"] = "stop"
            yield {"typ": "fehler", "meldung": f"Gleis {route['name']} brach ab: {ausnahme}"}
            yield {"typ": "ende", "grund": "stop"}
        return

    ZAEHLER["fehlschlaege"] += 1
    log.warning(f"Alle Gleise ausgefallen: {fehler}")
    if K["sim_fallback"]:
        meta["quelle"] = "simulation (Notbetrieb)"
        yield from _simulation_ereignisse(intern, meta, fehlerhinweis="; ".join(fehler)[:300])
        return
    raise KeinGleis(fehler or ["keine Route konfiguriert"])


def _simulation_ereignisse_und_meta(intern, meta):
    """Zaehler nachfuehren, wenn die Simulation antwortet."""
    ZAEHLER["simulationen"] += 1


# ------------------------------------------------------------------- Ausgabeformen

def _ndjson_chat_stream(modell, ereignisse, meta):
    """Ollama /api/chat als NDJSON-Strom."""
    for ereignis in ereignisse:
        art = ereignis["typ"]
        if art == "denken":
            yield {"model": modell, "created_at": _zeitstempel(),
                   "message": {"role": "assistant", "content": "", "thinking": ereignis["text"]},
                   "done": False}
        elif art == "text":
            yield {"model": modell, "created_at": _zeitstempel(),
                   "message": {"role": "assistant", "content": ereignis["text"]}, "done": False}
        elif art == "fehler":
            yield {"model": modell, "created_at": _zeitstempel(),
                   "message": {"role": "assistant", "content": ""}, "done": True,
                   "done_reason": "stop", "error": ereignis["meldung"]}
    yield _chat_abschluss(modell, meta)


def _chat_abschluss(modell, meta):
    nachricht = {"role": "assistant", "content": ""}
    if meta.get("denken"):
        nachricht["thinking"] = ""
    if meta.get("werkzeuge"):
        nachricht["tool_calls"] = meta["werkzeuge"]
    fertig = {"model": modell, "created_at": _zeitstempel(), "message": nachricht,
              "done": True, "done_reason": meta.get("grund", "stop")}
    fertig.update(_chat_stats(meta))
    return fertig


def _chat_stats(meta, intern=None):
    stats = dict(meta.get("stats") or {})
    if not stats:
        stats = {"total_duration": _nanosekunden(time.time() - meta.get("start", time.time())),
                 "load_duration": 0,
                 "prompt_eval_count": max(1, math.ceil(len(meta.get("prompt", "")) / 4)),
                 "prompt_eval_cached_count": 0,
                 "prompt_eval_duration": 0,
                 "eval_count": max(1, math.ceil(len(meta.get("text", "")) / 4)),
                 "eval_duration": 0}
    return stats


def _ndjson_generate_stream(modell, ereignisse, meta):
    """Ollama /api/generate als NDJSON-Strom."""
    denken_gesendet = False
    for ereignis in ereignisse:
        art = ereignis["typ"]
        if art == "denken":
            denken_gesendet = True
            yield {"model": modell, "created_at": _zeitstempel(), "response": "",
                   "thinking": ereignis["text"], "done": False}
        elif art == "text":
            yield {"model": modell, "created_at": _zeitstempel(), "response": ereignis["text"],
                   "done": False}
        elif art == "fehler":
            yield {"model": modell, "created_at": _zeitstempel(), "response": "",
                   "done": True, "done_reason": "stop", "error": ereignis["meldung"]}
    abschluss = {"model": modell, "created_at": _zeitstempel(), "response": "",
                 "done": True, "done_reason": meta.get("grund", "stop")}
    abschluss.update(_chat_stats(meta))
    yield abschluss


def _ndjson(zeilen):
    """Wörterbuch-Zeilen als NDJSON ausgeben."""
    for zeile in zeilen:
        yield json.dumps(zeile, ensure_ascii=False) + "\n"


def _sse(pakete, ende=True):
    """SSE-Strom fuer die OpenAI-kompatible Schicht."""
    for paket in pakete:
        yield "data: " + json.dumps(paket, ensure_ascii=False) + "\n\n"
    if ende:
        yield "data: [DONE]\n\n"


def _sse_ereignisse(ereignisse):
    """SSE-Strom mit Ereignisnamen - wie die Responses-API ihn schickt.

    Die Chat-Form kennt nur 'data:'-Zeilen; die Responses-API stellt jedem
    Ereignis zusaetzlich seinen Namen als 'event:'-Zeile voran.
    """
    for ereignis in ereignisse:
        yield "event: " + ereignis["type"] + "\n"
        yield "data: " + json.dumps(ereignis, ensure_ascii=False) + "\n\n"


def _openai_zeit():
    return int(time.time())


def _openai_chat_stream(modell, ereignisse, meta):
    kennung = "chatcmpl-" + uuid.uuid4().hex
    yield {"id": kennung, "object": "chat.completion.chunk", "created": _openai_zeit(),
           "model": modell, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                         "finish_reason": None}]}
    aufrufe_gesendet = False
    for ereignis in ereignisse:
        art = ereignis["typ"]
        if art == "denken":
            yield {"id": kennung, "object": "chat.completion.chunk", "created": _openai_zeit(),
                   "model": modell,
                   "choices": [{"index": 0, "delta": {"reasoning_content": ereignis["text"]},
                                "finish_reason": None}]}
        elif art == "text":
            yield {"id": kennung, "object": "chat.completion.chunk", "created": _openai_zeit(),
                   "model": modell,
                   "choices": [{"index": 0, "delta": {"content": ereignis["text"]},
                                "finish_reason": None}]}
        elif art == "werkzeug" and not aufrufe_gesendet:
            aufrufe_gesendet = True
            delta_aufrufe = []
            for index, aufruf in enumerate(ereignis["aufrufe"]):
                funktion = aufruf["function"]
                delta_aufrufe.append({
                    "index": index, "id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                    "function": {"name": funktion["name"],
                                 "arguments": json.dumps(funktion.get("arguments") or {}, ensure_ascii=False)},
                })
            yield {"id": kennung, "object": "chat.completion.chunk", "created": _openai_zeit(),
                   "model": modell, "choices": [{"index": 0, "delta": {"tool_calls": delta_aufrufe},
                                                 "finish_reason": None}]}
        elif art == "fehler":
            yield {"error": {"message": ereignis["meldung"], "type": "weichensteller_error",
                             "code": "gleis_abgebrochen"}}
    stats = _chat_stats(meta)
    grund = meta.get("grund", "stop")
    letztes = {"id": kennung, "object": "chat.completion.chunk", "created": _openai_zeit(),
               "model": modell, "choices": [{"index": 0, "delta": {}, "finish_reason": grund}],
               "usage": {"prompt_tokens": stats.get("prompt_eval_count", 0),
                         "completion_tokens": stats.get("eval_count", 0),
                         "total_tokens": stats.get("prompt_eval_count", 0) + stats.get("eval_count", 0)}}
    yield letztes


# ------------------------------------------------------- Responses-API (Ausgabe)

def _responses_stueck_text(text, kennung=None):
    """Ausgabestueck 'message' der Responses-API."""
    return {"type": "message", "id": kennung or "msg_" + uuid.uuid4().hex[:24],
            "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def _responses_stueck_denken(text, kennung=None):
    """Ausgabestueck 'reasoning' (Denken) der Responses-API."""
    return {"type": "reasoning", "id": kennung or "rs_" + uuid.uuid4().hex[:24],
            "summary": [{"type": "summary_text", "text": text}]}


def _responses_stueck_aufruf(aufruf):
    """Ausgabestueck 'function_call' der Responses-API."""
    funktion = aufruf.get("function") or {}
    return {"type": "function_call", "id": "fc_" + uuid.uuid4().hex[:24],
            "call_id": "call_" + uuid.uuid4().hex[:24], "status": "completed",
            "name": funktion.get("name") or "",
            "arguments": json.dumps(funktion.get("arguments") or {}, ensure_ascii=False)}


def _responses_geruest(kennung, modell, status, ausgabe, rohdaten):
    """Rumpf einer Antwort der Responses-API (fuer JSON und SSE)."""
    rohdaten = rohdaten or {}
    antwort = {
        "id": kennung,
        "object": "response",
        "created_at": _openai_zeit(),
        "status": status,
        "error": None,
        "incomplete_details": None,
        "model": modell,
        "output": list(ausgabe),
        "parallel_tool_calls": True,
        "store": bool(rohdaten.get("store")),
        "tools": _responses_werkzeuge(rohdaten.get("tools")),
        "usage": None,
    }
    for feld in ("temperature", "top_p", "max_output_tokens"):
        if rohdaten.get(feld) is not None:
            antwort[feld] = rohdaten[feld]
    return antwort


def _responses_verbrauch(meta):
    """Zaehlwerk der Responses-API (input_tokens/output_tokens/...)."""
    stats = _chat_stats(meta)
    eingabe = stats.get("prompt_eval_count", 0)
    ausgabe = stats.get("eval_count", 0)
    return {
        "input_tokens": eingabe,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": ausgabe,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": eingabe + ausgabe,
    }


def _responses_antwort(kennung, modell, text, denken, aufrufe, meta, rohdaten):
    """Fertige Antwort der Responses-API aus dem internen Ereignisstrom."""
    ausgabe = []
    if denken:
        ausgabe.append(_responses_stueck_denken(denken))
    if text or not aufrufe:
        ausgabe.append(_responses_stueck_text(text))
    for aufruf in aufrufe:
        ausgabe.append(_responses_stueck_aufruf(aufruf))
    antwort = _responses_geruest(kennung, modell, "completed", ausgabe, rohdaten)
    antwort["usage"] = _responses_verbrauch(meta)
    return antwort


def _responses_ereignis(stand, art, **felder):
    """Ein Ereignis der Responses-API mit fortlaufender Nummer."""
    stand["nummer"] += 1
    return {"type": art, "sequence_number": stand["nummer"], **felder}


def _responses_stueck_oeffnen(stand, art):
    """Ein neues Ausgabestueck ankuendigen ('added'-Ereignisse)."""
    stand["offen"] = art
    stand["index"] = len(stand["ausgabe"])
    stand["puffer"] = []
    stand["kennung"] = ("rs_" if art == "denken" else "msg_") + uuid.uuid4().hex[:24]
    if art == "denken":
        leer = {"type": "summary_text", "text": ""}
        return [
            _responses_ereignis(stand, "response.output_item.added", output_index=stand["index"],
                                item={"type": "reasoning", "id": stand["kennung"], "summary": [leer]}),
            _responses_ereignis(stand, "response.reasoning_summary_part.added",
                                item_id=stand["kennung"], output_index=stand["index"],
                                summary_index=0, part=leer),
        ]
    return [
        _responses_ereignis(stand, "response.output_item.added", output_index=stand["index"],
                            item={"type": "message", "id": stand["kennung"],
                                  "status": "in_progress", "role": "assistant", "content": []}),
        _responses_ereignis(stand, "response.content_part.added", item_id=stand["kennung"],
                            output_index=stand["index"], content_index=0,
                            part={"type": "output_text", "text": "", "annotations": []}),
    ]


def _responses_stueck_schliessen(stand):
    """Das offene Ausgabestueck abschliessen ('done'-Ereignisse)."""
    art = stand["offen"]
    if art is None:
        return []
    index = stand["index"]
    text = "".join(stand["puffer"])
    if art == "denken":
        stueck = _responses_stueck_denken(text, stand["kennung"])
        ereignisse = [
            _responses_ereignis(stand, "response.reasoning_summary_text.done",
                                item_id=stand["kennung"], output_index=index, summary_index=0,
                                text=text),
            _responses_ereignis(stand, "response.output_item.done", output_index=index, item=stueck),
        ]
    else:
        stueck = _responses_stueck_text(text, stand["kennung"])
        ereignisse = [
            _responses_ereignis(stand, "response.output_text.done", item_id=stand["kennung"],
                                output_index=index, content_index=0, text=text, logprobs=[]),
            _responses_ereignis(stand, "response.content_part.done", item_id=stand["kennung"],
                                output_index=index, content_index=0, part=stueck["content"][0]),
            _responses_ereignis(stand, "response.output_item.done", output_index=index, item=stueck),
        ]
    stand["ausgabe"].append(stueck)
    stand["offen"] = None
    stand["puffer"] = []
    return ereignisse


def _responses_strom(kennung, modell, ereignisse, meta, rohdaten):
    """Ereignisfolge der Responses-API als Strom (SSE).

    Reihenfolge und Namen folgen OpenAI: 'response.created', je Ausgabestueck
    'output_item.added', die Haeppchen ('output_text.delta' bzw.
    'reasoning_summary_text.delta'), die 'done'-Ereignisse und zum Schluss
    'response.completed' mit der fertigen Antwort. Ein '[DONE]' kennt die
    Responses-API nicht.
    """
    stand = {"nummer": 0, "offen": None, "index": 0, "puffer": [], "ausgabe": [],
             "kennung": ""}
    leer = _responses_geruest(kennung, modell, "in_progress", [], rohdaten)
    yield _responses_ereignis(stand, "response.created", response=leer)
    yield _responses_ereignis(stand, "response.in_progress", response=leer)

    denken, aufrufe = [], []
    for ereignis in ereignisse:
        art = ereignis["typ"]
        if art in ("denken", "text"):
            stueck_art = "denken" if art == "denken" else "text"
            if stand["offen"] == "text" and stueck_art == "denken":
                for abschluss in _responses_stueck_schliessen(stand):
                    yield abschluss
            if stand["offen"] != stueck_art:
                for start in _responses_stueck_oeffnen(stand, stueck_art):
                    yield start
            stand["puffer"].append(ereignis["text"])
            if stueck_art == "denken":
                denken.append(ereignis["text"])
                yield _responses_ereignis(stand, "response.reasoning_summary_text.delta",
                                          item_id=stand["kennung"], output_index=stand["index"],
                                          summary_index=0, delta=ereignis["text"])
            else:
                yield _responses_ereignis(stand, "response.output_text.delta",
                                          item_id=stand["kennung"], output_index=stand["index"],
                                          content_index=0, delta=ereignis["text"], logprobs=[])
        elif art == "werkzeug":
            for abschluss in _responses_stueck_schliessen(stand):
                yield abschluss
            for aufruf in ereignis["aufrufe"]:
                stueck = _responses_stueck_aufruf(aufruf)
                index = len(stand["ausgabe"])
                yield _responses_ereignis(stand, "response.output_item.added", output_index=index,
                                          item={**stueck, "arguments": "", "status": "in_progress"})
                yield _responses_ereignis(stand, "response.function_call_arguments.delta",
                                          item_id=stueck["id"], output_index=index,
                                          delta=stueck["arguments"])
                yield _responses_ereignis(stand, "response.function_call_arguments.done",
                                          item_id=stueck["id"], output_index=index,
                                          arguments=stueck["arguments"])
                yield _responses_ereignis(stand, "response.output_item.done", output_index=index,
                                          item=stueck)
                stand["ausgabe"].append(stueck)
                aufrufe.append(aufruf)
        elif art == "fehler":
            yield _responses_ereignis(stand, "error", code="gleis_abgebrochen",
                                      message=ereignis["meldung"])

    for abschluss in _responses_stueck_schliessen(stand):
        yield abschluss
    fertig = _responses_geruest(kennung, modell, "completed", stand["ausgabe"], rohdaten)
    fertig["usage"] = _responses_verbrauch(meta)
    yield _responses_ereignis(stand, "response.completed", response=fertig)


# ----------------------------------------------------------------------- Einbettung

def _sim_vektor(text, dimension):
    """Deterministischer Pseudo-Vektor (kein Modell, nur Pruefsummen-Bits)."""
    dimension = max(8, int(dimension or K["sim_embed_dim"]))
    vektor = []
    block = 0
    while len(vektor) < dimension:
        roh = hashlib.sha256(f"{text}|{block}".encode("utf-8")).digest()
        for i in range(0, len(roh), 4):
            if len(vektor) >= dimension:
                break
            stelle = int.from_bytes(roh[i:i + 4], "big")
            vektor.append(round((stelle / 2 ** 31) - 1.0, 6))
        block += 1
    norm = math.sqrt(sum(w * w for w in vektor)) or 1.0
    return [round(w / norm, 6) for w in vektor]


def _einbettungen_erzeugen(modell, eingaben, dimension):
    vektoren = [_sim_vektor(text, dimension) for text in eingaben]
    tokens = sum(max(1, math.ceil(len(t) / 4)) for t in eingaben)
    return {
        "model": modell,
        "embeddings": vektoren,
        "total_duration": _nanosekunden(tokens * 0.002),
        "load_duration": _nanosekunden(0.25 if not GELADEN else 0.01),
        "prompt_eval_count": tokens,
    }


def _einbettung_erlaubt(modell):
    """Katalogmodelle ohne 'embedding'-Faehigkeit abweisen (wie echtes Ollama).

    Das trifft vor allem das Pseudo-Modell: Ein Chat-Modell liefert keine
    Einbettungen. Ein Gleis mit eigenem Ziel-Modell kann sie aber liefern -
    dann gibt es hier keine Sperre.
    """
    eintrag = _modell_finden(modell)
    if eintrag is None or "embedding" in _faehigkeiten(eintrag):
        return None
    if K["modus"] != "simulation" and any(r["modell"] for r in K["routen"]):
        return None
    return f"model '{modell}' does not support embeddings"


def _einbettungen_durchreichen(modell, eingaben, dimension, meta):
    """Einbettungen von einem Gleis holen (OpenAI-Form)."""
    fehler = []
    for route in K["routen"]:
        payload = {"model": route["modell"] or modell, "input": eingaben if len(eingaben) > 1 else eingaben[0]}
        if dimension:
            payload["dimensions"] = dimension
        try:
            antwort = requests.post(route["embed_url"], json=payload,
                                    headers=_kopf(route, request.headers.get("Authorization", ""), False),
                                    timeout=route["timeout"])
        except requests.exceptions.RequestException as ausnahme:
            fehler.append(f"{route['name']}: {ausnahme}")
            continue
        if antwort.status_code != 200:
            fehler.append(f"{route['name']}: HTTP {antwort.status_code} {antwort.text[:120]}")
            antwort.close()
            continue
        try:
            daten = antwort.json()
        except ValueError:
            fehler.append(f"{route['name']}: keine JSON-Antwort")
            continue
        vektoren = [d.get("embedding") for d in daten.get("data") or []]
        if not vektoren:
            fehler.append(f"{route['name']}: keine Vektoren")
            continue
        meta["quelle"] = f"gleis:{route['name']}"
        ZAEHLER["durchgereicht"] += 1
        return {
            "model": modell,
            "embeddings": vektoren,
            "total_duration": _nanosekunden(time.time() - meta.get("start", time.time())),
            "load_duration": 0,
            "prompt_eval_count": (daten.get("usage") or {}).get("prompt_tokens",
                                                                sum(max(1, len(t) // 4) for t in eingaben)),
        }
    raise KeinGleis(fehler)


# -------------------------------------------------------------------------- Flask

app = Flask(__name__)


def _fehler(status, meldung):
    """Fehlerantwort in der Form der angesprochenen Schnittstelle."""
    if request.path.startswith("/v1/"):
        return jsonify({"error": {"message": meldung, "type": "invalid_request_error",
                                  "code": status}}), status
    return jsonify({"error": meldung}), status


def _erster_zug(quelle):
    """Ersten Zug am Ereignisstrom erzwingen (Gleispruefung vor dem 200er)."""
    erstes = next(quelle)

    def weiter():
        yield erstes
        yield from quelle

    return weiter()


def _json_koerper():
    """JSON-Rumpf lesen - unabhaengig vom Content-Type.

    Echtes Ollama wertet den Rumpf als JSON aus, egal was im Content-Type
    steht. Deshalb wird hier nachgefasst: 'curl -d '{...}'' schickt
    application/x-www-form-urlencoded und muesste sonst scheitern.
    """
    daten = request.get_json(silent=True)
    if daten is None:
        roh = request.get_data(cache=True)
        if roh:
            try:
                daten = json.loads(roh.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                daten = None
    return daten if isinstance(daten, dict) else None


def _katalog_liste():
    _geladen_aufraeumen()
    liste = []
    for eintrag in MODELLE.values():
        liste.append({
            "name": eintrag["name"],
            "model": eintrag["name"],
            "modified_at": eintrag.get("modified_at", _zeitstempel()),
            "size": eintrag.get("size", 0),
            "digest": eintrag.get("digest", _digest(eintrag["name"])),
            "details": _details(eintrag),
        })
    liste.sort(key=lambda m: m["name"])
    return liste


def _fenster(meta, modell):
    """Nur neutrale Kopfzeilen - wie bei einem Proxy ueblich.

    Die Herkunft einer Antwort (Gleis oder Simulation) steht im Protokoll und
    in /health, nicht in erfundenen X-Kopfzeilen.
    """
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }


@app.route("/", methods=["GET", "HEAD"])
def wurzel():
    """Lebenszeichen wie beim echten Ollama."""
    return Response("Ollama is running", mimetype="text/plain")


@app.route("/api/version", methods=["GET"])
def version():
    return jsonify({"version": K["sim_version"]})


@app.route("/health", methods=["GET"])
def health():
    """Eigene Diagnose (Gleise, Modus, Zaehler) - nicht Teil der Ollama-API."""
    _geladen_aufraeumen()
    return jsonify({
        "status": "ok",
        "fassung": FASSUNG,
        "modus": K["modus"],
        "pseudo_modell": K["virtual_model"],
        "simulierte_ollama_version": K["sim_version"],
        "sim_marker": K["sim_marker"],
        "sim_fallback": K["sim_fallback"],
        "gleise": [{"name": r["name"], "gruppe": r.get("gruppe", ""),
                    "url": r["url"], "modell": r["modell"],
                    "timeout": r["timeout"]} for r in K["routen"]],
        "gleis_reihenfolge": K["reihenfolge"],
        "sticky_fallback": K.get("sticky", False),
        "sticky_gleis": STICKY.get("name"),
        "llm_bahnhof_konfiguration": K.get("llm_env"),
        "gleis_start": [r["name"] for r in _gleis_reihenfolge()],
        "katalog": [m["name"] for m in MODELLE.values()],
        "geladen": list(GELADEN),
        "zaehler": dict(ZAEHLER),
        "pfade": {"konfiguration": ENV_PFAD, "zustand": K["state"]},
    })


@app.route("/api/tags", methods=["GET"])
def tags():
    return jsonify({"models": _katalog_liste()})


@app.route("/api/ps", methods=["GET"])
def ps():
    _geladen_aufraeumen()
    return jsonify({"models": list(GELADEN.values())})


@app.route("/api/show", methods=["POST"])
def show():
    daten = _json_koerper()
    if daten is None or not daten.get("model"):
        return _fehler(400, "Feld 'model' fehlt")
    name = daten["model"]
    eintrag = _modell_finden(name)
    if eintrag is None:
        if K["sim_unbekannt"] == "ablehnen":
            return _fehler(404, f"model '{name}' not found")
        eintrag = _modell_eintrag(name)
    familie = eintrag["family"]
    antwort = {
        "license": "simuliert (Ollama-Bahnhof) - kein echtes Modellgewicht",
        "modelfile": (f"# Modelfile des Ollama-Bahnhofs (simuliert)\nFROM {eintrag['name']}\n"
                      f"PARAMETER temperature 0.8\nPARAMETER num_ctx 8192\n"),
        "parameters": "temperature 0.8\ntop_p 0.9\nrepeat_penalty 1.1\nnum_ctx 8192\nstop \"<|im_end|>\"\n",
        "template": ("{{ if .System }}<|system|>\n{{ .System }}\n{{ end }}"
                     "<|user|>\n{{ .Prompt }}\n<|assistant|>\n"),
        "details": _details(eintrag),
        "capabilities": _faehigkeiten(eintrag),
    }
    if daten.get("verbose"):
        antwort["model_info"] = {
            f"{familie}.context_length": 131072,
            f"{familie}.embedding_length": 4096,
            f"{familie}.block_count": 32,
            "general.architecture": familie,
            "general.parameter_count": int(float(re.sub(r"[^0-9.]", "", eintrag["parameter_size"]) or 8)
                                           * 1e9),
            "general.quantization_version": 2,
            "general.name": eintrag["name"],
            "weichensteller.simulation": True,
        }
    return jsonify(antwort)


def _chat_gemeinsam(intern, stream, modell):
    """Gemeinsamer Weg fuer /api/chat und /api/generate."""
    meta = {}
    quelle = _erster_zug(_ereignis_quelle(intern, stream, meta))
    _geladen_melden(modell, intern.get("keep_alive"))
    ZAEHLER["anfragen"] += 1
    fenster = _fenster(meta, modell)
    return quelle, meta, fenster


@app.route("/api/chat", methods=["POST"])
def chat():
    rohdaten = _json_koerper()
    if rohdaten is None:
        return _fehler(400, "JSON-Objekt erwartet")
    if not rohdaten.get("model"):
        return _fehler(400, "Feld 'model' fehlt")
    if not rohdaten.get("messages"):
        return _fehler(400, "Feld 'messages' fehlt")

    intern = _intern_chat(rohdaten)
    modell = rohdaten.get("model")
    eintrag = _modell_finden(modell)
    if eintrag is None and not K["routen"] and K["sim_unbekannt"] == "ablehnen":
        return _fehler(404, f"model '{modell}' not found")
    stream = _ist_stream(rohdaten.get("stream", True))

    try:
        quelle, meta, fenster = _chat_gemeinsam(intern, stream, modell)
    except KeinGleis as ausnahme:
        return _fehler(503, "Kein Gleis lieferte eine Antwort: " + "; ".join(ausnahme.fehler))

    if not stream:
        text, denken, aufrufe = [], [], []
        for ereignis in quelle:
            if ereignis["typ"] == "text":
                text.append(ereignis["text"])
            elif ereignis["typ"] == "denken":
                denken.append(ereignis["text"])
            elif ereignis["typ"] == "werkzeug":
                aufrufe = ereignis["aufrufe"]
            elif ereignis["typ"] == "fehler":
                return _fehler(502, ereignis["meldung"])
        nachricht = {"role": "assistant", "content": "".join(text)}
        if denken:
            nachricht["thinking"] = "".join(denken)
        if aufrufe:
            nachricht["tool_calls"] = aufrufe
        antwort = {"model": modell, "created_at": _zeitstempel(), "message": nachricht,
                   "done": True, "done_reason": meta.get("grund", "stop")}
        antwort.update(_chat_stats(meta))
        return Response(json.dumps(antwort, ensure_ascii=False),
                        content_type="application/json", headers=fenster)

    return Response(_ndjson(_ndjson_chat_stream(modell, quelle, meta)),
                    mimetype="application/x-ndjson", headers=fenster)


@app.route("/api/generate", methods=["POST"])
def generate():
    rohdaten = _json_koerper()
    if rohdaten is None:
        return _fehler(400, "JSON-Objekt erwartet")
    if not rohdaten.get("model"):
        return _fehler(400, "Feld 'model' fehlt")
    prompt = rohdaten.get("prompt")
    if prompt is None:
        return _fehler(400, "Feld 'prompt' fehlt")

    messages = []
    if rohdaten.get("system"):
        messages.append({"role": "system", "content": str(rohdaten["system"])})
    if rohdaten.get("images"):
        messages.append({"role": "user", "content": "", "images": rohdaten["images"]})
    if rohdaten.get("raw"):
        messages.append({"role": "user", "content": str(prompt)})
    else:
        messages.append({"role": "user", "content": str(prompt)})

    intern = _intern_chat({
        "model": rohdaten["model"],
        "messages": messages,
        "options": rohdaten.get("options") or {},
        "format": rohdaten.get("format"),
        "think": rohdaten.get("think", False),
        "keep_alive": rohdaten.get("keep_alive"),
    })
    modell = rohdaten["model"]
    stream = _ist_stream(rohdaten.get("stream", True))

    try:
        quelle, meta, fenster = _chat_gemeinsam(intern, stream, modell)
    except KeinGleis as ausnahme:
        return _fehler(503, "Kein Gleis lieferte eine Antwort: " + "; ".join(ausnahme.fehler))

    if not stream:
        text, denken = [], []
        for ereignis in quelle:
            if ereignis["typ"] == "text":
                text.append(ereignis["text"])
            elif ereignis["typ"] == "denken":
                denken.append(ereignis["text"])
            elif ereignis["typ"] == "fehler":
                return _fehler(502, ereignis["meldung"])
        inhalt = "".join(text)
        antwort = {"model": modell, "created_at": _zeitstempel(), "response": inhalt,
                   "done": True, "done_reason": meta.get("grund", "stop"),
                   "context": _kontext_kennung(_prompt_aus_messages(messages))}
        if denken:
            antwort["thinking"] = "".join(denken)
        antwort.update(_chat_stats(meta))
        return Response(json.dumps(antwort, ensure_ascii=False),
                        content_type="application/json", headers=fenster)

    return Response(_ndjson(_ndjson_generate_stream(modell, quelle, meta)),
                    mimetype="application/x-ndjson", headers=fenster)


def _kontext_kennung(text):
    """Ollama liefert einen Token-Kontext zurueck; hier ein stabiler Ersatz."""
    roh = hashlib.sha256((text or "").encode("utf-8")).digest()
    return [b for b in roh[:32]]


@app.route("/api/embed", methods=["POST"])
def embed():
    daten = _json_koerper()
    if daten is None:
        return _fehler(400, "JSON-Objekt erwartet")
    rohe_eingabe = daten.get("input")
    if rohe_eingabe is None:
        return _fehler(400, "Feld 'input' fehlt")
    eingaben = rohe_eingabe if isinstance(rohe_eingabe, list) else [rohe_eingabe]
    eingaben = [str(t) for t in eingaben]
    modell = daten.get("model") or "embeddinggemma:300m"
    absage = _einbettung_erlaubt(modell)
    if absage:
        return _fehler(400, absage)
    dimension = daten.get("dimensions")
    _geladen_melden(modell, daten.get("keep_alive"))

    if K["routen"] and K["modus"] != "simulation":
        meta = {"start": time.time()}
        try:
            antwort = _einbettungen_durchreichen(modell, eingaben, dimension, meta)
            fenster = _fenster(meta, modell)
            return Response(json.dumps(antwort, ensure_ascii=False),
                            content_type="application/json", headers=fenster)
        except KeinGleis as ausnahme:
            if not K["sim_fallback"]:
                return _fehler(503, "Kein Gleis lieferte Einbettungen: " + "; ".join(ausnahme.fehler))

    meta = {"start": time.time(), "quelle": "simulation"}
    ZAEHLER["simulationen"] += 1
    antwort = _einbettungen_erzeugen(modell, eingaben, dimension)
    return Response(json.dumps(antwort, ensure_ascii=False),
                    content_type="application/json", headers=_fenster(meta, modell))


@app.route("/api/embeddings", methods=["POST"])
def embeddings_alt():
    daten = _json_koerper()
    if daten is None or daten.get("prompt") is None:
        return _fehler(400, "Feld 'prompt' fehlt")
    modell = daten.get("model") or "embeddinggemma:300m"
    absage = _einbettung_erlaubt(modell)
    if absage:
        return _fehler(400, absage)
    _geladen_melden(modell, daten.get("keep_alive"))
    meta = {"start": time.time(), "quelle": "simulation"}
    vektor = _sim_vektor(str(daten["prompt"]), K["sim_embed_dim"])
    return Response(json.dumps({"embedding": vektor}, ensure_ascii=False),
                    content_type="application/json", headers=_fenster(meta, modell))


def _fortschritt(schritte, stream, abschluss):
    """Fortschrittszeilen oder eine einzelne Abschlusszeile."""
    if not stream:
        yield abschluss
        return
    for zeile in schritte:
        yield zeile


@app.route("/api/pull", methods=["POST"])
def pull():
    daten = _json_koerper()
    if daten is None or not daten.get("model"):
        return _fehler(400, "Feld 'model' fehlt")
    name = daten["model"]
    eintrag = _modell_finden(name) or _modell_eintrag(name)
    ZAEHLER["simulationen"] += 1
    gesamt = eintrag["size"]
    schritte = [{"status": "pulling manifest"}]
    rest = gesamt
    stufe = 0
    while rest > 0:
        stufe += 1
        haeppchen = max(1, gesamt // 4)
        geladen = min(gesamt, stufe * haeppchen)
        rest = gesamt - geladen
        schritte.append({"status": f"downloading sha256:{eintrag['digest'][7:19]}",
                         "digest": eintrag["digest"], "total": gesamt, "completed": geladen})
    schritte += [{"status": "verifying sha256 digest"}, {"status": "writing manifest"},
                 {"status": "success"}]

    stream = _ist_stream(daten.get("stream", True))
    with _ZUSTAND_SPERRE:
        MODELLE[eintrag["name"]] = eintrag
        _zustand_speichern()
    log.info(f"Modell '{eintrag['name']}' simuliert gezogen (nicht heruntergeladen).")
    return Response(_ndjson(_fortschritt(schritte, stream, {"status": "success"})),
                    mimetype="application/x-ndjson")


@app.route("/api/push", methods=["POST"])
def push():
    daten = _json_koerper()
    if daten is None or not daten.get("model"):
        return _fehler(400, "Feld 'model' fehlt")
    name = daten["model"]
    eintrag = _modell_finden(name)
    if eintrag is None:
        return _fehler(404, f"model '{name}' not found")
    ZAEHLER["simulationen"] += 1
    schritte = [{"status": "retrieving manifest"}, {"status": "pushing manifest"},
                {"status": "success"}]
    stream = _ist_stream(daten.get("stream", True))
    return Response(_ndjson(_fortschritt(schritte, stream, {"status": "success"})),
                    mimetype="application/x-ndjson")


@app.route("/api/create", methods=["POST"])
def create():
    daten = _json_koerper()
    if daten is None or not daten.get("model"):
        return _fehler(400, "Feld 'model' fehlt")
    name = daten["model"]
    basis = _modell_finden(daten.get("from") or "") or _modell_eintrag(name)
    eintrag = dict(basis)
    eintrag["name"] = name
    eintrag["model"] = name
    eintrag["digest"] = _digest(name)
    eintrag["modified_at"] = _zeitstempel()
    if daten.get("system", "").strip():
        eintrag["capabilities"] = sorted(set(_faehigkeiten(eintrag)) | {"completion"})
    with _ZUSTAND_SPERRE:
        MODELLE[name] = eintrag
        _zustand_speichern()
    ZAEHLER["simulationen"] += 1
    schritte = [{"status": "reading model metadata"},
                {"status": "creating system layer"},
                {"status": "writing manifest"},
                {"status": "success"}]
    stream = _ist_stream(daten.get("stream", True))
    return Response(_ndjson(_fortschritt(schritte, stream, {"status": "success"})),
                    mimetype="application/x-ndjson")


@app.route("/api/copy", methods=["POST"])
def copy():
    daten = _json_koerper()
    if daten is None or not daten.get("source") or not daten.get("destination"):
        return _fehler(400, "Felder 'source' und 'destination' erforderlich")
    quelle = _modell_finden(daten["source"])
    if quelle is None:
        return _fehler(404, f"model '{daten['source']}' not found")
    ziel = dict(quelle)
    ziel["name"] = daten["destination"]
    ziel["model"] = daten["destination"]
    ziel["digest"] = _digest(daten["destination"])
    ziel["modified_at"] = _zeitstempel()
    with _ZUSTAND_SPERRE:
        MODELLE[ziel["name"]] = ziel
        _zustand_speichern()
    return Response("", status=200, mimetype="text/plain")


@app.route("/api/delete", methods=["DELETE"])
def delete():
    daten = _json_koerper() or {}
    name = daten.get("model") or request.args.get("model")
    if not name:
        return _fehler(400, "Feld 'model' fehlt")
    eintrag = _modell_finden(name)
    if eintrag is None:
        return _fehler(404, f"model '{name}' not found")
    with _ZUSTAND_SPERRE:
        MODELLE.pop(eintrag["name"], None)
        GELADEN.pop(eintrag["name"], None)
        _zustand_speichern()
    log.info(f"Modell '{eintrag['name']}' aus dem Weichensteller-Katalog entfernt.")
    return Response("", status=200, mimetype="text/plain")


@app.route("/api/blobs/<path:digest>", methods=["HEAD", "POST"])
def blobs(digest):
    ziel = os.path.join(BLOB_PFAD, digest.replace(":", "_"))
    if request.method == "HEAD":
        return Response("", status=200 if os.path.exists(ziel) else 404)
    daten = request.get_data()
    if not digest.startswith("sha256:"):
        return _fehler(400, "Nur sha256-Prüfsummen werden angenommen")
    erwartet = digest[7:]
    tatsaechlich = hashlib.sha256(daten).hexdigest()
    if tatsaechlich != erwartet:
        return _fehler(400, "digest mismatch")
    if os.path.exists(ziel):
        return _fehler(400, "blob already exists")
    os.makedirs(BLOB_PFAD, exist_ok=True)
    with open(ziel, "wb") as f:
        f.write(daten)
    return Response("", status=201)


# --------------------------------------------------- OpenAI-kompatible Schicht

@app.route("/v1/models", methods=["GET"])
@app.route("/api/v1/models", methods=["GET"])
def v1_models():
    daten = []
    for eintrag in sorted(MODELLE.values(), key=lambda m: m["name"]):
        daten.append({"id": eintrag["name"], "object": "model", "created": 0,
                      "owned_by": "weichensteller"})
    return jsonify({"object": "list", "data": daten})


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/api/v1/chat/completions", methods=["POST"])
def v1_chat():
    rohdaten = _json_koerper()
    if rohdaten is None:
        return _fehler(400, "JSON-Objekt erwartet")
    if not rohdaten.get("model") or not rohdaten.get("messages"):
        return _fehler(400, "Felder 'model' und 'messages' erforderlich")

    intern = _intern_aus_openai(rohdaten)
    modell = rohdaten["model"]
    stream = _ist_stream(rohdaten.get("stream", False))
    meta = {}
    try:
        quelle = _erster_zug(_ereignis_quelle(intern, stream, meta))
    except KeinGleis as ausnahme:
        return _fehler(503, "Kein Gleis lieferte eine Antwort: " + "; ".join(ausnahme.fehler))
    _geladen_melden(modell, rohdaten.get("keep_alive"))
    ZAEHLER["anfragen"] = ZAEHLER.get("anfragen", 0) + 1
    fenster = _fenster(meta, modell)

    if not stream:
        texte, denken, aufrufe = [], [], []
        for ereignis in quelle:
            if ereignis["typ"] == "text":
                texte.append(ereignis["text"])
            elif ereignis["typ"] == "denken":
                denken.append(ereignis["text"])
            elif ereignis["typ"] == "werkzeug":
                aufrufe = ereignis["aufrufe"]
            elif ereignis["typ"] == "fehler":
                return _fehler(502, ereignis["meldung"])
        stats = _chat_stats(meta)
        nachricht = {"role": "assistant", "content": "".join(texte) or None}
        if denken:
            nachricht["reasoning_content"] = "".join(denken)
        if aufrufe:
            nachricht["tool_calls"] = [{
                "id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                "function": {"name": a["function"]["name"],
                             "arguments": json.dumps(a["function"].get("arguments") or {},
                                                     ensure_ascii=False)},
            } for a in aufrufe]
            nachricht["content"] = nachricht["content"] or ""
        antwort = {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": _openai_zeit(),
            "model": modell,
            "choices": [{"index": 0, "message": nachricht,
                         "finish_reason": meta.get("grund", "stop")}],
            "usage": {"prompt_tokens": stats.get("prompt_eval_count", 0),
                      "completion_tokens": stats.get("eval_count", 0),
                      "total_tokens": stats.get("prompt_eval_count", 0) + stats.get("eval_count", 0)},
        }
        return Response(json.dumps(antwort, ensure_ascii=False),
                        content_type="application/json", headers=fenster)

    return Response(_sse(_openai_chat_stream(modell, quelle, meta)),
                    mimetype="text/event-stream", headers=fenster)


@app.route("/v1/responses", methods=["POST"])
@app.route("/api/v1/responses", methods=["POST"])
def v1_responses():
    """Responses-API - die Form, in der codehamr & Co. sprechen."""
    rohdaten = _json_koerper()
    if rohdaten is None:
        return _fehler(400, "JSON-Objekt erwartet")
    if not rohdaten.get("model"):
        return _fehler(400, "Feld 'model' fehlt")
    if rohdaten.get("input") is None:
        return _fehler(400, "Feld 'input' fehlt")

    intern = _intern_aus_responses(rohdaten)
    modell = rohdaten["model"]
    stream = _ist_stream(rohdaten.get("stream", False))
    kennung = "resp_" + uuid.uuid4().hex[:24]
    meta = {}
    try:
        quelle = _erster_zug(_ereignis_quelle(intern, stream, meta))
    except KeinGleis as ausnahme:
        return _fehler(503, "Kein Gleis lieferte eine Antwort: " + "; ".join(ausnahme.fehler))
    _geladen_melden(modell, None)
    ZAEHLER["anfragen"] = ZAEHLER.get("anfragen", 0) + 1
    fenster = _fenster(meta, modell)

    if not stream:
        texte, denken, aufrufe = [], [], []
        for ereignis in quelle:
            if ereignis["typ"] == "text":
                texte.append(ereignis["text"])
            elif ereignis["typ"] == "denken":
                denken.append(ereignis["text"])
            elif ereignis["typ"] == "werkzeug":
                aufrufe = ereignis["aufrufe"]
            elif ereignis["typ"] == "fehler":
                return _fehler(502, ereignis["meldung"])
        antwort = _responses_antwort(kennung, modell, "".join(texte), "".join(denken), aufrufe,
                                     meta, rohdaten)
        return Response(json.dumps(antwort, ensure_ascii=False),
                        content_type="application/json", headers=fenster)

    return Response(_sse_ereignisse(_responses_strom(kennung, modell, quelle, meta, rohdaten)),
                    mimetype="text/event-stream", headers=fenster)


@app.route("/v1/completions", methods=["POST"])
def v1_completions():
    rohdaten = _json_koerper()
    if rohdaten is None:
        return _fehler(400, "JSON-Objekt erwartet")
    if not rohdaten.get("model"):
        return _fehler(400, "Feld 'model' fehlt")
    prompt = rohdaten.get("prompt")
    if prompt is None:
        return _fehler(400, "Feld 'prompt' fehlt")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    intern = _intern_aus_openai({
        "model": rohdaten["model"],
        "messages": [{"role": "user", "content": str(prompt)}],
        "max_tokens": rohdaten.get("max_tokens"),
        "temperature": rohdaten.get("temperature"),
        "stop": rohdaten.get("stop"),
    })
    modell = rohdaten["model"]
    stream = _ist_stream(rohdaten.get("stream", False))
    meta = {}
    try:
        quelle = _erster_zug(_ereignis_quelle(intern, stream, meta))
    except KeinGleis as ausnahme:
        return _fehler(503, "Kein Gleis lieferte eine Antwort: " + "; ".join(ausnahme.fehler))
    _geladen_melden(modell, None)
    ZAEHLER["anfragen"] = ZAEHLER.get("anfragen", 0) + 1
    fenster = _fenster(meta, modell)

    if not stream:
        texte = []
        for ereignis in quelle:
            if ereignis["typ"] == "text":
                texte.append(ereignis["text"])
            elif ereignis["typ"] == "fehler":
                return _fehler(502, ereignis["meldung"])
        stats = _chat_stats(meta)
        antwort = {"id": "cmpl-" + uuid.uuid4().hex, "object": "text_completion",
                   "created": _openai_zeit(), "model": modell,
                   "choices": [{"text": "".join(texte), "index": 0, "logprobs": None,
                                "finish_reason": meta.get("grund", "stop")}],
                   "usage": {"prompt_tokens": stats.get("prompt_eval_count", 0),
                             "completion_tokens": stats.get("eval_count", 0),
                             "total_tokens": stats.get("prompt_eval_count", 0) + stats.get("eval_count", 0)}}
        return Response(json.dumps(antwort, ensure_ascii=False),
                        content_type="application/json", headers=fenster)

    kennung = "cmpl-" + uuid.uuid4().hex

    def strom():
        for ereignis in quelle:
            if ereignis["typ"] == "text":
                yield {"id": kennung, "object": "text_completion", "created": _openai_zeit(),
                       "model": modell,
                       "choices": [{"text": ereignis["text"], "index": 0, "logprobs": None,
                                    "finish_reason": None}]}
        yield {"id": kennung, "object": "text_completion", "created": _openai_zeit(),
               "model": modell,
               "choices": [{"text": "", "index": 0, "logprobs": None,
                            "finish_reason": meta.get("grund", "stop")}]}

    return Response(_sse(strom()), mimetype="text/event-stream", headers=fenster)


@app.route("/v1/embeddings", methods=["POST"])
def v1_embeddings():
    daten = _json_koerper()
    if daten is None:
        return _fehler(400, "JSON-Objekt erwartet")
    rohe_eingabe = daten.get("input")
    if rohe_eingabe is None:
        return _fehler(400, "Feld 'input' fehlt")
    eingaben = rohe_eingabe if isinstance(rohe_eingabe, list) else [rohe_eingabe]
    eingaben = [str(t) for t in eingaben]
    modell = daten.get("model") or "embeddinggemma:300m"
    absage = _einbettung_erlaubt(modell)
    if absage:
        return _fehler(400, absage)
    meta = {"start": time.time(), "quelle": "simulation"}

    if K["routen"] and K["modus"] != "simulation":
        try:
            antwort = _einbettungen_durchreichen(modell, eingaben, daten.get("dimensions"), meta)
        except KeinGleis as ausnahme:
            if not K["sim_fallback"]:
                return _fehler(503, "Kein Gleis lieferte Einbettungen: " + "; ".join(ausnahme.fehler))
            antwort = _einbettungen_erzeugen(modell, eingaben, daten.get("dimensions"))
            meta["quelle"] = "simulation (Notbetrieb)"
    else:
        antwort = _einbettungen_erzeugen(modell, eingaben, daten.get("dimensions"))
        ZAEHLER["simulationen"] += 1

    daten_liste = [{"object": "embedding", "index": i, "embedding": vektor}
                   for i, vektor in enumerate(antwort["embeddings"])]
    return Response(json.dumps({
        "object": "list",
        "data": daten_liste,
        "model": modell,
        "usage": {"prompt_tokens": antwort["prompt_eval_count"],
                  "total_tokens": antwort["prompt_eval_count"]},
    }, ensure_ascii=False), content_type="application/json", headers=_fenster(meta, modell))


# ------------------------------------------------------------------- Fehlerbilder

@app.errorhandler(404)
def nicht_gefunden(fehler):
    return _fehler(404, f"unbekannter Endpunkt: {request.path}")


@app.errorhandler(405)
def nicht_erlaubt(fehler):
    return _fehler(405, f"Methode {request.method} ist fuer {request.path} nicht vorgesehen")


@app.errorhandler(500)
def innen_fehler(fehler):
    log.error(f"Innenfehler: {fehler}")
    return _fehler(500, "Innenfehler im Ollama-Bahnhof")


@app.after_request
def cors(fenster):
    if K["cors"]:
        fenster.headers["Access-Control-Allow-Origin"] = "*"
        fenster.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        fenster.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, HEAD, OPTIONS"
    return fenster


# ------------------------------------------------------------------------ Start

def _pruefen():
    """Konfiguration anzeigen und pruefen (--pruefen)."""
    quelle = ENV_PFAD if os.path.exists(ENV_PFAD) else f"{ENV_PFAD} (nicht vorhanden)"
    print(f"Weichensteller (Ollama + LLM-Bahnhof) {FASSUNG}")
    print(f"Konfiguration      : {quelle}")
    print(f"Modus              : {K['modus']}"
          + ("  (ohne Gleis wird simuliert)" if K["modus"] == "auto" else ""))
    print(f"Pseudo-Modell      : {K['virtual_model']}   "
          f"({','.join(_faehigkeiten(K['virtual']))})"
          + ("   -> wird ins Gleis-Zielmodell uebersetzt" if K["routen"] else "   -> wird simuliert"))
    print(f"Notbetrieb (SIM_FALLBACK): {'an' if K['sim_fallback'] else 'aus'}")
    print(f"Adresse            : {K['host']}:{K['port']}")
    print(f"Simulierte Version : {K['sim_version']}   Marker im Text: {'an' if K['sim_marker'] else 'aus'}")
    print(f"Einbettungsdimension: {K['sim_embed_dim']}   Takt: {K['sim_tokens_pro_s']} tok/s")
    print(f"Zustandsordner     : {K['state']}")
    print(f"Gleise             : {len(K['routen'])}   "
          f"(Reihenfolge: {K['reihenfolge']}, Sticky-Fallback: "
          f"{'an' if K.get('sticky') else 'aus'})")
    llm_pfad = K.get("llm_env") or ""
    vorhanden = "vorhanden" if (llm_pfad and os.path.exists(llm_pfad)) else "nicht vorhanden"
    print(f"LLM-Bahnhof-Konfig.: {llm_pfad} ({vorhanden})")
    for route in K["routen"]:
        schluessel = "eigener Schlüssel" if route["key"] else "kein Schlüssel (durchreichen)"
        print(f"  - {route['name']} [{route.get('gruppe', '-')}] {route['url']}"
              f" | Modell {route['modell'] or '(vom Client)'}"
              f" | Timeout {route['timeout']} | {schluessel}")
    start = [r["name"] for r in _gleis_reihenfolge()]
    if start and K["routen"] and start[0] != K["routen"][0]["name"]:
        print(f"  (Sticky: naechste Anfrage beginnt bei {start[0]})")
    print(f"Katalog            : {len(K['katalog'])} Modell(e) aus der Konfiguration, "
          f"{len(MODELLE)} insgesamt")
    for name in sorted(MODELLE):
        eintrag = MODELLE[name]
        print(f"  - {name:<32} {eintrag['parameter_size']:>6} {eintrag['quantization_level']:<8}"
              f" {eintrag['size'] / 1024 ** 3:5.1f} GB  {','.join(_faehigkeiten(eintrag))}")
    fehler = []
    if K["modus"] == "durchreichen" and not K["routen"]:
        fehler.append("MODUS=durchreichen, aber kein ROUTE_xx konfiguriert")
    if K["sim_delay_ms"] < 0:
        fehler.append("SIM_DELAY_MS ist negativ")
    for f in fehler:
        print(f"FEHLER: {f}")
    return 1 if fehler else 0


def _starten():
    log.info(f"{KENNUNG} startet auf {K['host']}:{K['port']} (Modus {K['modus']}, "
             f"{len(K['routen'])} Gleis(e), {len(MODELLE)} Modell(e) im Katalog)")
    if not K["routen"]:
        log.info("Kein Gleis konfiguriert - der Weichensteller antwortet rein simuliert.")
    app.run(host=K["host"], port=K["port"], debug=False, threaded=True)


def main(argv=None):
    global K, ENV_PFAD
    wahl = argparse.ArgumentParser(description="Ollama-Bahnhof - simuliert einen Ollama-Endpunkt")
    wahl.add_argument("--env", help="Konfigurationsdatei (Vorgabe: .env; sonst WEICHENSTELLER_ENV/DOTENV_PATH)")
    wahl.add_argument("--port", type=int, help="Port (Vorgabe 11434 wie Ollama)")
    wahl.add_argument("--modus", choices=("auto", "simulation", "durchreichen"),
                      help="Betriebsart")
    wahl.add_argument("--pruefen", action="store_true",
                      help="Konfiguration anzeigen und pruefen, nicht starten")
    wahl.add_argument("--fassung", action="store_true", help="Fassung ausgeben")
    args = wahl.parse_args(argv)

    if args.fassung:
        print(KENNUNG)
        return 0
    if args.env:
        ENV_PFAD = args.env
        os.environ["WEICHENSTELLER_ENV"] = args.env
        K = _einstellungen(_env_lesen(args.env))
        _zustand_laden()
    if args.modus:
        K["modus"] = args.modus
    if args.port:
        K["port"] = args.port
    _logger_setzen(K["log"])
    if args.pruefen:
        return _pruefen()
    _starten()
    return 0


if __name__ == "__main__":
    sys.exit(main())
