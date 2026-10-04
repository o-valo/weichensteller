#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rauchtest des Weichenstellers - echte Anfragen durch die ganze Kette.

Aufbau (alles lokal, kein Netz):
  Client -> Weichensteller :18320 (komplettes Ollama)
              Gleis 1  llm-ROUTE_01  -> toter Port  (faellt aus)
              Gleis 2  ROUTE_01      -> Stub-Anbieter (antwortet)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request

ORDNER = Path(__file__).resolve().parent
PORT = 18320
STUB_PORT = 18321
TOT = "http://127.0.0.1:1/v1"          # garantiert geschlossener Port

stub = Flask("stub")
ergebnisse: list[tuple[bool, str]] = []


@stub.route("/v1/chat/completions", methods=["POST"])
def stub_chat():
    daten = request.get_json(silent=True) or {}
    modell = daten.get("model", "unbekannt")
    if daten.get("stream"):
        def strom():
            for teil in ["Hallo ", "vom ", "Stub."]:
                paket = {"choices": [{"delta": {"content": teil}, "finish_reason": None}]}
                yield f"data: {json.dumps(paket)}\n\n"
            yield "data: [DONE]\n\n"
        return Response(strom(), mimetype="text/event-stream")
    return jsonify({"choices": [{"message": {"role": "assistant", "content": "Hallo vom Stub."},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 4},
                    "model": modell})


@stub.route("/v1/embeddings", methods=["POST"])
def stub_embed():
    daten = request.get_json(silent=True) or {}
    eingabe = daten.get("input")
    texte = eingabe if isinstance(eingabe, list) else [eingabe or ""]
    return jsonify({"data": [{"embedding": [0.1, 0.2, 0.3], "index": i}
                             for i, _ in enumerate(texte)],
                    "model": daten.get("model", "unbekannt")})


def pruefe(name: str, bedingung: bool, detail: str = "") -> None:
    ergebnisse.append((bedingung, name))
    print(f"  {'✓' if bedingung else '✗'} {name}{('  ' + detail) if detail else ''}")


def hole(pfad: str, methode: str = "GET", **kw):
    return requests.request(methode, f"http://127.0.0.1:{PORT}{pfad}", timeout=20, **kw)


def main() -> int:
    temp = Path(tempfile.mkdtemp(prefix="rauchtest-"))

    # 1) "LLM-Bahnhof": ein totes Gleis - erzwingt den Fallback auf das eigene.
    llm_env = temp / "llm.env"
    llm_env.write_text(f"ROUTE_01={TOT}|none|llm-modell|5s\n", encoding="utf-8")

    # 2) Weichensteller: eigenes Gleis zeigt auf den Stub.
    env = temp / ".env"
    env.write_text(
        "[TEMPLATE]\n"
        "MODUS=durchreichen\n"          # nur echte Gleise - kein Simulations-Rueckfall
        "SIM_FALLBACK=0\n"
        f"WEICHENSTELLER_PORT={PORT}\n"
        "WEICHENSTELLER_HOST=127.0.0.1\n"
        f"LLM_BAHNHOF_ENV={llm_env}\n"
        "GLEIS_REIHENFOLGE=llm-zuerst\n"
        "STICKY_FALLBACK=1\n"
        "VIRTUAL_MODEL=weichensteller:latest\n"
        f"ROUTE_01=http://127.0.0.1:{STUB_PORT}/v1|none|stub-modell|10s\n"
        "MODEL_01=testmodell:1b|1.1B|Q4_K_M|testfamilie|completion,tools,thinking,embedding|0.8\n",
        encoding="utf-8")

    threading.Thread(target=lambda: stub.run(port=STUB_PORT, threaded=True),
                     daemon=True).start()
    time.sleep(0.7)

    # sys.executable statt eines festen "venv/bin/python3": so laeuft der
    # Rauchtest mit genau dem Interpreter, der ihn gestartet hat - und es
    # braucht kein venv im Projektordner.
    prozess = subprocess.Popen(
        [sys.executable, str(ORDNER / "weichensteller.py"), "--env", str(env)],
        cwd=ORDNER, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(50):
            try:
                hole("/api/version")
                break
            except requests.exceptions.RequestException:
                time.sleep(0.2)
        else:
            print("  ✗ Weichensteller ist nicht hochgekommen")
            return 1

        print("\n-- Ollama-Oberfläche nach außen --")
        r = hole("/")
        pruefe("GET / meldet 'Ollama is running'", r.text.strip() == "Ollama is running", r.text.strip())
        r = hole("/api/version")
        pruefe("GET /api/version liefert eine Ollama-Version", "version" in r.json(), str(r.json()))
        r = hole("/api/tags")
        katalog = [m["name"] for m in r.json().get("models", [])]
        pruefe("GET /api/tags listet Pseudo-Modell und Katalog",
               "weichensteller:latest" in katalog and "testmodell:1b" in katalog,
               str(katalog))

        print("\n-- Verbindung: beide Gleis-Quellen, feste Reihenfolge --")
        r = hole("/health").json()
        namen = [g["name"] for g in r["gleise"]]
        gruppen = [g["gruppe"] for g in r["gleise"]]
        pruefe("beide Quellen sind geladen", namen == ["llm-ROUTE_01", "ROUTE_01"], str(namen))
        pruefe("Herkunft ist gekennzeichnet", gruppen == ["llm", "eigene"], str(gruppen))
        pruefe("Reihenfolge llm-zuerst", r["gleis_reihenfolge"] == "llm-zuerst")
        pruefe("vorher beginnt es beim LLM-Bahnhof", r["gleis_start"][0] == "llm-ROUTE_01",
               str(r["gleis_start"]))
        pruefe("kein Sticky-Gleis gemerkt", r["sticky_gleis"] is None)

        print("\n-- Anfrage durch die Kette (erstes Gleis faellt aus) --")
        r = hole("/api/chat", "POST", json={"model": "testmodell:1b",
                                            "messages": [{"role": "user", "content": "Hallo"}],
                                            "stream": False})
        daten = r.json()
        pruefe("POST /api/chat antwortet trotz Ausfall (Fallback)", r.status_code == 200, str(r.status_code))
        pruefe("Antwort in Ollama-Form (message.content)",
               daten.get("message", {}).get("content") == "Hallo vom Stub.", str(daten.get("message")))
        pruefe("Zählwerte wie bei Ollama vorhanden", "eval_count" in daten and "total_duration" in daten)

        print("\n-- Sticky-Fallback --")
        r = hole("/health").json()
        pruefe("erfolgreiches Gleis gemerkt", r["sticky_gleis"] == "ROUTE_01", str(r["sticky_gleis"]))
        pruefe("naechste Anfrage startet dort", r["gleis_start"][0] == "ROUTE_01",
               str(r["gleis_start"]))
        pruefe("Gleis-Zaehler mitgezaehlt", r["zaehler"]["sticky_setzungen"] == 1,
               str(r["zaehler"]["sticky_setzungen"]))
        r = hole("/api/chat", "POST", json={"model": "testmodell:1b",
                                            "messages": [{"role": "user", "content": "Nochmal"}],
                                            "stream": False})
        pruefe("zweite Anfrage antwortet direkt", r.status_code == 200 and
               r.json()["message"]["content"] == "Hallo vom Stub.")

        print("\n-- Die Endpunkte, die der LLM-Bahnhof nicht hat --")
        r = hole("/v1/completions", "POST", json={"model": "testmodell:1b",
                                                  "prompt": "Hallo", "stream": False})
        pruefe("POST /v1/completions antwortet", r.status_code == 200, str(r.status_code))
        r = hole("/api/embed", "POST", json={"model": "testmodell:1b", "input": "Hallo"})
        pruefe("POST /api/embed antwortet", r.status_code == 200 and "embeddings" in r.json(),
               str(r.status_code))
        r = hole("/v1/embeddings", "POST", json={"model": "testmodell:1b", "input": "Hallo"})
        pruefe("POST /v1/embeddings antwortet", r.status_code == 200, str(r.status_code))

        print("\n-- Streaming (NDJSON wie echtes Ollama) --")
        r = hole("/api/chat", "POST", json={"model": "testmodell:1b",
                                            "messages": [{"role": "user", "content": "Hallo"}],
                                            "stream": True}, stream=True)
        zeilen = [json.loads(z) for z in r.iter_lines() if z]
        pruefe("NDJSON-Strom mit done am Ende",
               bool(zeilen) and zeilen[-1].get("done") is True and "message" in zeilen[0],
               f"{len(zeilen)} Zeilen")
    finally:
        prozess.terminate()
        try:
            prozess.wait(timeout=5)
        except subprocess.TimeoutExpired:
            prozess.kill()

    fehler = [n for ok, n in ergebnisse if not ok]
    print(f"\nErgebnis: {len(ergebnisse) - len(fehler)}/{len(ergebnisse)} bestanden")
    if fehler:
        print("Fehlgeschlagen: " + "; ".join(fehler))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
