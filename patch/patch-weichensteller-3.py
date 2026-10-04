#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch 3: Responses-API (/v1/responses) sicherstellen.

Der Endpunkt gehoert zur Ollama-Oberflaeche und steht seit **Ollama-Bahnhof
1.2.0** in `~/ollama-bahnhof/ollama_bahnhof.py` - er kommt mit einer frischen
Kopie also von selbst mit. Dieses Skript ist das Sicherheitsnetz fuer den
anderen Fall: eine Kopie aus einer aelteren Fassung, in der er noch fehlt
(aufgefallen am 21.09.2026: `codehamr` spricht die Responses-API und bekam
vom Weichensteller ein 404).

Deshalb ist es **idempotent**:

  * fehlt der Endpunkt, wird er eingesetzt - Anfrage-Uebersetzer, Ausgabeformen,
    Route und der Eintrag in der Endpunktliste, dazu die zugehoerigen
    Pruefungen in `tests/`,
  * ist er schon da, meldet das Skript es und aendert **nichts**.

Geprueft werden alle Ankerstellen vorher; passt eine nicht, wird nichts
geschrieben. Die README-Stellen des Weichenstellers sind nicht betroffen: diese
Datei ergaenzt nur das Programm und seine Pruefungen.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
MODUL = WURZEL / "weichensteller.py"
TESTS = WURZEL / "tests" / "test_weichensteller.py"

ENDPUNKT_ZEILE = "    POST   /v1/responses          SSE-Stream oder JSON (Responses-API)\n"

# --- 1) Anfrage der Responses-API auf die interne Form bringen ---------------
UEBERSETZER = r'''def _responses_nachrichten(eingabe):
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


'''

# --- 2) SSE mit Ereignisnamen ------------------------------------------------
SSE = r'''def _sse_ereignisse(ereignisse):
    """SSE-Strom mit Ereignisnamen - wie die Responses-API ihn schickt.

    Die Chat-Form kennt nur 'data:'-Zeilen; die Responses-API stellt jedem
    Ereignis zusaetzlich seinen Namen als 'event:'-Zeile voran.
    """
    for ereignis in ereignisse:
        yield "event: " + ereignis["type"] + "\n"
        yield "data: " + json.dumps(ereignis, ensure_ascii=False) + "\n\n"


'''

# --- 3) Ausgabeformen der Responses-API -------------------------------------
AUSGABE = r'''# ------------------------------------------------------- Responses-API (Ausgabe)

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
'''

# --- 4) Route ----------------------------------------------------------------
ROUTE = r'''@app.route("/v1/responses", methods=["POST"])
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


'''

# --- 5) Pruefungen -----------------------------------------------------------
PRUEFUNGEN = r'''    # ------------------------------------------------------------ Responses-API
    def test_v1_responses_ohne_stream(self):
        daten = self.client.post("/v1/responses", json={
            "model": "testmodell:1b", "input": "Hallo"}).get_json()
        self.assertEqual(daten["object"], "response")
        self.assertEqual(daten["status"], "completed")
        stueck = daten["output"][0]
        self.assertEqual(stueck["type"], "message")
        self.assertEqual(stueck["content"][0]["type"], "output_text")
        self.assertIn("Simulation", stueck["content"][0]["text"])
        self.assertGreater(daten["usage"]["total_tokens"], 0)
        self.assertEqual(daten["usage"]["input_tokens"] + daten["usage"]["output_tokens"],
                         daten["usage"]["total_tokens"])

    def test_v1_responses_alias_und_fehler(self):
        self.assertEqual(self.client.post("/api/v1/responses", json={
            "model": "testmodell:1b", "input": "Hallo"}).status_code, 200)
        ohne_eingabe = self.client.post("/v1/responses", json={"model": "testmodell:1b"})
        self.assertEqual(ohne_eingabe.status_code, 400)
        self.assertEqual(ohne_eingabe.get_json()["error"]["type"], "invalid_request_error")

    def test_v1_responses_stream_mit_ereignisnamen(self):
        antwort = self.client.post("/v1/responses", json={
            "model": "testmodell:1b", "input": "Hallo", "stream": True})
        self.assertIn("text/event-stream", antwort.headers["Content-Type"])
        rohtext = antwort.data.decode("utf-8")
        self.assertTrue(rohtext.startswith("event: response.created\ndata: "))
        # Die Responses-API kennt kein [DONE] - das Ende ist 'response.completed'.
        self.assertNotIn("[DONE]", rohtext)
        namen = [z[7:] for z in rohtext.splitlines() if z.startswith("event: ")]
        self.assertEqual(namen[0], "response.created")
        self.assertEqual(namen[-1], "response.completed")
        self.assertIn("response.output_text.delta", namen)
        self.assertIn("response.output_item.done", namen)
        pakete = [json.loads(z[6:]) for z in rohtext.splitlines() if z.startswith("data: ")]
        self.assertEqual(len(pakete), len(namen))
        self.assertEqual([p["sequence_number"] for p in pakete], list(range(1, len(namen) + 1)))
        for paket in pakete:
            self.assertEqual(paket["type"], namen[paket["sequence_number"] - 1])
        text = "".join(p.get("delta", "") for p in pakete
                       if p["type"] == "response.output_text.delta")
        self.assertIn("Simulation", text)
        fertig = pakete[-1]["response"]
        self.assertEqual(fertig["status"], "completed")
        self.assertEqual(fertig["output"][0]["content"][0]["text"], text)
        self.assertGreater(fertig["usage"]["total_tokens"], 0)

    def test_v1_responses_werkzeugaufruf(self):
        daten = self.client.post("/v1/responses", json={
            "model": "testmodell:1b",
            "input": "nutze das Werkzeug bitte",
            "tools": [{"type": "function", "name": "datei_lesen",
                       "parameters": {"type": "object", "properties": {"pfad": {"type": "string"}}}}]}).get_json()
        aufrufe = [s for s in daten["output"] if s["type"] == "function_call"]
        self.assertEqual(len(aufrufe), 1)
        self.assertEqual(aufrufe[0]["name"], "datei_lesen")
        self.assertTrue(aufrufe[0]["call_id"].startswith("call_"))
        self.assertEqual(json.loads(aufrufe[0]["arguments"])["pfad"], "beispiel")

    def test_v1_responses_eingabe_als_teile(self):
        daten = self.client.post("/v1/responses", json={
            "model": "testmodell:1b",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "Was ist auf dem Bild?"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAB"}]}]}).get_json()
        self.assertIn("Bild", daten["output"][0]["content"][0]["text"])

    def test_responses_anfrage_wird_uebersetzt(self):
        intern = wst._intern_aus_responses({
            "model": "testmodell:1b",
            "instructions": "Sei knapp.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hallo"}]}],
            "max_output_tokens": 64,
            "reasoning": {"effort": "low"},
            "tools": [{"type": "function", "name": "datei_lesen",
                       "parameters": {"type": "object"}}],
            "text": {"format": {"type": "json_schema", "name": "antwort",
                                "schema": {"type": "object"}}}})
        self.assertEqual(intern["modell"], "testmodell:1b")
        self.assertEqual(intern["messages"][0], {"role": "system", "content": "Sei knapp."})
        self.assertEqual(intern["messages"][1]["content"], "Hallo")
        self.assertEqual(intern["options"]["num_predict"], 64)
        self.assertTrue(intern["denken"])
        self.assertEqual(intern["tools"][0]["function"]["name"], "datei_lesen")
        self.assertEqual(intern["format"], {"type": "object"})

    def test_responses_werkzeug_gespraech_bleibt_zusammenhaengend(self):
        intern = wst._intern_aus_responses({
            "model": "testmodell:1b",
            "input": [
                {"type": "function_call", "call_id": "call_1", "name": "datei_lesen",
                 "arguments": "{\"pfad\": \"a.txt\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "Inhalt"}]})
        self.assertEqual(intern["messages"][0]["tool_calls"][0]["function"]["name"], "datei_lesen")
        self.assertEqual(intern["messages"][1]["role"], "tool")
        self.assertEqual(intern["messages"][1]["content"], "Inhalt")

'''

# Einfuegestellen im Modul: (Anker, Text davor/danach)
PATCHES = [
    (
        "    GET    /v1/models\n"
        "    POST   /v1/chat/completions   SSE-Stream oder JSON\n"
        "    POST   /v1/completions",
        "    GET    /v1/models\n"
        "    POST   /v1/chat/completions   SSE-Stream oder JSON\n"
        + ENDPUNKT_ZEILE +
        "    POST   /v1/completions",
    ),
    (
        "def _openai_payload(intern, zielmodell, stream):",
        UEBERSETZER + "def _openai_payload(intern, zielmodell, stream):",
    ),
    (
        "def _openai_zeit():\n    return int(time.time())",
        SSE + "def _openai_zeit():\n    return int(time.time())",
    ),
    (
        '@app.route("/v1/completions", methods=["POST"])\ndef v1_completions():',
        ROUTE + '@app.route("/v1/completions", methods=["POST"])\ndef v1_completions():',
    ),
]

# Die Ausgabeformen gehoeren vor die Einbettungen - als Muster, weil die
# Ueberschrift des Ollama-Bahnhofs je Fassung anders viele Striche hat.
EINBETTUNG = re.compile(r"\n\n+# -+ Einbettung\n")
PRUEFLAUF = re.compile(r"    # -+ Prueflauf\n")


def _einsetzen(datei: Path, name: str, schon_da: str, ist_modul: bool) -> int:
    text = datei.read_text(encoding="utf-8")
    if schon_da in text:
        print(f"  ✓ {name}: Responses-API schon vorhanden - nichts zu tun")
        return 0

    if ist_modul:
        fehlend = [anker for anker, _ in PATCHES if anker not in text]
        if not EINBETTUNG.search(text):
            fehlend.append("# ---- Einbettung (Ueberschrift)")
        if fehlend:
            for anker in fehlend:
                print(f"  ✗ {name}: Anker nicht gefunden: {anker.splitlines()[0][:60]}")
            return 1
        for anker, neu in PATCHES:
            text = text.replace(anker, neu, 1)
        text = EINBETTUNG.sub(
            lambda treffer: "\n\n\n" + AUSGABE + "\n\n" + treffer.group(0).lstrip("\n"),
            text, count=1)
    else:
        if not PRUEFLAUF.search(text):
            print("  ✗ tests/: Ankerstelle fuer die Pruefungen nicht gefunden")
            return 1
        text = PRUEFLAUF.sub(lambda treffer: PRUEFUNGEN + treffer.group(0).lstrip("\n"),
                             text, count=1)
    datei.write_text(text, encoding="utf-8")
    print(f"  ✓ {name}: Responses-API eingesetzt")
    return 0


def main() -> int:
    if _einsetzen(MODUL, "weichensteller.py", "def v1_responses(", True):
        return 1
    return _einsetzen(TESTS, "tests/", "def test_v1_responses_ohne_stream(", False)


if __name__ == "__main__":
    sys.exit(main())
