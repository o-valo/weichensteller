#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruefungen fuer den Ollama-Bahnhof (ohne Netz, ohne echtes Modell).

Start:  ./venv/bin/python3 -m unittest discover -s tests -v

Geprueft wird: der komplette Ollama-Endpunkt (Version, Katalog, ps, show,
chat, generate, embed, Verwaltung, Blobs), die OpenAI-kompatible Schicht
(/v1/...), die Simulation (Marker, Denken, Werkzeuge, JSON-Format) und das
Durchreichen an ein Gleis (gegen einen eigenen Stub-Server im Test).
"""

import http.server
import json
import os
import sys
import tempfile
import threading
import unittest

HIER = os.path.dirname(os.path.abspath(__file__))
WURZEL = os.path.dirname(HIER)
sys.path.insert(0, WURZEL)

# Eigene Konfiguration im Testordner - so bleibt der echte Weichensteller unberuehrt.
TESTORDNER = tempfile.mkdtemp(prefix="weichensteller-test-")
ENV_PFAD = os.path.join(TESTORDNER, ".env")
with open(ENV_PFAD, "w", encoding="utf-8") as f:
    f.write("[TEMPLATE]\n"
            "MODUS=simulation\n"
            "SIM_DELAY_MS=0\n"
            "SIM_MARKER=1\n"
            "SIM_EMBED_DIM=16\n"
            "SIM_TOOLCALL=auto\n"
            "SIM_UNBEKANNT=annehmen\n"
            "VIRTUAL_MODEL=test-pseudo:1b\n"
            "VIRTUAL_MODEL_FAEHIGKEITEN=completion,tools,thinking\n"
            "VIRTUAL_MODEL_FAMILIE=pseudofamilie\n"
            "MODEL_01=testmodell:1b|1.1B|Q4_K_M|testfamilie|completion,tools,thinking|0.8\n"
            "MODEL_02=test-embed:klein|0.1B|F16|testembed|embedding|0.1\n")

os.environ["WEICHENSTELLER_ENV"] = ENV_PFAD
os.environ["WEICHENSTELLER_STATE"] = os.path.join(TESTORDNER, "state")
os.environ["WEICHENSTELLER_LOG"] = os.path.join(TESTORDNER, "weichensteller.log")
# Die Gleisliste des LLM-Bahnhofs NICHT mitlesen: die Pruefungen sollen
# nicht davon abhaengen, was gerade in ~/llm-bahnhof/.env steht.
os.environ["LLM_BAHNHOF_ENV"] = os.path.join(TESTORDNER, "kein-llm-wst.env")

import weichensteller as wst  # noqa: E402  (erst nach dem Setzen der Umgebung)


def ndjson_zeilen(rohdaten):
    return [json.loads(zeile) for zeile in rohdaten.decode("utf-8").splitlines() if zeile.strip()]


class WeichenstellerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        wst.app.config["TESTING"] = True
        cls.client = wst.app.test_client()

    # ------------------------------------------------------------- Grundlagen
    def test_wurzel_meldet_ollama(self):
        antwort = self.client.get("/")
        self.assertEqual(antwort.status_code, 200)
        self.assertIn(b"Ollama is running", antwort.data)

    def test_version(self):
        antwort = self.client.get("/api/version")
        self.assertEqual(antwort.status_code, 200)
        self.assertEqual(antwort.get_json()["version"], "0.34.2")

    def test_health(self):
        daten = self.client.get("/health").get_json()
        self.assertEqual(daten["status"], "ok")
        self.assertEqual(daten["modus"], "simulation")
        self.assertIn("testmodell:1b", daten["katalog"])

    def test_tags_liefert_katalog(self):
        daten = self.client.get("/api/tags").get_json()
        namen = [m["name"] for m in daten["models"]]
        self.assertIn("testmodell:1b", namen)
        eintrag = [m for m in daten["models"] if m["name"] == "testmodell:1b"][0]
        self.assertEqual(eintrag["details"]["parameter_size"], "1.1B")
        self.assertEqual(eintrag["details"]["quantization_level"], "Q4_K_M")
        self.assertTrue(eintrag["digest"].startswith("sha256:"))
        self.assertGreater(eintrag["size"], 0)

    # ----------------------------------------------------------- Pseudo-Modell
    def test_pseudo_modell_steht_im_katalog(self):
        daten = self.client.get("/api/tags").get_json()
        namen = [m["name"] for m in daten["models"]]
        self.assertIn("test-pseudo:1b", namen)
        self.assertIn("test-pseudo:1b",
                      [m["id"] for m in self.client.get("/v1/models").get_json()["data"]])
        self.assertEqual(self.client.get("/health").get_json()["pseudo_modell"], "test-pseudo:1b")

    def test_pseudo_modell_show_und_faehigkeiten(self):
        daten = self.client.post("/api/show", json={"model": "test-pseudo:1b"}).get_json()
        self.assertEqual(daten["details"]["family"], "pseudofamilie")
        self.assertEqual(sorted(daten["capabilities"]), ["completion", "thinking", "tools"])

    def test_pseudo_modell_ohne_tag_wird_gefunden(self):
        """Ollama ergaenzt ':latest' - 'test-pseudo' trifft denselben Eintrag."""
        zeilen = ndjson_zeilen(self.client.post("/api/chat", json={
            "model": "test-pseudo",
            "messages": [{"role": "user", "content": "Hallo Pseudo"}]}).data)
        self.assertTrue(zeilen[-1]["done"])
        self.assertEqual(zeilen[-1]["model"], "test-pseudo")
        text = "".join(z["message"].get("content", "") for z in zeilen)
        self.assertIn("Simulation", text)

    def test_pseudo_modell_weist_einbettungen_ab(self):
        antwort = self.client.post("/api/embed", json={"model": "test-pseudo:1b", "input": "Hallo"})
        self.assertEqual(antwort.status_code, 400)
        self.assertIn("does not support embeddings", antwort.get_json()["error"])
        alt = self.client.post("/api/embeddings", json={"model": "test-pseudo:1b", "prompt": "Hallo"})
        self.assertEqual(alt.status_code, 400)
        self.assertEqual(self.client.post("/v1/embeddings", json={
            "model": "test-pseudo:1b", "input": "Hallo"}).status_code, 400)

    def test_show_mit_faehigkeiten(self):
        antwort = self.client.post("/api/show", json={"model": "testmodell:1b", "verbose": True})
        daten = antwort.get_json()
        self.assertEqual(antwort.status_code, 200)
        self.assertIn("completion", daten["capabilities"])
        self.assertIn("thinking", daten["capabilities"])
        self.assertIn("template", daten)
        self.assertIn("modelfile", daten)
        self.assertIn("testfamilie.context_length", daten["model_info"])

    def test_unbekannter_endpunkt(self):
        antwort = self.client.get("/api/gibtsnicht")
        self.assertEqual(antwort.status_code, 404)
        self.assertIn("error", antwort.get_json())

    # ------------------------------------------------------------------ Chat
    def test_chat_stream_ist_ndjson_mit_abschluss(self):
        antwort = self.client.post("/api/chat", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "Hallo Weichensteller"}],
        })
        self.assertEqual(antwort.status_code, 200)
        self.assertIn("application/x-ndjson", antwort.headers["Content-Type"])
        # Wie ein Proxy: keine erfundenen X-Kopfzeilen ueber die Herkunft.
        self.assertNotIn("X-Ollama-Bahnhof", antwort.headers)
        zeilen = ndjson_zeilen(antwort.data)
        self.assertGreater(len(zeilen), 3)
        self.assertFalse(zeilen[0]["done"])
        letzte = zeilen[-1]
        self.assertTrue(letzte["done"])
        self.assertEqual(letzte["model"], "testmodell:1b")
        self.assertEqual(letzte["done_reason"], "stop")
        for feld in ("total_duration", "load_duration", "prompt_eval_count", "eval_count", "eval_duration"):
            self.assertIn(feld, letzte)
        self.assertGreater(letzte["eval_count"], 0)
        text = "".join(z["message"]["content"] for z in zeilen[:-1])
        self.assertIn("Simulation", text)

    def test_chat_ohne_stream(self):
        antwort = self.client.post("/api/chat", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "Kurz bitte"}],
            "stream": False,
        })
        daten = antwort.get_json()
        self.assertEqual(daten["message"]["role"], "assistant")
        self.assertIn("Simulation", daten["message"]["content"])
        self.assertTrue(daten["done"])
        self.assertGreater(daten["prompt_eval_count"], 0)

    def test_chat_denken(self):
        antwort = self.client.post("/api/chat", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "Denk nach"}],
            "think": True,
        })
        zeilen = ndjson_zeilen(antwort.data)
        denken = [z for z in zeilen if z.get("message", {}).get("thinking")]
        self.assertTrue(denken, "Es sollte mindestens ein Denk-Haeppchen geben")

    def test_chat_werkzeugaufruf(self):
        antwort = self.client.post("/api/chat", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "Nutze das Werkzeug und lies die Datei"}],
            "tools": [{"type": "function", "function": {
                "name": "datei_lesen",
                "description": "liest eine Datei",
                "parameters": {"type": "object",
                               "properties": {"pfad": {"type": "string"}},
                               "required": ["pfad"]}}}],
        })
        zeilen = ndjson_zeilen(antwort.data)
        letzte = zeilen[-1]
        self.assertEqual(letzte["done_reason"], "tool_calls")
        self.assertEqual(letzte["message"]["tool_calls"][0]["function"]["name"], "datei_lesen")
        self.assertEqual(letzte["message"]["tool_calls"][0]["function"]["arguments"]["pfad"], "beispiel")

    def test_chat_ohne_werkzeugwunsch_ruft_nichts_auf(self):
        antwort = self.client.post("/api/chat", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "Erzaehl mir etwas über Zuege"}],
            "tools": [{"type": "function", "function": {"name": "datei_lesen", "parameters": {}}}],
            "stream": False,
        })
        self.assertNotIn("tool_calls", antwort.get_json()["message"])

    def test_chat_format_json(self):
        antwort = self.client.post("/api/chat", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "Gib JSON zurueck"}],
            "format": "json",
            "stream": False,
        })
        inhalt = antwort.get_json()["message"]["content"]
        daten = json.loads(inhalt)
        self.assertTrue(daten["simulation"])
        self.assertEqual(daten["modell"], "testmodell:1b")

    def test_chat_merkt_modell_als_geladen(self):
        self.client.post("/api/chat", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "bleib geladen"}],
            "keep_alive": "30m",
            "stream": False,
        })
        daten = self.client.get("/api/ps").get_json()
        namen = [m["name"] for m in daten["models"]]
        self.assertIn("testmodell:1b", namen)
        eintrag = [m for m in daten["models"] if m["name"] == "testmodell:1b"][0]
        self.assertIn("expires_at", eintrag)
        self.assertIn("size_vram", eintrag)

    def test_rumpf_ohne_json_content_type(self):
        """curl -d schickt keinen JSON-Content-Type - Ollama nimmt es trotzdem."""
        antwort = self.client.post("/api/chat", data=json.dumps({
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "Hallo"}],
            "stream": False}), content_type="application/x-www-form-urlencoded")
        self.assertEqual(antwort.status_code, 200)
        self.assertIn("Simulation", antwort.get_json()["message"]["content"])

    def test_chat_ohne_modell_ist_ein_fehler(self):
        antwort = self.client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(antwort.status_code, 400)

    # -------------------------------------------------------------- Generate
    def test_generate_stream_und_kontext(self):
        strom = self.client.post("/api/generate", json={
            "model": "testmodell:1b", "prompt": "Schreib einen Satz", "system": "Sei kurz"})
        zeilen = ndjson_zeilen(strom.data)
        self.assertTrue(zeilen[-1]["done"])
        self.assertTrue(all("response" in z for z in zeilen))

        einfach = self.client.post("/api/generate", json={
            "model": "testmodell:1b", "prompt": "Schreib einen Satz", "stream": False}).get_json()
        self.assertIn("Simulation", einfach["response"])
        self.assertTrue(einfach["context"])
        self.assertTrue(all(isinstance(w, int) for w in einfach["context"]))

    def test_generate_ohne_prompt(self):
        self.assertEqual(self.client.post("/api/generate", json={"model": "x"}).status_code, 400)

    # ------------------------------------------------------------ Einbettung
    def test_embed_liefert_vektoren(self):
        daten = self.client.post("/api/embed", json={
            "model": "test-embed:klein", "input": ["erster Text", "zweiter Text"]}).get_json()
        self.assertEqual(len(daten["embeddings"]), 2)
        self.assertEqual(len(daten["embeddings"][0]), 16)
        self.assertEqual(daten["model"], "test-embed:klein")
        self.assertGreater(daten["prompt_eval_count"], 0)

    def test_embed_ist_deterministisch(self):
        eins = self.client.post("/api/embed", json={"model": "m", "input": "gleicher Text"}).get_json()
        zwei = self.client.post("/api/embed", json={"model": "m", "input": "gleicher Text"}).get_json()
        self.assertEqual(eins["embeddings"], zwei["embeddings"])

    def test_embeddings_altform(self):
        daten = self.client.post("/api/embeddings", json={
            "model": "test-embed:klein", "prompt": "alt"}).get_json()
        self.assertEqual(len(daten["embedding"]), 16)

    # ---------------------------------------------------------- Verwaltung
    def test_pull_legt_modell_in_den_katalog(self):
        strom = self.client.post("/api/pull", json={"model": "neu:7b"})
        zeilen = ndjson_zeilen(strom.data)
        self.assertEqual(zeilen[-1]["status"], "success")
        self.assertTrue(any(s.get("digest") for s in zeilen))
        namen = [m["name"] for m in self.client.get("/api/tags").get_json()["models"]]
        self.assertIn("neu:7b", namen)
        # Aufraeumen, damit andere Pruefungen sauber bleiben.
        self.client.delete("/api/delete", json={"model": "neu:7b"})

    def test_create_copy_delete(self):
        anlegen = ndjson_zeilen(self.client.post("/api/create", json={
            "model": "eigen:klein", "from": "testmodell:1b", "system": "Sei knapp"}).data)
        self.assertEqual(anlegen[-1]["status"], "success")
        self.assertEqual(self.client.post("/api/copy", json={
            "source": "eigen:klein", "destination": "kopie:klein"}).status_code, 200)
        namen = [m["name"] for m in self.client.get("/api/tags").get_json()["models"]]
        self.assertIn("kopie:klein", namen)
        self.assertEqual(self.client.delete("/api/delete", json={"model": "kopie:klein"}).status_code, 200)
        self.assertEqual(self.client.delete("/api/delete", json={"model": "eigen:klein"}).status_code, 200)
        self.assertEqual(self.client.delete("/api/delete", json={"model": "kopie:klein"}).status_code, 404)

    def test_blobs_mit_pruefsummenpruefung(self):
        nutzlast = b"beispielinhalt fuer einen Blob"
        import hashlib
        digest = "sha256:" + hashlib.sha256(nutzlast).hexdigest()
        self.assertEqual(self.client.head(f"/api/blobs/{digest}").status_code, 404)
        self.assertEqual(self.client.post(f"/api/blobs/{digest}", data=nutzlast).status_code, 201)
        self.assertEqual(self.client.head(f"/api/blobs/{digest}").status_code, 200)
        self.assertEqual(self.client.post(f"/api/blobs/{digest}", data=nutzlast).status_code, 400)
        falsch = "sha256:" + "0" * 64
        self.assertEqual(self.client.post(f"/api/blobs/{falsch}", data=nutzlast).status_code, 400)

    def test_push(self):
        zeilen = ndjson_zeilen(self.client.post("/api/push", json={"model": "testmodell:1b"}).data)
        self.assertEqual(zeilen[-1]["status"], "success")
        self.assertEqual(self.client.post("/api/push", json={"model": "weg:1b"}).status_code, 404)

    # --------------------------------------------------- OpenAI-kompatibel
    def test_v1_models(self):
        daten = self.client.get("/v1/models").get_json()
        self.assertEqual(daten["object"], "list")
        self.assertIn("testmodell:1b", [m["id"] for m in daten["data"]])

    def test_v1_chat_ohne_stream(self):
        antwort = self.client.post("/v1/chat/completions", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "Hallo"}]})
        daten = antwort.get_json()
        self.assertEqual(daten["object"], "chat.completion")
        self.assertEqual(daten["choices"][0]["finish_reason"], "stop")
        self.assertIn("Simulation", daten["choices"][0]["message"]["content"])
        self.assertGreater(daten["usage"]["total_tokens"], 0)

    def test_v1_chat_stream_ist_sse(self):
        antwort = self.client.post("/v1/chat/completions", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "Hallo"}],
            "stream": True})
        self.assertIn("text/event-stream", antwort.headers["Content-Type"])
        rohtext = antwort.data.decode("utf-8")
        self.assertIn("data: ", rohtext)
        self.assertTrue(rohtext.rstrip().endswith("data: [DONE]"))
        pakete = [json.loads(z[5:]) for z in rohtext.splitlines()
                  if z.startswith("data: ") and z[6:] != "[DONE]"]
        self.assertEqual(pakete[0]["choices"][0]["delta"]["role"], "assistant")
        text = "".join(p["choices"][0]["delta"].get("content", "") for p in pakete)
        self.assertIn("Simulation", text)
        self.assertEqual(pakete[-1]["choices"][0]["finish_reason"], "stop")

    def test_v1_chat_werkzeuge_als_json_string(self):
        antwort = self.client.post("/v1/chat/completions", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "nutze das Werkzeug bitte"}],
            "tools": [{"type": "function", "function": {
                "name": "datei_lesen", "parameters": {"type": "object", "properties": {"pfad": {"type": "string"}}}}}]})
        daten = antwort.get_json()
        aufruf = daten["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(aufruf["function"]["name"], "datei_lesen")
        self.assertEqual(json.loads(aufruf["function"]["arguments"])["pfad"], "beispiel")
        self.assertEqual(daten["choices"][0]["finish_reason"], "tool_calls")

    def test_v1_completions(self):
        daten = self.client.post("/v1/completions", json={
            "model": "testmodell:1b", "prompt": "Weiter bitte"}).get_json()
        self.assertEqual(daten["object"], "text_completion")
        self.assertIn("Simulation", daten["choices"][0]["text"])

    def test_v1_embeddings(self):
        daten = self.client.post("/v1/embeddings", json={
            "model": "test-embed:klein", "input": ["a", "b"]}).get_json()
        self.assertEqual(daten["object"], "list")
        self.assertEqual(len(daten["data"]), 2)
        self.assertEqual(len(daten["data"][0]["embedding"]), 16)
        self.assertEqual(daten["data"][1]["index"], 1)

    def test_v1_bild_im_inhalt(self):
        daten = self.client.post("/v1/chat/completions", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Was ist auf dem Bild?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAB"}}]}]}).get_json()
        self.assertIn("Bild", daten["choices"][0]["message"]["content"])

    # ------------------------------------------------------------ Responses-API
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

    # ------------------------------------------------------------ Prueflauf
    def test_pruefen_laeuft_durch(self):
        self.assertEqual(wst._pruefen(), 0)# ------------------------------------------------------- Durchreichen (Gleis)

class StubGleis(http.server.BaseHTTPRequestHandler):
    """Minimaler OpenAI-kompatibler Endpunkt fuer die Durchreichpruefung."""

    erwischt = []

    def do_POST(self):                                     # noqa: N802 - Vorgabe
        laenge = int(self.headers.get("Content-Length", 0))
        try:
            anfrage = json.loads(self.rfile.read(laenge) or b"{}")
        except ValueError:
            anfrage = {}
        StubGleis.erwischt.append(anfrage)

        if self.path.endswith("/embeddings"):
            nutzlast = {"data": [{"embedding": [0.5, 0.25, 0.125]}],
                        "usage": {"prompt_tokens": 4, "total_tokens": 4}}
        elif anfrage.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for stueck in ("Vor", "lauf", "ig?"):
                paket = {"choices": [{"index": 0, "delta": {"content": stueck}, "finish_reason": None}]}
                self.wfile.write(b"data: " + json.dumps(paket).encode() + b"\n\n")
            self.wfile.write(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                             b'"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
            return
        else:
            nutzlast = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "Stub-Antwort."},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 4}}
        rohdaten = json.dumps(nutzlast).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(rohdaten)))
        self.end_headers()
        self.wfile.write(rohdaten)

    def log_message(self, *args):
        pass


class DurchreichTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StubGleis)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.client = wst.app.test_client()
        # Modus und Gleis umstellen, wie es eine echte Konfiguration taete.
        cls.alte_routen = wst.K["routen"]
        cls.alter_modus = wst.K["modus"]
        wst.K["modus"] = "durchreichen"
        wst.K["routen"] = [{"name": "ROUTE_01",
                                "url": f"http://127.0.0.1:{cls.port}/v1/chat/completions",
                                "embed_url": f"http://127.0.0.1:{cls.port}/v1/embeddings",
                                "key": "none", "modell": "ziel-modell", "timeout": 30.0}]

    @classmethod
    def tearDownClass(cls):
        wst.K["routen"] = cls.alte_routen
        wst.K["modus"] = cls.alter_modus
        cls.server.shutdown()

    def test_chat_stream_kommt_vom_gleis(self):
        StubGleis.erwischt.clear()
        antwort = self.client.post("/api/chat", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "Hallo"}],
            "options": {"temperature": 0.4, "num_predict": 64},
        })
        self.assertEqual(antwort.status_code, 200)
        self.assertNotIn("X-Ollama-Bahnhof", antwort.headers)
        zeilen = ndjson_zeilen(antwort.data)
        text = "".join(z["message"].get("content", "") for z in zeilen)
        self.assertIn("Vorlauf", text)
        self.assertIn("ig?", text)
        self.assertTrue(zeilen[-1]["done"])
        # Der Zielname des Gleises ging raus, das Client-Modell bleibt in der Antwort.
        self.assertEqual(zeilen[-1]["model"], "testmodell:1b")
        self.assertEqual(StubGleis.erwischt[0]["model"], "ziel-modell")
        self.assertEqual(StubGleis.erwischt[0]["temperature"], 0.4)
        self.assertEqual(StubGleis.erwischt[0]["max_tokens"], 64)

    def test_chat_ohne_stream_kommt_vom_gleis(self):
        antwort = self.client.post("/api/chat", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "Hallo"}],
            "stream": False,
        })
        daten = antwort.get_json()
        self.assertEqual(daten["message"]["content"], "Stub-Antwort.")
        self.assertEqual(daten["prompt_eval_count"], 5)
        self.assertEqual(daten["eval_count"], 4)

    def test_v1_chat_stream_kommt_vom_gleis(self):
        antwort = self.client.post("/v1/chat/completions", json={
            "model": "testmodell:1b",
            "messages": [{"role": "user", "content": "Hallo"}],
            "stream": True,
        })
        rohtext = antwort.data.decode("utf-8")
        self.assertIn("Vor", rohtext)
        self.assertTrue(rohtext.rstrip().endswith("data: [DONE]"))

    def test_embed_kommt_vom_gleis(self):
        daten = self.client.post("/api/embed", json={
            "model": "test-embed:klein", "input": "Hallo"}).get_json()
        self.assertEqual(daten["embeddings"], [[0.5, 0.25, 0.125]])

    def test_pseudo_modell_geht_als_ziel_modell_raus(self):
        """Wie beim llm-bahnhof: der Client nennt das Pseudo-Modell, das Gleis
        bekommt sein eigenes Ziel-Modell zu sehen."""
        StubGleis.erwischt.clear()
        zeilen = ndjson_zeilen(self.client.post("/api/chat", json={
            "model": "test-pseudo:1b",
            "messages": [{"role": "user", "content": "Hallo"}]}).data)
        self.assertEqual(StubGleis.erwischt[0]["model"], "ziel-modell")
        self.assertEqual(zeilen[-1]["model"], "test-pseudo:1b")
        self.assertTrue(zeilen[-1]["done"])

    def test_pseudo_modell_im_gleisbetrieb_nicht_simuliert(self):
        zeilen = ndjson_zeilen(self.client.post("/api/chat", json={
            "model": "test-pseudo",
            "messages": [{"role": "user", "content": "Hallo"}]}).data)
        text = "".join(z["message"].get("content", "") for z in zeilen)
        self.assertIn("Vor", text)
        self.assertNotIn("Simulation", text)

    def test_ausfall_ohne_notbetrieb_meldet_fehler(self):
        wst.K["routen"] = [dict(wst.K["routen"][0], url="http://127.0.0.1:1/v1/chat/completions")]
        try:
            antwort = self.client.post("/api/chat", json={
                "model": "testmodell:1b",
                "messages": [{"role": "user", "content": "Hallo"}]})
            self.assertEqual(antwort.status_code, 503)
            self.assertIn("error", antwort.get_json())
        finally:
            wst.K["routen"] = [dict(wst.K["routen"][0],
                                        url=f"http://127.0.0.1:{self.port}/v1/chat/completions")]

    def test_ausfall_mit_notbetrieb_simuliert(self):
        wst.K["sim_fallback"] = True
        wst.K["routen"] = [dict(wst.K["routen"][0], url="http://127.0.0.1:1/v1/chat/completions")]
        try:
            zeilen = ndjson_zeilen(self.client.post("/api/chat", json={
                "model": "testmodell:1b",
                "messages": [{"role": "user", "content": "Hallo"}]}).data)
            text = "".join(z["message"].get("content", "") for z in zeilen)
            self.assertIn("Simulation", text)
            self.assertIn("ausgefallen", text)
        finally:
            wst.K["sim_fallback"] = False
            wst.K["routen"] = [dict(wst.K["routen"][0],
                                        url=f"http://127.0.0.1:{self.port}/v1/chat/completions")]


# ------------------------------------------------------------ .env lesen

class EnvLesenTest(unittest.TestCase):
    """Die .env folgt dem Format des llm-bahnhofs."""

    @classmethod
    def setUpClass(cls):
        cls.pfad = os.path.join(TESTORDNER, "vorlage.env")
        with open(cls.pfad, "w", encoding="utf-8") as f:
            f.write("[TEMPLATE]\n"
                    "# Kommentar\n"
                    "; auch ein Kommentar\n"
                    "\n"
                    "VIRTUAL_MODEL=llm-bahnhof\n"
                    'export ROUTE_01="http://10.7.0.124:8000/v1|none|glm-4.7-flash|15m"\n')

    def test_toleriert_kopfzeile_export_und_anfuehrungszeichen(self):
        gelesen = wst._env_lesen(self.pfad)
        self.assertEqual(gelesen["VIRTUAL_MODEL"], "llm-bahnhof")
        self.assertEqual(gelesen["ROUTE_01"],
                         "http://10.7.0.124:8000/v1|none|glm-4.7-flash|15m")

    def test_fehlende_datei_ist_kein_fehler(self):
        self.assertEqual(wst._env_lesen(os.path.join(TESTORDNER, "gibtsnicht.env")), {})

    def test_routen_aus_der_env(self):
        routen = wst._routen_lesen(wst._env_lesen(self.pfad))
        self.assertEqual(len(routen), 1)
        self.assertEqual(routen[0]["url"], "http://10.7.0.124:8000/v1/chat/completions")
        self.assertEqual(routen[0]["modell"], "glm-4.7-flash")
        self.assertEqual(routen[0]["timeout"], 900.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
