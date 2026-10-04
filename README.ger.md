<p align="center">
  <img src="weichensteller.jpg" alt="Weichensteller-Banner: in einem Rechenzentrum verläuft ein Eisenbahngleis zwischen Server-Schränken, während ein Roboter eine Weiche umstellt." width="100%">
</p>



# 🚉 Weichensteller (Ollama + LLM-Bahnhof)

**[English](README.md) | Deutsch**

**Ein Programm, zwei Bahnhöfe.** Nach außen **simuliert** dieser Weichensteller
eine **vollständige Ollama-API** – jedes Werkzeug, das „Ollama kann“, redet mit
ihm und merkt keinen Unterschied. Die Anfragen leitet er aber an **beliebige
Endpunkte** weiter, die er *Gleise* nennt: das kann ein **echter
Ollama-Endpunkt** sein, ein **OpenAI-kompatibler** Endpunkt, ein Free-Tier-
Anbieter oder ein ganzer Router. Fällt eines aus, springt er zum nächsten.

Im Alltag dient er dabei vor allem als **Ersatz für den LLM-Bahnhof**: er liest
dessen Gleisliste samt Sticky-Fallback, ohne dass man etwas umstellen muss.

Er entstand, weil der LLM-Bahnhof nur `/v1/models`, `/v1/chat/completions`
(+ `/api/v1/…`-Alias) und `/health` kennt: **kein `/v1/completions`, keine
Einbettungen, keine Ollama-API**. Der Weichensteller vereint beides in einem
Prozess – die Ollama-Oberfläche des Ollama-Bahnhofs (unverändert) und die
Routing-Eigenschaften des LLM-Bahnhofs.

| | Ollama-Bahnhof | LLM-Bahnhof | **Weichensteller** |
|---|---|---|---|
| Ausgabe nach außen | komplettes Ollama | OpenAI (`/v1`) | **komplettes Ollama** |
| `/v1/completions` | ✅ | ❌ | **✅** |
| Responses-API (`/v1/responses`) | ❌ | ❌ | **✅** |
| Einbettungen (`/api/embed`, `/v1/embeddings`) | ✅ | ❌ | **✅** |
| Gleisliste des LLM-Bahnhofs | – | ✅ | **✅ (gelesen)** |
| Sticky-Fallback | – | ✅ | **✅** |
| Simulation ohne Gleis | ✅ | – | **✅** |

## Was ist ein Gleis?

Ein Gleis ist **irgendein OpenAI-kompatibler Endpunkt**. Wie die Basis-URL in
`.env` zu schreiben ist, spielt keine Rolle – der Weichensteller normalisiert
sie selbst:

| Angegeben als | Daraus wird | Passt zu |
|---|---|---|
| `https://api.openai.com/v1` | `…/v1/chat/completions` | OpenAI und alles OpenAI-Kompatible |
| `http://rechner:11434/v1` | `…/v1/chat/completions` | **ein echtes Ollama** (über dessen OpenAI-Endpunkt) |
| `http://rechner:8000` | `…/v1/chat/completions` | llm-bahnhof und jeder Router ohne `/v1` |
| `…/v1/chat/completions` | bleibt unverändert | schon ein fertiger Endpunkt |

⚠️ **Bei einem echten Ollama auf dessen `/v1` zeigen, niemals auf `/api/chat`.**
Der native Ollama-Endpunkt ist nicht OpenAI-kompatibel. Gibt man ihn an, hängt
der Weichensteller `/v1/chat/completions` dahinter und es entsteht Unsinn:

```
http://rechner:11434/api/chat  ->  http://rechner:11434/api/chat/v1/chat/completions
```

Richtig ist `http://rechner:11434/v1`.

Damit kann er **ein echtes Ollama ersetzen** (`OLLAMA_HOST` zeigt auf ihn, das
echte Ollama bekommt nichts mehr zu tun) – oder **ein Ollama vortäuschen, das
es gar nicht gibt** (Doku, Vorführung, Test ohne GPU). Beides ist derselbe
Mechanismus.

Eine Gleiszeile hat vier Felder:

```
ROUTE_01=Basis-URL|Schlüssel|Ziel-Modell|Timeout
```

- **Schlüssel** – kommt als `Authorization: Bearer …` unverändert mit. `none`,
  `-`, `ollama` oder `leer` heißt: **kein** eigener Schlüssel; dann reicht der
  **eingehende** Schlüssel des Clients durch. Mit `Bearer …` am Anfang wird er
  nicht noch einmal umgepackt.
- **Ziel-Modell** – **ersetzt** den Modellnamen, den der Client genannt hat. Der
  Client sieht weiter sein eigenes Modell, das Gleis sieht das seine. Ohne
  Angabe (`|` leer lassen) gilt der Clientname.
- **Timeout** – `45s`, `10m`, `1h`. Fehlt er, sind 600 s Vorgabe.

## Die Verbindung

Zwei Gleis-Quellen werden zu **einer Kette** verbunden – in fester Reihenfolge:

```
1. Gleise des LLM-Bahnhofs     aus LLM_BAHNHOF_ENV (Vorgabe: ~/llm-bahnhof/.env)
2. eigene Gleise               ROUTE_xx in dieser .env
```

Reihenfolge umstellen:

```
GLEIS_REIHENFOLGE=llm-zuerst      # Vorgabe: erst LLM-Bahnhof, dann eigene
GLEIS_REIHENFOLGE=eigene-zuerst
GLEIS_REIHENFOLGE=nur-llm
GLEIS_REIHENFOLGE=nur-eigene
```

**Warum die Trennung?** Die Free-Tier-Liste wird an *einer* Stelle gepflegt
(`~/llm-bahnhof/.env`) und hier mitbenutzt – trotzdem greift eine eigene Route,
wenn der LLM-Bahnhof nichts liefert. Die Namen bleiben auseinandergehalten:
aus `ROUTE_01` der fremden Datei wird `llm-ROUTE_01`, die eigenen bleiben
`ROUTE_01`.

**Sticky-Fallback** (`STICKY_FALLBACK=1`, Vorgabe): ist ein Gleis erfolgreich,
beginnt die **nächste** Anfrage dort und läuft im Kreis weiter
(`ROUTE_02 → ROUTE_03 → … → ROUTE_01`). So springt der Weichensteller nicht bei jeder
Anfrage wieder auf ein Gleis zurück, das eben noch ausgefallen war. Mit
`STICKY_FALLBACK=0` beginnt jede Anfrage wieder beim ersten Gleis – das
Verhalten des Ollama-Bahnhofs.

Der Ausfall eines Gleises **mitten im Strom** kann nicht mehr abgefangen werden
(der Client hat schon Daten bekommen); dann bricht der Strom mit einer
Fehlerzeile ab – wie beim Ollama-Bahnhof.

## Nach außen: 100 % Ollama

Die gesamte Oberfläche des Ollama-Bahnhofs ist unverändert übernommen:

| Ollama nativ | | OpenAI-kompatibel |
|---|---|---|
| `GET /`, `/api/version`, `/api/tags`, `/api/ps` | | `GET /v1/models` |
| `POST /api/show`, `/api/chat`, `/api/generate` | | `POST /v1/chat/completions` |
| `POST /api/embed`, `/api/embeddings` | | `POST /v1/completions` |
| `POST /api/pull`, `/api/push`, `/api/create`, `/api/copy` | | `POST /v1/embeddings` |
| | | `POST /v1/responses` (Responses-API) |
| `DELETE /api/delete`, `POST/HEAD /api/blobs/<sha256:…>` | | `GET /health` (eigene Diagnose) |

`/api/chat` und `/api/generate` liefern **NDJSON**, `/v1/chat/completions`
SSE – das Format erzeugt der Weichensteller selbst, unabhängig davon, ob die Antwort
aus einem Gleis oder aus der Simulation kommt. Ein Client wie Open WebUI, die
`ollama`-CLI oder Continue merkt keinen Unterschied.

⚠️ **Ehrlichkeit:** simulierte Antworten sind im Text gekennzeichnet
(`SIM_MARKER=1`). Der Weichensteller erfindet **keine Kopfzeilen** – woher eine Antwort
kam, steht in `weichensteller.log` und in `/health`.

## Installation und Start

```bash
git clone <url-dieses-repos> weichensteller
cd weichensteller
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

cp .env.example .env        # einmalig; Vorgaben reichen zum Loslegen

./venv/bin/python3 weichensteller.py --pruefen   # zeigt beide Gleis-Gruppen
./venv/bin/python3 weichensteller.py             # startet auf dem Ollama-Port 11434
# oder:
./weichensteller.sh
```

**Port:** Vorgabe **11434** – der echte Ollama-Port. Der Weichensteller tritt damit
ohne Umstellung an die Stelle eines echten Ollama; ein Client, der auf
`http://127.0.0.1:11434` zeigt, merkt den Unterschied nicht.

> Nur **einer** darf auf 11434 lauschen. Der Ollama-Bahnhof wurde dafür am
> 21.09.2026 gestoppt (`kill -TERM <PID>`); er lauscht sonst auf denselben Port.
> Umgekehrt: soll der Ollama-Bahnhof wieder laufen, hier in der `.env` auf
> 11435 zurückstellen.

## Client anbinden

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
ollama list
ollama run weichensteller "Hallo"

# OpenAI-kompatible Basis-URL
http://127.0.0.1:11434/v1        # Schlüssel egal
```

**Basis-URL ohne `/v1` bei Werkzeugen, die selbst anhängen.** `codehamr`,
Cline oder das OpenAI-SDK sprechen je nach Ausführung die **Responses-API**
(`POST /v1/responses`) und hängen den Pfad an das an, was in der Konfiguration
steht. Steht dort schon `…/v1`, fragt der Client `…/v1/v1/responses` und bekommt
vom Weichensteller zu Recht ein 404. Also `http://10.7.0.116:11434` eintragen, nicht
`http://10.7.0.116:11434/v1`.

## Diagnose

```bash
curl -s http://127.0.0.1:11434/health | python3 -m json.tool
```

Zeigt zusätzlich zum Ollama-Bahnhof: `gleis_reihenfolge`, `sticky_fallback`,
`sticky_gleis` (wo die nächste Anfrage beginnt), `llm_bahnhof_konfiguration`
und `gleis_start` (die tatsächliche Reihenfolge dieser Anfrage). Jedes Gleis
trägt seine Herkunft als `gruppe` (`llm` oder `eigene`).

## Prüfungen

```bash
./venv/bin/python3 -m unittest discover -s tests -v
```

- `tests/test_weichensteller.py` – die 54 Prüfungen des Ollama-Bahnhofs
  (Oberfläche, Simulation, Gleis-Durchreichen gegen einen Stub-Server)
- `tests/test_verbindung.py` – das Neue: zwei Gleis-Quellen in fester
  Reihenfolge, fehlende LLM-Bahnhof-Datei, Namensraum-Trennung, Zeitangabe im
  Modellfeld, Sticky-Fallback (Start am gemerkten Gleis, Abschalten,
  unbekanntes Merkmal)

Alles läuft **ohne Netz** und ohne ein echtes Modell.

Dazu der **Rauchtest durch die ganze Kette** – startet den Weichensteller und
einen Stub-Anbieter, lässt das erste Gleis ausfallen und prüft die
Ollama-Oberfläche, die feste Gleis-Reihenfolge, den Fallback, das
Sticky-Verhalten, `/v1/completions`, Einbettungen und den NDJSON-Strom
(19 Prüfungen):

```bash
./venv/bin/python3 rauchtest.py
```

## Grenzen

- **Keine Modellgewichte, keine Inferenz.** Der Weichensteller formt Antworten.
- Einbettungen sind aus der Prüfsumme des Textes gerechnet – deterministisch,
  aber bedeutungslos (wenn kein Gleis mit `/embeddings` konfiguriert ist).
- Die Gleislisten beider Quellen werden **beim Start** gelesen; eine geänderte
  `~/llm-bahnhof/.env` wirkt erst nach einem Neustart.
- **Einbettungen brauchen ein Gleis, das sie kann.** Die Gleisliste des
  LLM-Bahnhofs liefert **keine** Einbettungen (der LLM-Bahnhof kennt
  `/v1/embeddings` nicht – im Rauchtest mit 404 belegt). Im Modus
  `durchreichen` scheitert `/api/embed` dann mit **503**. Mit `MODUS=auto`
  (Vorgabe) oder `SIM_FALLBACK=1` antwortet stattdessen die Simulation –
  deterministisch, aber bedeutungslos. Wer echte Einbettungen braucht,
  trägt ein Gleis mit `/embeddings` ein (z. B. ein echtes Ollama).
- `pull`/`push` laden und senden nichts – sie verwalten nur den eigenen Katalog.

## Lizenz

**GNU Affero General Public License v3.0 oder später** – der vollständige
Text steht in [`LICENSE`](LICENSE).

AGPL-3.0 ist hier nicht zufällig gewählt: das Programm ist ein abgeleitetes
Werk. Es geht auf den `llm-bahnhof` zurück, der seinerseits aus dem
`ollama-bahnhof` hervorging. Wer eine ältere Fassung kopiert und verändert,
muss diese Änderungen unter derselben Lizenz weitergeben – und wer den
Weichensteller als Netzwerkdienst nutzt, muss den Quelltext anbieten (Abschnitt 13).

Zwei Namen, die nicht dasselbe sind: Das Programm heißt **Weichensteller**,
nach außen gibt es sich aber als **Ollama** aus. Ein Client soll nicht
unterscheiden können, ob ein echter Ollama-Server antwortet oder dieser
Proxy. Deshalb heißen die Endpunkte `/api/chat`, `/api/tags`, `/api/ps` und
so weiter, und `/v1/models` meldet `owned_by: weichensteller`.


---

**Powered by AI** – entwickelt mit Unterstützung KI-gestützter Werkzeuge für die
Codeänderungen und die Dokumentation. Der Code, die Tests und die
Architekturentscheidungen stammen von einer menschlich verantwortlichen Person;
für verbleibende Fehler ist entsprechend das Projekt verantwortlich.
