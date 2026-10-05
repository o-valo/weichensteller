<p align="center">
  <img src="weichensteller.jpg" alt="Weichensteller banner: inside a data centre a railway track runs between server racks while a robot throws a set of points." width="100%">
</p>

# 🚉 Weichensteller – switchman (Ollama + LLM-Bahnhof)

English | **[Deutsch](README.ger.md)**

**One program, two stations.** *Weichensteller* is German for a **switchman** –
the one who throws the points and sets the course. Outward, this Weichensteller
**simulates a complete Ollama API** – every tool that "can do Ollama" talks to
it and cannot tell the difference. But it does **not** forward those requests to a
real Ollama: it forwards them to **any endpoint** it calls a *track*. That can be a
**real Ollama endpoint**, an **OpenAI-compatible** endpoint, a free-tier
provider, or an entire router. If one fails, it moves on to the next.

In practice it serves mainly as a **stand-in for the LLM-Bahnhof**: it reads
that router's track list including its sticky fallback, with nothing to
reconfigure.

It came about because the LLM-Bahnhof only knows `/v1/models`,
`/v1/chat/completions` (plus an `/api/v1/…` alias) and `/health`: **no
`/v1/completions`, no embeddings, no Ollama API**. The Weichensteller combines
both in a single process – the Ollama surface of the Ollama-Bahnhof (unchanged)
and the routing behaviour of the LLM-Bahnhof.

| | Ollama-Bahnhof | LLM-Bahnhof | **Weichensteller** |
|---|---|---|---|
| What clients see | complete Ollama | OpenAI (`/v1`) | **complete Ollama** |
| `/v1/completions` | ✅ | ❌ | **✅** |
| Responses API (`/v1/responses`) | ❌ | ❌ | **✅** |
| Embeddings (`/api/embed`, `/v1/embeddings`) | ✅ | ❌ | **✅** |
| Reads the LLM-Bahnhof's track list | – | ✅ | **✅ (read-only)** |
| Sticky fallback | – | ✅ | **✅** |
| Simulation without any track | ✅ | – | **✅** |

## What is a track?

A track is **any OpenAI-compatible endpoint**. How the base URL is written in
`.env` does not matter – the Weichensteller normalises it itself:

| Written as | Becomes | Fits |
|---|---|---|
| `https://api.openai.com/v1` | `…/v1/chat/completions` | OpenAI and anything OpenAI-compatible |
| `http://host:11434/v1` | `…/v1/chat/completions` | **a real Ollama** (via its OpenAI endpoint) |
| `http://host:8000` | `…/v1/chat/completions` | the llm-bahnhof and any router without `/v1` |
| `…/v1/chat/completions` | left unchanged | already a finished endpoint |

⚠️ **For a real Ollama, point at its `/v1`, never at `/api/chat`.** Ollama's
native endpoint is not OpenAI-compatible. Give it anyway and the Weichensteller
appends `/v1/chat/completions` behind it, producing nonsense:

```
http://host:11434/api/chat  ->  http://host:11434/api/chat/v1/chat/completions
```

The correct value is `http://host:11434/v1`.

That means it can **replace a real Ollama** (`OLLAMA_HOST` points at it and the
real Ollama has nothing left to do) – or **pretend an Ollama that does not
exist at all** (docs, demos, testing without a GPU). Both are the same
mechanism.

A track line has four fields:

```
ROUTE_01=base-url|key|target-model|timeout
```

- **Key** – passed on unchanged as `Authorization: Bearer …`. `none`, `-`,
  `ollama` or `leer` means: **no** key of its own; the **incoming** client key
  is then forwarded. If it already starts with `Bearer ` it is not wrapped a
  second time.
- **Target model** – **replaces** the model name the client asked for. The
  client keeps seeing its own model, the track sees its own. Leave it empty
  (`|`) to use the client's name.
- **Timeout** – `45s`, `10m`, `1h`. If omitted the default is 600 s.

## The connection

Two sources of tracks are joined into **one chain**, in a fixed order:

```
1. Tracks of the LLM-Bahnhof     from LLM_BAHNHOF_ENV (default: ~/llm-bahnhof/.env)
2. Own tracks                    ROUTE_xx in this .env
```

Change the order:

```
GLEIS_REIHENFOLGE=llm-zuerst      # default: LLM-Bahnhof first, then own tracks
GLEIS_REIHENFOLGE=eigene-zuerst
GLEIS_REIHENFOLGE=nur-llm
GLEIS_REIHENFOLGE=nur-eigene
```

**Why keep them separate?** The free-tier list is maintained in *one* place
(`~/llm-bahnhof/.env`) and reused here – yet an own route still takes effect
when the LLM-Bahnhof delivers nothing. The names stay apart: an `ROUTE_01` from
the foreign file becomes `llm-ROUTE_01`, your own stay `ROUTE_01`.

**Sticky fallback** (`STICKY_FALLBACK=1`, the default): once a track succeeds,
the **next** request starts there and continues round the circle
(`ROUTE_02 → ROUTE_03 → … → ROUTE_01`). That way the Weichensteller does not
hop back to a track that just failed on every single request. With
`STICKY_FALLBACK=0` every request starts at the first track again – the
behaviour of the Ollama-Bahnhof.

A track failing **mid-stream** can no longer be caught (the client has already
received data); the stream then aborts with an error line – as with the
Ollama-Bahnhof.

## Outward: 100 % Ollama

The entire surface of the Ollama-Bahnhof is carried over unchanged:

| Ollama native | | OpenAI-compatible |
|---|---|---|
| `GET /`, `/api/version`, `/api/tags`, `/api/ps` | | `GET /v1/models` |
| `POST /api/show`, `/api/chat`, `/api/generate` | | `POST /v1/chat/completions` |
| `POST /api/embed`, `/api/embeddings` | | `POST /v1/completions` |
| `POST /api/pull`, `/api/push`, `/api/create`, `/api/copy` | | `POST /v1/embeddings` |
| | | `POST /v1/responses` (Responses API) |
| `DELETE /api/delete`, `POST/HEAD /api/blobs/<sha256:…>` | | `GET /health` (own diagnostics) |

`/api/chat` and `/api/generate` return **NDJSON**, `/v1/chat/completions` returns
SSE – the Weichensteller produces those formats itself, regardless of whether
the answer came from a track or from the simulation. A client such as Open
WebUI, the `ollama` CLI or Continue cannot tell the difference.

⚠️ **Honesty first:** simulated answers are marked in the text
(`SIM_MARKER=1`). The Weichensteller invents **no headers** – where an answer
came from is written to `weichensteller.log` and to `/health`.

## Installation and running

```bash
git clone <url-of-this-repo> weichensteller
cd weichensteller
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

cp .env.example .env        # once; the defaults are enough to get going

./venv/bin/python3 weichensteller.py --pruefen   # shows both track groups
./venv/bin/python3 weichensteller.py             # starts on the Ollama port 11434
# or:
./weichensteller.sh
```

**Port:** the default is **11434** – the real Ollama port. The Weichensteller
therefore takes the place of a real Ollama without any reconfiguration; a
client pointing at `http://127.0.0.1:11434` cannot tell the difference.

> Only **one** process may listen on 11434. The Ollama-Bahnhof was stopped for
> that reason on 21.09.2026 (`kill -TERM <PID>`); otherwise it listens on the
> same port. The other way round: if the Ollama-Bahnhof should run again, set
> the port back to 11435 in this `.env`.

## Connecting a client

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
ollama list
ollama run weichensteller "Hello"

# OpenAI-compatible base URL
http://127.0.0.1:11434/v1        # any API key will do
```

**Base URL without `/v1` for tools that append it themselves.** `codehamr`,
Cline or the OpenAI SDK speak the **Responses API** (`POST /v1/responses`)
depending on how they are invoked, and they append the path to whatever is in
the configuration. If that already reads `…/v1`, the client asks for
`…/v1/v1/responses` and rightly gets a 404 from the Weichensteller. So enter
`http://10.7.0.116:11434`, not `http://10.7.0.116:11434/v1`.

## Diagnostics

```bash
curl -s http://127.0.0.1:11434/health | python3 -m json.tool
```

Beyond what the Ollama-Bahnhof reports, this shows `gleis_reihenfolge`,
`sticky_fallback`, `sticky_gleis` (where the next request starts),
`llm_bahnhof_konfiguration` and `gleis_start` (the actual order for this
request). Every track carries its origin as `gruppe` (`llm` or `eigene`).

## Tests

```bash
./venv/bin/python3 -m unittest discover -s tests -v
```

- `tests/test_weichensteller.py` – the 54 checks of the Ollama-Bahnhof
  (surface, simulation, passing requests through to a stub server)
- `tests/test_verbindung.py` – the new part: two track sources in fixed order,
  a missing LLM-Bahnhof file, namespace separation, a duration in the model
  field, sticky fallback (starting on the remembered track, switching it off,
  an unknown marker)

All of it runs **without a network** and without a real model.

Plus the **smoke test through the entire chain** – it starts the Weichensteller
and a stub provider, lets the first track fail, and checks the Ollama surface,
the fixed track order, the fallback, the sticky behaviour, `/v1/completions`,
embeddings and the NDJSON stream (19 checks):

```bash
./venv/bin/python3 rauchtest.py
```

## Limits

- **No model weights, no inference.** The Weichensteller shapes answers.
- Embeddings are computed from the checksum of the text – deterministic, but
  meaningless (unless a track with `/embeddings` is configured).
- The track lists of both sources are read **at startup**; a changed
  `~/llm-bahnhof/.env` only takes effect after a restart.
- **Embeddings need a track that can do them.** The LLM-Bahnhof's track list
  delivers **no** embeddings (the LLM-Bahnhof does not know `/v1/embeddings` –
  demonstrated with a 404 in the smoke test). In `durchreichen` mode
  `/api/embed` therefore fails with **503**. With `MODUS=auto` (the default) or
  `SIM_FALLBACK=1` the simulation answers instead – deterministic, but
  meaningless. If you need real embeddings, configure a track that offers
  `/embeddings` (a real Ollama, for example).
- `pull`/`push` neither download nor upload anything – they only manage the
  program's own catalogue.

## License

**GNU Affero General Public License v3.0 or later** – the full text is in
[`LICENSE`](LICENSE).

AGPL-3.0 was not chosen by accident here: the program is a derived work. It
goes back to the `llm-bahnhof`, which in turn emerged from the
`ollama-bahnhof`. Anyone who copies and modifies an older version must pass
those modifications on under the same license – and anyone who runs the
Weichensteller as a network service must offer the source code (section 13).

Two names that are not the same thing: the program is called **Weichensteller**,
but outward it presents itself as **Ollama**. A client should not be able to
tell whether a real Ollama server is answering or this proxy. That is why the
endpoints are called `/api/chat`, `/api/tags`, `/api/ps` and so on, and why
`/v1/models` reports `owned_by: weichensteller`.


---

**Powered by AI** – developed with the support of AI-assisted tools for the code
changes and the documentation. The code, the tests and the architectural
decisions come from a human person who is responsible for them; the project is
likewise responsible for any remaining errors.
