# Speech-to-Phrase

Constrained, on-device speech-to-text for Home Assistant over the Wyoming
protocol. It recognizes only the sentences you choose — your own custom
commands plus a selectable subset of Home Assistant's built-in voice commands —
which keeps recognition fast and accurate on modest hardware.

The add-on is **speech-to-text only**: it turns audio into a transcript and
hands that to Home Assistant, whose own conversation agent decides what the
command means and runs it. Nothing here replaces or bypasses that agent.

## How it works

The recognizer (speech-to-phrase-lib) compiles your enabled sentences into a
grammar and decodes audio against it. Three sentence sources feed the grammar:

1. **Built-in commands** — curated, speech-first templates the add-on ships for
   each supported language, organized by intent and *slot combination*
   (e.g. `HassTurnOn / name_only`). You enable/disable these in the web UI.
2. **Your custom sentences** — free text you write in the web UI.
3. **Your entities** — `{name}` slots are filled from your Home Assistant
   registry, **scoped by domain** (a lock named "front door" is only recognized
   in lock phrasings, never "front door on").

## Configuration UI

Open the add-on's **Web UI** (ingress) and pick a **language**. Saving re-trains
the grammar.

- **Commands** — one row per built-in command, written the way you'd say it.
  Switch off what you don't use: a smaller grammar is recognized faster and
  more accurately, and the meter at the top shows what each choice costs.
  **Details** shows every phrasing that command accepts.
- **Custom commands** — your own phrasings, recognized and passed to Home
  Assistant as text. Supports `{name}`, `{area}`, `{floor}`, `[optional]`,
  `(a|b)` and `{0..100:slot}`; a sentence using `{name}` also picks which
  device types that slot may match.
- **Devices & Lists** — the entity, area and floor names pulled from your
  registry. Switching one off removes it from *every* command, which is the
  main way to shrink a grammar cluttered with devices you never speak to.
- **Settings** — the per-language score gate (see `max_score` below).

## Options

| Option | Description |
|---|---|
| `language` | Default language for the recognizer. |
| `backend` | `auto` (per-language), `citrinet`, or `coqui`. |
| `default_importance` | Built-ins at/above this tier are on by default. |
| `expansion_budget` | Soft cap on grammar size (sentence-equivalents). |
| `use_score_gating` / `max_score` | Defer low-confidence results to the cloud. `max_score` is the default score gate (lower = stricter). Leave it unset to use a per-backend default (Citrinet `5.0`, Coqui `2.0` — the scales differ); set it to override for all languages. It can also be overridden per-language in the web UI (**Settings → Recognition**), which the STT server hot-reloads. |
| `debug_logging` | Verbose logs. |

## Developer notes

- Curated templates: `sentences/<lang>/<Intent>/<slot_combination>.yaml`
  (Speech-to-Phrase template syntax; ranges use hassil's `{from..to[,step]:slot}`).
- `tools/subset_check.py` — CI gate proving, per `(lang, intent, combo)`, that
  the curated language is a **subset** of home-assistant-intents (FST difference).
- `tools/audio_test.py` — pipeline sanity check: TTS → device RIR + noise sweep →
  recognizer, plus OOV false-accept detection.
- Run the UI locally:
  `python src/app.py --data ./data --port 8099`
- `src/intent_server.py` is a Wyoming *intent* service (text in → intent out).
  It is complete but not started by the add-on; pass `--intent` to app.py to
  bring it up for development. Custom-command `intent`/`action` modes and the
  matcher's alias/per-command-exclusion support only take effect through it.
