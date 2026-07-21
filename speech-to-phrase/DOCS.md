# Speech-to-Phrase

Constrained, on-device speech-to-text for Home Assistant over the Wyoming
protocol. It recognizes only the sentences you choose — your own custom
commands plus a selectable subset of Home Assistant's built-in voice commands —
which keeps recognition fast and accurate on modest hardware.

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

Open the add-on's **Web UI** (ingress). Pick a **language**, edit your custom
sentences, and toggle which built-in commands to recognize. Each built-in shows
its importance (`required` / `usable` / `complete` / `optional`) and an example.
Saving re-trains the grammar. Disabling commands you don't use (or for devices
you don't own) keeps recognition fast and accurate.

## Options

| Option | Description |
|---|---|
| `language` | Default language for the recognizer. |
| `backend` | `auto` (per-language), `citrinet`, or `coqui`. |
| `default_importance` | Built-ins at/above this tier are on by default. |
| `expansion_budget` | Soft cap on grammar size (sentence-equivalents). |
| `use_score_gating` / `max_score` | Defer low-confidence results to the cloud. `max_score` is the default score gate (lower = stricter). Leave it unset to use a per-backend default (Citrinet `5.0`, Coqui `2.0` — the scales differ); set it to override for all languages. It can also be overridden per-language in the web UI (**Commands → Recognition**), which the STT server hot-reloads. |
| `debug_logging` | Verbose logs. |

## Developer notes

- Curated templates: `sentences/<lang>/<Intent>/<slot_combination>.yaml`
  (Speech-to-Phrase template syntax; ranges use hassil's `{from..to[,step]:slot}`).
- `tools/subset_check.py` — CI gate proving, per `(lang, intent, combo)`, that
  the curated language is a **subset** of home-assistant-intents (FST difference).
- `tools/audio_test.py` — pipeline sanity check: TTS → device RIR + noise sweep →
  recognizer, plus OOV false-accept detection.
- Run the UI locally:
  `python src/app.py --intents-yaml <intents.yaml> --data ./data --port 8099`
