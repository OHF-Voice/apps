# Speech-to-Phrase

Constrained, on-device speech-to-text for Home Assistant over the Wyoming
protocol. It recognizes only the sentences you choose — your own custom
commands plus a selectable subset of Home Assistant's built-in voice commands —
which keeps recognition fast and accurate on modest hardware.

The add-on is **speech-to-text only**: it turns audio into a transcript and
hands that to Home Assistant, whose own conversation agent decides what the
command means and runs it. Nothing here replaces or bypasses that agent.

## How it works

The recognizer (the vendored `speech_to_phrase` library in `lib/`) compiles
your enabled sentences into a
grammar and decodes audio against it. Four sentence sources feed the grammar:

1. **Built-in commands** — curated, speech-first templates the add-on ships for
   each supported language, organized by intent and *slot combination*
   (e.g. `HassTurnOn / name_only`). You enable/disable these in the web UI.
2. **Your custom sentences** — free text you write in the web UI.
3. **Your entities** — `{name}` slots are filled from your Home Assistant
   registry, **scoped by domain** (a lock named "front door" is only recognized
   in lock phrasings, never "front door on").
4. **Phrases Home Assistant already listens for** — your automations' sentence
   triggers and the `answers:` of any `assist_satellite.ask_question` action.
   These are picked up automatically (see the `sentence_triggers` and
   `question_answers` options): a trigger phrase that isn't in the grammar can
   never be transcribed, so the trigger would never fire.

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
- **Settings** — the per-language score gate (see `max_score` below), and a
  switch per Home-Assistant sentence source (see `sentence_triggers` and
  `question_answers`). Each switch lists the phrases it currently contributes
  and what they cost, and flags any that can't be recognized — a phrase with no
  spoken form, or one using a list Speech-to-Phrase can't fill in. These
  phrases are part of the grammar, so they're included in the Commands meter's
  total. The switches override the add-on options for the selected language.

## Options

| Option | Description |
|---|---|
| `language` | Default language for the recognizer. |
| `backend` | `auto` (per-language), `citrinet`, or `coqui`. |
| `default_importance` | Built-ins at/above this tier are on by default. |
| `sentence_triggers` | Add your automations' `conversation:` sentence-trigger phrases to the grammar (default on). Without this, a trigger phrase that isn't otherwise recognizable is never transcribed, so the automation never fires. Turn it off to keep the grammar to what you configured here. Overridable per-language in the web UI (**Settings → From Home Assistant**). |
| `question_answers` | Add the `answers:` sentences of every `assist_satellite.ask_question` action to the grammar (default on), so a spoken reply to a question your automation asks can be recognized. Finding them means reading each automation and script config, so on a large installation this costs a little time at every retrain. Sentences containing Jinja2 templates (`{{ … }}`) are skipped — they have no fixed spoken form. Overridable per-language in the web UI. |
| `max_score` | Defer low-confidence results to the cloud. This is the default score gate (lower = stricter). Leave it unset to use a per-backend default (Citrinet `5.0`, Coqui `2.0` — the scales differ); set it to override for all languages. It can also be overridden per-language in the web UI (**Settings → Recognition**), which the STT server hot-reloads. |
| `token_bonus` | Word-insertion reward per emitted token. Leave it unset to use a per-backend default (Citrinet `2.0`, Coqui `0.0` — the cost scales differ and Coqui has not been measured); `0` disables it. The FST decode picks the lowest-cost path, and audio a path doesn't account for is absorbed by CTC blanks almost for free — so without a bonus a shorter in-grammar phrase can beat the longer one actually spoken ("set the office light brightness to ten percent" heard as "office light off"). Too high and the decoder starts inserting words. Re-fit with `tools/audio_test.py --token-bonus`. |
| `debug_logging` | Verbose logs. |

## Self-contained build

The recognition library lives in `lib/` (package `speech_to_phrase`, built from
`lib/speech_to_phrase/_fst/fstmodule.cc` against OpenFST) and is installed from
source by `requirements.txt`'s `./lib` entry. There is no dependency on an
external checkout or git host, so the image builds from this directory alone.

## Image size and the bundled model

The build compiles the vendored library's native OpenFST module, so it needs
`cmake`, `g++` and `libfst-dev`. Those are purged in the same layer they
are installed in (~200 MB of toolchain that never reaches the shipped image);
the OpenFST *runtime* library is detected from what the built module links and
marked manual so `--auto-remove` can't take it with them.

The default English Citrinet model (~140 MB) is baked in, so a fresh install
starts without downloading anything. A model already present in `--models-dir`
(`/data/models`, which survives add-on updates) still wins, so a user who
downloaded one keeps using it. Build a lean image that downloads on demand with:

    docker build --build-arg BUNDLE_MODEL= ...

## Developer notes

- Curated templates: `sentences/<lang>/<Intent>/<slot_combination>.yaml`
  (Speech-to-Phrase template syntax; ranges use hassil's `{from..to[,step]:slot}`).
- `tools/subset_check.py` — CI gate proving, per `(lang, intent, combo)`, that
  the curated language is a **subset** of home-assistant-intents (FST difference).
- `tools/audio_test.py` — pipeline sanity check: TTS → device RIR + noise sweep →
  recognizer, plus OOV false-accept detection.
- Run the UI locally:
  `python src/app.py --data ./data --port 8099`
- `tools/audio_test.py` needs a Home Assistant instance for TTS; point it at one
  with `HA_TOKEN` (and `HA_URL`, default `http://homeassistant.local:8123`).
  Clips are cached under `tests/wav/.tts_cache`, so a re-run with the same
  `--seed` makes no TTS calls.
- `lib/` is the recognition library; rebuild it after editing with
  `pip install ./lib` (needs `cmake`, `g++`, `libfst-dev`).
- `src/intent_server.py` is a Wyoming *intent* service (text in → intent out).
  It is complete but not started by the add-on; pass `--intent` to app.py to
  bring it up for development. Custom-command `intent`/`action` modes and the
  matcher's alias/per-command-exclusion support only take effect through it.
