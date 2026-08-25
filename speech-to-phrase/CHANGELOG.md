# Changelog

## 0.2.0

- Added **debug mode** (web UI, **Settings → Debug mode**): a live list of what
  the recognizer heard, with the phrasing it matched, the source that phrasing
  came from (built-in command / custom command / sentence trigger / question
  answer), its score against the gate, and whether the gate accepted it. While
  it is on the STT server hands Home Assistant an empty transcript for every
  utterance, so tuning `max_score` against commands that *should* be rejected
  cannot fire the ones that pass. Applies immediately — the toggle changes only
  runtime reporting, not the grammar, so it needs no retrain.

- Sentence triggers and question answers configured in Home Assistant are added
  to the grammar again, each behind its own option (`sentence_triggers`,
  `question_answers`, both on by default). A `conversation:` trigger phrase or an
  `assist_satellite.ask_question` answer that isn't in the grammar can never be
  transcribed, so the automation could never fire or branch on it. Sentences
  carrying Jinja2 templates, and answers of disabled automations, are skipped.
  Both appear in the web UI (**Settings → From Home Assistant**) as a switch per
  source — overriding the add-on option for that language — listing the phrases
  each contributes, what they cost, and any that can't be recognized. Their
  phrases are counted in the Commands grammar meter, which previously described
  only the built-in commands.

- Ship as speech-to-text only: the Wyoming intent service is no longer started
  (`--intent` re-enables it for development). Home Assistant handles the
  transcript.
- Simplified web UI:
  - Custom commands are speech-to-text only — sentences plus, when one uses
    `{name}`, the device types it may match. Intent/action/response editing is
    gone.
  - Command **Details** is read-only: it shows the ways a command can be said.
    Per-command target narrowing and user-added phrasings are gone.
  - **Devices & Lists** is a plain on/off switch per entity/area/floor —
    exclusion is global. Spoken aliases are gone.
  - The **Test** tab and its `/api/test` and `/api/validate_sentence` endpoints
    are gone.
- Added the `token_bonus` option (`--token-bonus`, also on
  `tools/audio_test.py`). The library has supported a word-insertion reward all
  along; the add-on never passed one, leaving the CTC length bias unopposed, so
  long commands could decode as short in-grammar phrases ("set the office light
  brightness to ten percent" heard as "office light off"). Defaults per backend:
  Citrinet `2.0` — fit on en, where exact decodes over 12 long
  brightness/speed commands went 3/12 at `0` to 10/12 at `2.0` — and Coqui `0.0`,
  which is unmeasured.
- `tools/audio_test.py` no longer embeds a Home Assistant token; it reads
  `HA_TOKEN` and `HA_URL` from the environment.
- Removed the `expansion_budget` and `use_score_gating` add-on options: nothing
  read them.
- Fixed: the add-on's run script passed a `--intents-yaml` flag app.py does not
  accept, which aborted startup.
- Fixed: the background watch thread raised `UnboundLocalError` on its first
  pass and never retrained, and applied one language's voice-targeting
  overrides to all of them.
- Fixed: the Wyoming service advertised a hardcoded version `0.1.0`; it now
  reports the add-on manifest version.
- Fixed: `wyoming_server.py --backend auto` always chose Citrinet instead of
  the per-language backend, so a Coqui-only language found no model.
- Fixed: an unparseable `max_score` was persisted as `0.1`, a gate that accepts
  nothing; it is now rejected and the previous value stands.
- Dropped the unused `share:rw` mapping from the add-on manifest.
- The recognition library is now vendored in `lib/` and built from source, so
  the add-on is self-contained: no `speech-to-phrase-lib` git dependency, and
  no `git` in the image. The vendored copy is the newer speech-to-phrase-2
  code, which adds a subword-segmentation lattice so the grammar accepts any
  valid tokenization of a word rather than one hard-coded segmentation.
- Grammar correctness: list values written as hassil patterns (`(closed|shut)`,
  `[securely] locked`) are expanded into their spoken forms, and written-only
  forms are dropped — `timer_half` no longer trains a dead `1/2` path, and a
  hyphenated word is spaced rather than demanding an unpronounceable token.
  The en grammar drops 667 -> 539 templates with only the two unsayable
  `{0..100}%` phrasings actually removed; the rest were duplicate spellings
  collapsing onto their spoken form.
- `tools/lang_check.py`: per-language round-trip check (package templates ->
  grammar -> TTS -> decode) for validating a new language.
- Smaller image: the C++/CMake/git toolchain needed to build the native OpenFST
  module is now purged in the layer that installs it (~200 MB), keeping only
  the shared libraries the built module actually links against.
- The default English Citrinet model is bundled into the image, so a fresh
  install no longer downloads ~140 MB on first boot and works offline. A model
  already in `/data/models` still takes precedence; `--build-arg BUNDLE_MODEL=`
  builds without it.

## 0.1.0

- Initial scaffolding: Wyoming STT add-on with web UI.
- Per-language custom sentences + selectable built-in slot-combinations.
- Domain-scoped entity-name training (no cross-domain false commands).
- CI tooling: FST subset gate and audio pipeline sanity check.
