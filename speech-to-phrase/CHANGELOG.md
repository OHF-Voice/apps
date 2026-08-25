# Changelog

## 2.0.0

- The add-on options are down to two: **the language, and verbose logging.**
  Everything else that was there — which commands are on, the score gate,
  whether to pull in the phrases Home Assistant already listens for, the
  recognizer backend, the word-insertion reward — belongs to a language and is
  edited in the web UI, next to the meter that shows what each choice costs. An
  add-on option could show none of that and applied to every language at once.
  Nothing configured per language is lost, and the shipped behaviour is
  unchanged: the same backend selection (whichever has a model), the same
  `usable` starting set, both sentence sources on, the same per-backend gate and
  reward. `default_importance` is the one worth calling out — it only ever
  seeded a language's *first* run, so changing it later did nothing at all,
  which is a poor thing for a settings page to offer.
- `language` is a **list**, defaulting to English, of the languages that ship
  Speech-to-Phrase templates: `ca`, `cs`, `de`, `en`, `es`, `fr`, `it`, `nl`.
  It was free text, so it was possible to type a language with no sentences to
  recognize and get an add-on that refused to start.

- Seven more languages: `home-assistant-intents` 2026.8.25 ships the
  Speech-to-Phrase sentence blocks for **Catalan, Czech, Dutch, French, German,
  Italian and Spanish**, so those languages now get a grammar instead of
  nothing. Each one round-trips every example command in its own language
  through the real path (package templates → grammar → HA Cloud TTS → decode):
  de 59/59, ca 56/56, cs 59/59, es 55/55, fr 56/56, it 50/50, nl 53/53
  commands resolved to the right command, all within the score gate.
- Fixed: changing the acoustic model for a language left the old grammar in
  place, and the recognizer then returned an empty transcript for every
  utterance — silently, and until something else happened to change the
  grammar. A `grammar.fst` is compiled against one model's vocabulary (its arc
  labels *are* that model's token ids), but the staleness fingerprint covered
  only the templates, the entity/area/floor values and the backend. Swapping
  French from Citrinet to Conformer (1024 tokens → 128) therefore looked like no
  change at all. The model is part of the fingerprint now, so a model change
  retrains once on the first start after the update.
- Fixed: setting `language` to one of the languages with an acoustic model but
  no sentence templates yet (`zh`, `ru`, `hr`, `hi`, `sl`) took the add-on down
  on every start with `ValueError: No sentence templates provided`. The
  configured language is checked up front and reported as what it is — a
  setting to change, with the supported languages listed — and an empty grammar
  anywhere else leaves the previous one alone instead of raising.
- Fixed: asking for a `backend` that has no model for the language quietly
  substituted the *other* backend's model, downloaded it, and then failed to
  load it on every start (`language: cs` with `backend: citrinet` died on
  `cs_CZ-coqui/tokens.txt`). A missing model now reads as missing: the add-on
  serves the web UI, and the log says which backends that language does have.
- Removed the `es_ES-coqui` mapping: the model cannot load at all (its alphabet
  has 36 symbols where the decode path expects 30), so offering it only gave
  anyone who set `backend: coqui` a download followed by a crash. Spanish runs
  on Citrinet.
- Hardened: the `lang` a web-UI request asks for is checked against the
  languages the add-on actually serves before it is used to build a path.
  `<data>/<lang>/` is a join, so a crafted value wrote `enabled.json`,
  `settings.json` and `custom_commands.json` outside the data directory.
- The web UI now follows the `language` option instead of offering a picker.
  One recognizer runs, for one language, but the UI let you select any of the
  eight and then edit it — so you could spend a while tuning commands for a
  language speech-to-text was not serving and conclude the add-on was broken.
  The header shows the active language, every request is about that language,
  and a page left open from before the option changed refuses to save rather
  than applying those edits to the current language.
- Debug mode reports **how long recognition took** — the time from the audio
  stopping to the transcript being ready, which is the pause a user actually
  experiences. Hovering it gives the length of speech and the ratio between
  them. Tens of milliseconds once warm, and visibly more on the first utterance
  after a start; a decode that is slow and one that is wrong were previously
  indistinguishable from the outside.
- Fixed: a custom command whose text contained markup was rendered as markup in
  the **Devices & Lists** tab. Every value on that page was escaped except the
  "Used by:" labels, and a custom command's label is its own first sentence.
- Fixed: one utterance can no longer grow the audio buffer without limit. It
  grew on every chunk and only an audio-stop emptied it, so a satellite that
  stopped sending one — crashed, wedged, or streaming an open microphone —
  grew it until the add-on was killed for using too much memory. Past 30
  seconds (or 16 MiB, whichever comes first) the tail is dropped and the head
  kept, since the command follows the wake word, and the log says which
  satellite to look at.
- Fixed: a `-inf` score in the debug feed would have serialized as invalid
  JSON and broken the live view. The check listed `+inf` and `nan` by hand;
  it tests for a finite number now.
- Hardened: acoustic-model archives extract with tarfile's `data` filter, so a
  member cannot escape the extraction directory or bring along a link, a device
  node or a setuid bit. This is also the Python 3.14 default, so behaviour no
  longer shifts on an interpreter bump.
- Fixed in `tools/lang_check.py`: it measured whichever `speech_to_phrase` was
  importable, and on a development machine an editable install of the upstream
  library shadows the vendored `lib/` through `sys.meta_path` — which
  `sys.path` cannot override. Upstream has no subword-segmentation lattice, so
  clearly-spoken commands came back empty or as a different command, and a
  language looked broken when only the harness was.

- Added **debug mode** (web UI, **Settings → Debug mode**): a live list of what
  the recognizer heard, with the phrasing it matched, the source that phrasing
  came from (built-in command / custom command / sentence trigger / question
  answer), its score against the gate, and whether the gate accepted it. While
  it is on the STT server hands Home Assistant an empty transcript for every
  utterance, so tuning `max_score` against commands that *should* be rejected
  cannot fire the ones that pass. Applies immediately — the toggle changes only
  runtime reporting, not the grammar, so it needs no retrain — and is
  session-only: it is never written to `settings.json`, so a restart always
  comes back with it off rather than booting into a mute assistant with nothing
  on disk to explain it.
- Fixed: a custom command written with `[optional]` or `(a|b)` was reported by
  debug mode as "not in the grammar". The trainer expands those inside the FST,
  so the command stays a single template while the decoder can emit any of its
  phrasings; attribution compared against the template as written and matched
  none of them.
- Fixed: `{0..100:slot}` — the range form the custom-command syntax documents —
  was read as a reference to a list named `0..100`. Debug mode could not
  attribute any numeric custom command, and the grammar-size meter priced a
  101-value range at a single phrase.

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
