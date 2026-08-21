# Changelog

## 0.2.0

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

## 0.1.0

- Initial scaffolding: Wyoming STT add-on with web UI.
- Per-language custom sentences + selectable built-in slot-combinations.
- Domain-scoped entity-name training (no cross-domain false commands).
- CI tooling: FST subset gate and audio pipeline sanity check.
