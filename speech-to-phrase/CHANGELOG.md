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
- Fixed: the add-on's run script passed a `--intents-yaml` flag app.py does not
  accept, which aborted startup.

## 0.1.0

- Initial scaffolding: Wyoming STT add-on with web UI.
- Per-language custom sentences + selectable built-in slot-combinations.
- Domain-scoped entity-name training (no cross-domain false commands).
- CI tooling: FST subset gate and audio pipeline sanity check.
