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
   These are picked up automatically, because a trigger phrase that isn't in the
   grammar can never be transcribed and the trigger would never fire. Either
   source can be switched off in the web UI.

## Configuration UI

Open the add-on's **Web UI** (ingress). Everything about what gets recognized
lives here rather than in the add-on options, because it belongs to one language
and its cost is worth seeing next to the choice. Saving re-trains the grammar.

The header shows the language being edited, which is always the one the
recognizer is running — set it in the add-on options (`language`) and reload.

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
- **Settings** — **Max score**, the confidence gate: a decode is accepted only
  when its score is at or below this, and anything above is handed back to Home
  Assistant as nothing so it can fall back to cloud speech-to-text. Lower is
  stricter. It defaults to a value fitted per recognizer backend (Citrinet
  `5.0`, Coqui `2.0` — the scales differ), applies immediately with no retrain,
  and **Debug mode** below is how you tune it.
  Also a switch per Home-Assistant sentence source. Each lists the phrases it
  currently contributes and what they cost, and flags any that can't be
  recognized — a phrase with no spoken form, or one using a list
  Speech-to-Phrase can't fill in. Those phrases are part of the grammar, so
  they're counted in the Commands meter's total.

## Debug mode

**Settings → Debug mode** shows a live list of what the recognizer hears: each
utterance, the phrasing it matched, which source that phrasing came from (a
built-in command, one of your custom commands, a sentence trigger, or a question
answer), its score against the current gate, whether the gate accepted it, and
how long recognition **took**. It applies immediately — no save or retrain.

The **Took** column is the time from the audio stopping to the transcript being
ready — the pause a user actually experiences, covering format conversion,
level normalization, silence trimming and the decode. Hover it for the length of
speech it explains and the ratio between them. Expect tens of milliseconds once
warm; the first utterance after a start or a retrain is slower, because the
model and the voice-activity detector are loaded lazily. A number that climbs
with grammar size is the signal to turn commands off — a decode and a mishearing
are indistinguishable from the outside otherwise.

**While debug mode is on, Home Assistant receives an empty transcript for every
utterance, so nothing you say is acted on.** That is the point: tuning the score
gate means deliberately speaking commands that *should* be rejected, and the
ones that pass shouldn't run your lights while you do it. The UI keeps a warning
on screen the whole time it's on; switch it off when you're done.

Because of that, debug mode is **session-only**: it is held in memory, never
written to `settings.json`, and a restarted add-on always comes back with it
off. There is no add-on option for it, and no way to leave it on by accident —
a mute assistant with nothing on disk to explain it is not a state to boot
into. The recognition log is discarded at the same time.

Rejected rows are the useful ones — they show what a phrase was misheard as and
by how much it missed, which tells you whether to raise the gate or to add the
phrase as a custom command.

## Options

There are two, and that is on purpose. Everything else about how
Speech-to-Phrase behaves belongs to a *language* — which commands you want, how
confident a decode has to be, whether to pull in the phrases your automations
listen for — and lives in the web UI, next to the grammar-size meter that shows
what each choice costs. An add-on option cannot show you that, and would apply
to every language at once.

| Option | Description |
|---|---|
| `language` | The language to recognize. A list, because only the languages that ship Speech-to-Phrase sentence templates can be recognized at all: `ca`, `cs`, `de`, `en`, `es`, `fr`, `it`, `nl`. Defaults to `en`. This is also the only language the web UI edits — it has no language picker, because one recognizer runs and editing a language it wasn't serving was a way to wonder why nothing changed. Change it here, then reload the UI; a page left open from before refuses to save and says so. |
| `debug_logging` | Verbose logs. Includes a line per utterance with the score and how long recognition took, whether or not **Debug mode** is on. |

### Things that used to be options

Upgrading from an earlier version? These were removed, and nothing you had
configured per language is lost — the web UI's copy is what was always in
effect.

| Was | Now |
|---|---|
| `default_importance` | Fixed at `usable`, about half the catalogue: a larger grammar has more ways to mishear, so brightness, volume, mute, fan speed, cover position and "is the door open" start off and are one click away in **Commands**. It only ever seeded a language's *first* run — changing it later did nothing at all, which is a poor thing for a settings page to offer. |
| `sentence_triggers`, `question_answers` | Both on. Switch either off per language in **Settings → From Home Assistant**, which also lists the phrases it contributes and what they cost. |
| `max_score` | **Settings → Max score**, per language, applied without a retrain. Still defaults to the value fitted per backend (Citrinet `5.0`, Coqui `2.0`). |
| `backend` | Chosen automatically: whichever has a model for the language. Nobody setting up a voice assistant should have to pick a CTC topology, and the wrong pick only ever produced a model that would not load. |
| `token_bonus` | Fixed at the value fitted per backend (Citrinet `2.0`, Coqui `0.0`). Re-fit with `tools/audio_test.py --token-bonus` if you are working on the recognizer itself. |

