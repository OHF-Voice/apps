# Speech-to-Phrase

Constrained, on-device speech-to-text for Home Assistant over the Wyoming
protocol. It recognizes only the sentences you choose — your own custom
commands plus a selectable subset of Home Assistant's built-in voice commands —
which keeps recognition fast and accurate on modest hardware.

The add-on is **speech-to-text only**: it turns audio into a transcript and
hands that to Home Assistant, whose own conversation agent decides what the
command means and runs it. Nothing here replaces or bypasses that agent.

## How it works

The recognizer compiles your enabled sentences into a grammar and decodes audio
against it, so it can only ever return something you chose. Four sentence
sources feed that grammar:

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

Open the add-on's **Web UI** (ingress). Everything about what gets recognized is
configured here; saving re-trains the grammar.

The header shows the language, which is the one set in the add-on options and
the one the recognizer is running. Change it there and reload the page.

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
ready — the pause you actually experience, covering format conversion, level
normalization, silence trimming and the decode. Hover it for the length of
speech it explains and the ratio between them. Expect tens of milliseconds once
warm; the first utterance after a start or a retrain is slower, because the model
and the voice-activity detector load on demand. If it climbs as your grammar
grows, turn commands off.

**While debug mode is on, Home Assistant receives an empty transcript for every
utterance, so nothing you say is acted on.** That is the point: tuning the score
gate means deliberately speaking commands that *should* be rejected, and the
ones that pass shouldn't run your lights while you do it. The UI keeps a warning
on screen the whole time it's on; switch it off when you're done.

Debug mode is **session-only**: it is held in memory, never written to disk, and
a restarted add-on always comes back with it off. The recognition log is
discarded at the same time.

Rejected rows are the useful ones — they show what a phrase was misheard as and
by how much it missed, which tells you whether to raise the gate or to add the
phrase as a custom command.

## Options

There are two. Everything else is configured in the web UI, per language, where
the grammar-size meter shows what each choice costs.

| Option | Description |
|---|---|
| `language` | The language to recognize: `ca`, `cs`, `de`, `en`, `es`, `fr`, `it`, `nl`. Defaults to `en`. This is also the language the web UI edits — one recognizer runs, so there is nothing to pick there. Change it here and reload the page; a page left open from before the change refuses to save and says so. |
| `debug_logging` | Verbose logs, including a line per utterance with its score and how long recognition took (whether or not **Debug mode** is on). |

The rest is chosen for you:

- The **recognizer backend** is whichever has a model for the language.
- The **score gate** and the decoder's word-insertion reward start at values
  fitted per backend. The gate is adjustable in **Settings**.
- **Which commands are on** the first time a language is set up is about half
  the catalogue: a larger grammar has more ways to mishear, so brightness,
  volume, mute, fan speed, cover position and "is the door open" start off, one
  click away in **Commands**.
- Both **Home Assistant sentence sources** are on, switchable in **Settings**.

