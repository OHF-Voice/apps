# Script Agent

A local [conversation][] agent that runs your custom Home Assistant [scripts][].
It is currently powered by [Gemma 4][gemma4].

![architecture](https://raw.githubusercontent.com/OHF-Voice/apps/main/script-agent/res/architecture.png)

## Getting started

To install the app, first add `https://github.com/OHF-Voice/apps`
to your app repositories by:

1. Go to "Settings -> Apps"
2. Click the "Install app" button
3. In the 3-dot menu, click "Repositories"
4. Click the "Add" button

After adding the repository, make sure you're in "Settings -> Apps" and "Install
apps". You may need to click "Check for updates" from the 3-dot menu.

Scroll down and find "OHF Apps" and choose "Script Agent",
then click "Install". To check on installation progress, go to "Settings ->
System -> Logs" and change to "Supervisor".

### First boot

Installing the app can take quite a while, since it builds an optimized version
of [llama.cpp][] for your CPU.

The ARM64 image requires ARMv8.2 with FP16 and dot-product instructions. This
includes Raspberry Pi 5 and newer hardware, but not Raspberry Pi 4.

On the first boot, [Gemma 4][gemma4] is downloaded (about 4GB) and the cached
model state is built. This can take several minutes. Setting `hf_token` to a
Hugging Face API token before starting the app makes the download faster and
more reliable.

The web UI is available while this happens and shows a "model is still loading"
notice; voice commands are not handled until it is ready. Once the app is ready,
check "Settings -> Devices & services" for a newly discovered [wyoming][]
conversation agent called "Script Agent" and add it. Select this agent
(`script-agent`) in your voice pipeline, optionally checking "Prefer handling
commands locally" if you want Home Assistant to try to recognize commands before
sending them to the LLM.

## Options

- `hf_repo`: Hugging Face repository id for the Gemma 4 GGUF model.
- `hf_model`: model filename within `hf_repo` (see [LLM](#llm) below).
- `hf_token`: Hugging Face API token, used for faster model downloads.
- `tool_call_cache_size`: number of recognized sentences to remember (default:
  `100`, see [Tool call caching](#tool-call-caching)).
- `n_ctx`: model context size in tokens (default: `0`, which sizes it
  automatically — see [Context size](#context-size)).
- `n_ctx_overhead`: tokens to reserve beyond the system prompt when `n_ctx` is
  automatic (default: `128`).
- `n_threads`: CPU threads for the model (default: `0`, llama.cpp's own default
  — see [CPU threads](#cpu-threads)).
- `max_tokens`: maximum tokens the model may generate per command (default:
  `128`). Complex scripts with many fields need more output tokens. Raising this
  also raises the automatic context size.
- `flash_attention`: use llama.cpp flash attention (default: enabled). On a
  Raspberry Pi 5 this reduced median recognition latency by about 3.6% in the
  bundled benchmark. The CPU kernels are already part of llama.cpp, so this
  does not add a separate compilation step.
- `debug_logging`: enable verbose logging, including llama.cpp's own output.

Changes to these options take effect on restart.

## LLM

Gemma 4 is run on the CPU using [llama.cpp][] and a [quantized version][] of the
[official model][]. Changing the quantization level, such as from Q5 to Q8, will
change the accuracy, speed, and RAM usage of the agent.

The default model is a 5-bit (Q5) version:

- `hf_repo`: `bartowski/google_gemma-4-E2B-it-GGUF`
- `hf_model`: `google_gemma-4-E2B-it-Q5_K_M.gguf`

If you'd like to try the higher-precision official Q8 quantization, use these
settings:

- `hf_repo`: `ggml-org/gemma-4-E2B-it-GGUF`
- `hf_model`: `gemma-4-E2B-it-Q8_0.gguf`

### Context size

The size of the [llama.cpp][] context (`n_ctx`) is automatically determined based
on the size of the generated tools. If your scripts have large lists of entities
and areas, a larger context size will be needed and increase app RAM usage.
Decreasing the number of [exposed][expose] entities can help keep the context
size small.

### CPU threads

The number of CPU threads used by the model is set with `n_threads` (default:
`0`, which uses llama.cpp's own default). Generation speed is limited by memory
bandwidth rather than CPU, so adding threads past a few cores gives little or no
speedup. Because the model's threads synchronize on every token, a thread that
gets descheduled by other work (such as Home Assistant running on the same
machine) stalls the rest, causing latency spikes under load. On a box shared
with Home Assistant, leave a core or two free (for example, set `n_threads` to
`2` on a 4-core device) for more consistent response times.

## Scripts and selectors

Create scripts and [expose][] them to voice by "More Info -> Settings -> Voice
assistants" and clicking "Expose". You **must** also expose any entities that
you want to be able to refer to by name (or [alias][aliases]). After changing
scripts, exposure, names, or aliases, use **Sync & rebuild** on the Scripts page
to pick up the changes without restarting the app.

Give your scripts descriptive names and a description — the description is what
the model matches a command against, so it matters more than the name. A script
built from a blueprint inherits the blueprint's description if it has none of its
own. The Scripts page shows what the model ends up being told for each script.
Add [fields][] to have the model pass variables to your script. Make sure to add
descriptions!

See the [blueprints][] for example scripts.

The following field [selectors][] are supported:

- Area
    - Uses all available area names and [aliases][]
- Boolean
- Color temperature
- Date
    - The current date is given to the model with each command, so "Saturday"
      and "tomorrow" become real dates
- Date & time
    - As above; the model is told to answer with `YYYY-MM-DDTHH:MM:SS`
- Duration
    - The model fills this in as `HH:MM:SS`, where Home Assistant's own UI gives
      a mapping of `days`/`hours`/`minutes`/`seconds`. Write templates that
      accept both — see the `create_calendar_event` [blueprint][blueprints]
    - Beware that `as_timedelta` reads a two-part `01:00` as *one minute*, not
      one hour
- Entity
    - Uses all [exposed][expose] entity names
    - Add a [domain filter][] to restrict possible entities. Both the current
      `filter` form and the older `domain` shorthand work
- Floor
    - Uses all available floor names and [aliases][]
- Number
    - Set min/max if it makes sense
- RGB color
- Select
- Text
- Time

`Text`, `Select`, `Entity`, `Area`, and `Floor` selectors also support Home
Assistant's `multiple: true` option. Multiple values are passed to the script as
a list. For name-based fields, every name must resolve unambiguously or the
entire field is omitted; a required field then prevents the script from running.

### satellite variable

A special `satellite` variable is passed to each script with information about
the [voice satellite][] that initiated the command. This variable has the
following properties:

- `entity_id` - entity id of the [voice satellite][]
    - Useful for [responding][announce] back with a message
- `area_id` - id of the [area][] where the satellite is located
    - Useful for commands that target the current area
- `floor_id` - id of the [floor][] where the satellite is located
- `device_id` - id of the satellite's device (usually an [ESPHome][esphome] device)
    - Useful if you want to get entities associated with the satellite device
- `media_player_id` - id of the closest [media player][]
    - Search order for media players is satellite device, satellite area, and satellite floor
- `music_player_id` - id of the closest [media player][] that supports [Music Assistant][]
    - Search order for music players is satellite device, satellite area, and satellite floor
- `music_assistant_id` - config entry id of [Music Assistant][]
    - For calling actions like `music_assistant.search`
- `language` - requested response language code
    - May be something like `de`, `en_GB`, or `pt-BR`

## Multiple commands

The model can recognize and run multiple scripts, for example "turn on the lights
and play The Beatles". This works best with larger models, such as the official
Q8 version (see above). Make sure to write your scripts so that more than one
could run at a time!

## State caching

To keep the speed reasonable, the agent caches the LLM state on startup whenever
the scripts or [exposed][expose] entities have changed. Rebuilding the cached
state can take several minutes. The cache is also rebuilt after a model,
llama.cpp, context-size, or flash-attention change. If the saved state is
damaged or cannot be restored, the app rebuilds it automatically.

This rebuild creates a reusable prompt cache; it does not train or fine-tune the
model. Voice requests received during a rebuild return an error immediately
instead of being queued and possibly running much later.

## Tool call caching

If a sentence has been previously recognized, its result will be cached and the
LLM will be skipped next time. The number of cached sentences is controlled by
`tool_call_cache_size` (default: 100). The cache is cleared when the app
restarts.

## Web UI

The **Scripts** page lists every script in Home Assistant, split into targeted
scripts—the only ones the model is told about and the only ones that can
run—and scripts that are not targeted. For each script you see its description,
fields, which fields are required, and how many names a name-based field can
match.

The **Test** page lets you type a command, choose the satellite it came from and
the response language, and see which script it would run and what variables it
would get, including the special `satellite` variable. It does not actually run
the script. This is the quickest way to find out why a command misfires. It
reports three distinct outcomes:

- **Would run** — the script and all its variables resolved.
- **Would not run** — a required field held a name that matches nothing in your
  home, so running the script would do the wrong thing. Expose the entity or
  add an [alias][aliases] for it.
- **The model made this name up** — it produced a tool name that is not an
  exposed script. Usually a sign the exposed scripts don't cover what you asked.

A **Run in Home Assistant** button appears for each **Would run** result. It
executes that exact resolved call once; test results expire after five minutes
and cannot be reused.

Names and fields that don't resolve are always dropped rather than passed to the
script, so a script never receives a value it cannot act on. Optional fields
simply go missing; required fields that are missing, ambiguous, or unresolved
stop the script from running.

The **Settings** page changes the maximum number of tokens the model may
generate for one command. The value is saved in `/data/overrides.yaml`, takes
effect immediately, and grows an automatically sized context when necessary.
The page also shows the effective model, context size, CPU threads, and flash
attention setting.

### Prompts

The same page edits the two prompts, also saved in `/data/overrides.yaml`:

- The **system prompt** comes before the tools and is the cached prefix, so
  changing it rebuilds the prompt cache — the same wait as changing which
  scripts are targeted.
- The **user prompt** wraps each command and is built fresh every time, so
  anything that changes belongs here rather than in the system prompt. It may
  use these placeholders, of which `{text}` is required:

    - `{text}` — the sentence to recognize
    - `{language}` — the requested response language
    - `{date}` — the current date, as `YYYY-MM-DD`
    - `{time}` — the current time, as `HH:MM`
    - `{datetime}` — the current date and time, ISO 8601
    - `{weekday}` — the current day of the week

The default user prompt carries the date, which is how a date or date & time
field gets an actual date out of "for an hour at 2pm on Saturday". Recognized
sentences are cached against the finished prompt, so including `{time}` or
`{datetime}` makes [tool call caching](#tool-call-caching) nearly useless: the
prompt then changes every minute. A longer user prompt also costs time on every
command, since it is evaluated per utterance rather than cached.

**Restore defaults** puts both prompts back. Prompts that match the defaults are
not recorded, so the app keeps following the default if a later version improves
it.

### Choosing which scripts are targeted

Each script has a checkbox. Exposing a script to voice in Home Assistant still
targets it by default, so you only need this page for the exceptions:

- **Untick a targeted script** to keep it out of the tool set without changing
  its exposure in Home Assistant. Useful for a script the model keeps confusing
  with another one.
- **Tick an untargeted script** to add it to the tool set even though it is not
  exposed to voice. This app *runs* scripts, so only do this for scripts you
  want a voice command to be able to trigger — a script you deliberately left
  unexposed is deliberately out of reach. Scripts targeted this way are labelled
  *enabled here*.

Changes are staged until you press **Apply**, which rebuilds the model prefix:
recognition pauses while that happens, from a few seconds on a fast machine to
several minutes on a Raspberry Pi. The page shows how large the context would
become before you commit, since that drives RAM use.

The exceptions are stored in `/data/overrides.yaml` as two lists:

```yaml
scripts:
  enabled: [garage_door]     # not exposed to voice, but targeted
  disabled: [nightly_backup] # exposed to voice, but not targeted
```

Only the exceptions are stored, never the resulting list, so exposing a new
script in Home Assistant is picked up without touching this file. Entries naming
scripts that no longer exist are dropped automatically.

### Syncing from Home Assistant

Home Assistant is read once when the app starts. **Sync & rebuild** on the
Scripts page re-reads it and rebuilds the tools and prompt cache. Use it to pick
up a newly exposed script, a renamed entity, or a new alias without restarting.
Recognition pauses during the rebuild, and incoming requests receive an error
rather than waiting in a queue. The operation does not train or fine-tune the
model.

## Names

Enum values come from Home Assistant's names and aliases. Those are not always
what someone says out loud, and they sometimes include entities whose names only
confuse the model — several entities can even share one name, in which case the
model has no way to pick between them.

The **Names** page lists every entity exposed to Assist, plus every area and
floor, with the names the model may use for each. Edit the comma-separated list
to change them. Use the search bar to filter by ID, name, alias, or entity type;
the Entities, Areas, and Floors sections can each be collapsed independently.

- **Add a name** for something people say a different way ("office light" beside
  "Overhead light").
- **Rename** to disambiguate. If three media players are all called "Media
  Player", the model cannot target any of them reliably; give them distinct
  names.
- **Uncheck Targetable** to keep something out of the enums entirely, so the
  model can never target it. Good for entities that exist but should not be
  voice targets, and for noise like "Built-in Audio Analog Stereo".
- **Reset** goes back to Home Assistant's own names.

### Names shared by more than one thing

Several things can end up with the same name — a "Media Player" in three
bedrooms, or two speakers in one room. The model only ever says the name, so
something has to pick.

The **area of the voice satellite that heard the command** is used first: a
candidate in that area wins, then one on that floor. That resolves the common
case, where the duplicates are in different rooms, without any configuration —
asking the bedroom satellite for "the media player" gets the bedroom one.

When that does not settle it (nothing in that area, no area known, or the
duplicates are in the *same* area) the field is dropped rather than guessed:
acting on whichever one happened to be listed first is worse than not acting.
If the field was required the script does not run, and the tester says which
candidates it was torn between. Giving them distinct names here always fixes it.

Use **Details** beside a name-based field on the Scripts page to inspect its
candidates and see which names need disambiguation.

Nothing in Home Assistant is modified — this only changes what this app offers
the model. If you want a name to work everywhere in Home Assistant, add it as an
[alias][aliases] there instead and use **Sync & rebuild**.

### Names for one script's field

Sometimes a name only needs fixing for one script. Select **Details** beside a
name-based field on the Scripts page to open an editor in a dialog. It lists
exactly that field's candidates after its domain filter, and lets you override
or exclude names without leaving the Scripts page.

Names set there beat the ones on the Names page, for that field only. Everything
else keeps using the global names. Clearing a box excludes that thing from *this*
field while leaving it targetable by other scripts, which is the usual reason to
reach for this: two speakers in one room can be told apart for the script that
needs to, without renaming them everywhere.

Names are stored in the same overrides file, keyed by Home Assistant id, and
only where they differ from Home Assistant:

```yaml
names:
  entity:
    light.office_overhead_light: [Overhead light, office light]
    media_player.built_in_audio_analog_stereo: []   # never offered to the model
  area:
    kitchen: [Kitchen, the cookhouse]
  floor:
    first_floor: [Downstairs]
  per_script:
    device_on_off:
      device_name:
        media_player.voice_media_player: [the test speaker]
```

Entries for entities, areas, floors, or scripts that no longer exist are dropped
automatically, so deleting something in Home Assistant cleans up after itself.

A name that no longer resolves is never passed to a script: see the tester's
"would not run" case above.

The raw tools given to the model (OpenAI function spec) are at `/tools.json`,
for debugging.

A benchmark page is available at `/benchmark` (relative to the ingress URL). It
runs a fixed fixture of tools and sentences (`benchmark.yaml`) against the loaded
model and reports per-sentence latency, throughput (tokens/sec), and correctness
(the parsed tool call vs. the expected one). The fixture is independent of your
exposed scripts so results are reproducible and comparable across machines and
model/config changes. The live model state is snapshotted and restored, so
running a benchmark does not disturb the assistant &mdash; but voice handling
pauses for its duration. While the model is still loading the page says so and
the button stays disabled.

## Docker with GPU

`Dockerfile.gpu` builds llama.cpp with CUDA and runs the model on an NVIDIA GPU.
The host must have the [NVIDIA Container Toolkit][] configured.

From the `script-agent` directory, build and run the image with:

```shell
export HASS_API=http://192.168.1.100:8123/api
docker build --file Dockerfile.gpu --tag script-agent:gpu .
docker run --rm --gpus all \
  --name script-agent \
  --env HASS_TOKEN="${HASS_TOKEN}" \
  --env HASS_API="${HASS_API}" \
  --publish 10500:10500 \
  --publish 5000:5000 \
  --volume script-agent-data:/data \
  script-agent:gpu
```

Set `HASS_TOKEN` to a Home Assistant long-lived access token before running the
command, and replace `192.168.1.100` with the LAN address of your Home Assistant
server. A `.local` mDNS hostname such as `homeassistant.local` may not resolve
inside a Docker container, so use an IP address or a hostname provided by DNS.

### Docker Compose

A Compose service can instead be configured as follows:

```yaml
services:
  script-agent:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    gpus: all
    environment:
      HASS_TOKEN: ${HASS_TOKEN}
      HASS_API: ${HASS_API}
    ports:
      - "10500:10500"
      - "5000:5000"
    volumes:
      - script-agent-data:/data

volumes:
  script-agent-data:
```

Configure Home Assistant's Wyoming integration with the Docker host and port
`10500`.

Every agent CLI option has an uppercase environment variable:
`URI`, `HTTP_HOST`, `HTTP_PORT`, `HASS_TOKEN`, `HASS_API`, `HF_REPO`,
`HF_FILENAME`, `TOOL_CALL_CACHE_SIZE`, `LLAMA_STATE`, `N_CTX`,
`N_CTX_OVERHEAD`, `N_THREADS`, `N_GPU_LAYERS`, `MAX_TOKENS`,
`FLASH_ATTENTION`, `BENCHMARK_FIXTURE`, `OVERRIDES`, and `DEBUG`. The image
defaults `N_GPU_LAYERS` to `-1` to offload all model layers. `HF_TOKEN` may
also be set for authenticated Hugging Face downloads. Boolean variables accept
`true`/`false`, `yes`/`no`, `on`/`off`, or `1`/`0`.


## Benchmarks

Seconds per command with 5 scripts and 35 exposed entities.

- AMD Ryzen 9 5950X - 0.5-1.5 seconds
- Intel Core i5-4570T - 2-3 seconds
- Raspberry Pi 5 - 3-6 seconds
- Home Assistant Green - 15-20 seconds

<!-- Links -->
[conversation]: https://www.home-assistant.io/integrations/conversation/
[gemma4]: https://deepmind.google/models/gemma/gemma-4/
[scripts]: https://www.home-assistant.io/integrations/script/
[expose]: https://www.home-assistant.io/voice_control/voice_remote_expose_devices/
[fields]: https://www.home-assistant.io/integrations/script/#passing-variables-to-scripts
[voice satellite]: https://www.home-assistant.io/integrations/assist_satellite/
[esphome]: https://www.home-assistant.io/integrations/esphome
[announce]: https://www.home-assistant.io/integrations/assist_satellite/#action-announce
[area]: https://www.home-assistant.io/getting-started/concepts-terminology/#areas
[floor]: https://www.home-assistant.io/getting-started/concepts-terminology/#floors
[selectors]: https://www.home-assistant.io/docs/blueprint/selectors/
[aliases]: https://www.home-assistant.io/voice_control/aliases/
[domain filter]: https://www.home-assistant.io/docs/blueprint/selectors/#domain
[llama.cpp]: https://github.com/ggml-org/llama.cpp
[quantized version]: https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF
[official model]: https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF
[media player]: https://www.home-assistant.io/integrations/media_player
[Music Assistant]: https://www.home-assistant.io/integrations/music_assistant/
[NVIDIA Container Toolkit]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
[wyoming]: https://www.home-assistant.io/integrations/wyoming/
[blueprints]: https://github.com/OHF-Voice/apps/tree/main/script-agent/blueprints
