# Voice

App for Home Assistant [voice control](https://www.home-assistant.io/voice_control/).

This app is a work in progress, so expect things to be unpolished!

## Installation and first boot

Installing the app can take quite a while, since it builds an optimized version of [llama.cpp] for your CPU.

On first boot of the app, the [LLM model](#conversation) must be downloaded (about 4GB). If you have a HuggingFace account, putting your token in the app settings (`hf_token`) may speed up the download.

## Conversation

An LLM [conversation][] agent built on [Gemma 4][gemma4] is used to recognize [intents][] from voice commands.

The Gemma 4 LLM is run on the CPU using [llama.cpp][] using a [quantized version][] of the [official model][]. Changing the quantization level (like Q5 to Q8) will change the accuracy, speed, and RAM usage of the agent.

The default model is a 5-bit (Q5) version:

- repo: `bartowski/google_gemma-4-E2B-it-GGUF`
- model: `google_gemma-4-E2B-it-Q5_K_M.gguf`

If you'd like to try the larger official model, use these settings:

- repo: `ggml-org/gemma-4-E2B-it-GGUF`
- model: `gemma-4-E2B-it-Q8_0.gguf`

### Supported voice commands

Voice commands are implemented as [tools][] given to the LLM. When given a voice command, the LLM must select a tool to call and the appropriate parameters.

An important setting for tools is `include_names_in_tools`, which is enabled by default. This injects your [exposed entities][] and area/floor names into the tools themselves, allowing the LLM to have more context. This includes [aliases][] as well. While more accurate, this makes processing slower and increases RAM usage.

| Command                            | Example                                        |
|------------------------------------|------------------------------------------------|
| cancel                             | nevermind                                      |
| time                               | what time is it?                               |
| date                               | what's the date?                               |
| weather                            | what's the weather forecast?                   |
| temperature                        | what's the thermostat set to?                  |
| start timer                        | set a timer for 10 minutes                     |
| pause timer                        | pause timer                                    |
| resume timer                       | resume timer                                   |
| cancel timer                       | cancel timer                                   |
| timer status                       | how much time is left on my timer?             |
| lights on/off (current area)       | turn on/off the lights                         |
| lights on/off (area/floor)         | turn on/off lights in the kitchen              |
| light brightness (current)         | set brightness to 50%                          |
| light brightness (name)            | set ceiling light brightness to 50%            |
| light brightness (area/floor)      | set brightness in the kitchen to 50%           |
| light color (current)              | set lights to blue                             |
| light color (name)                 | set ceiling lights to blue                     |
| light color (area/floor)           | set lights in the kitchen to blue              |
| device on/off                      | turn on/off the TV                             |
| open/close                         | open/close garage door                         |
| cover position                     | set pergola roof to 50%                        |
| lock/unlock                        | lock/unlock front door                         |
| fan speed                          | set ceiling fan to 50%                         |
| media pause (current area)         | pause the music                                |
| media pause (area/floor)           | pause music in the kitchen                     |
| media pause (name)                 | pause music on smart speaker                   |
| media resume (current area)        | resume music                                   |
| media resume (area/floor)          | resume music in the kitchen                    |
| media resume (name)                | resume music on smart speaker                  |
| media next (current area)          | next track                                     |
| media next (area/floor)            | next track in the kitchen                      |
| media next (name)                  | next track on smart speaker                    |
| volume up/down (current area)      | increase/decrease volume                       |
| volume up/down (area/floor)        | increase/decrease volume in the kitchen        |
| volume up/down (name)              | increase/decrease smart speaker volume         |
| volume up/down by % (current area) | increase/decrease volume by 20%                |
| volume up/down by % (area/floor)   | increase/decrease volume by 20% in the kitchen |
| volume up/down by % (name)         | increase/decrease smart speaker volume by 20%  |
| set volume (current area)          | set volume to 20%                              |
| set volume (area/floor)            | set volume to 20% in the kitchen               |
| set volume (name)                  | set smart speaker volume to 20%                |
| play music (current area)          | play the beatles                               |
| play music (area/floor)            | play the beatles in the kitchen                |
| play music (name)                  | play the beatles on the smart speaker          |
| add todo item                      | put clean the garage on my todo list           |
| complete todo item                 | check off clean the garage on my todo list        |
| remove todo item                   | remove clean the garage from my todo list      |

**Notes**

- Turning devices on/off only works for lights, switches, fans, media players, and input booleans
- Open/close commands work with covers and valves
- Some commands, like setting a cover's position or pausing a media player, require devices that explicitly support those features
- Playing music requires a Music Assistant media player that is also exposed to Assist
- Todo list names match what's in Home Assistant, so you can just name one "shopping list"

### Multiple commands

Gemma 4 can recognize and run multiple voice commands, such as "turn off the lights and lock the front door". This works best with larger models, such as the official Q8 version (see above).

### State caching

In order to keep the LLM speed reasonable, the agent caches the LLM state's whenever the tools change. If you have `include_names_in_tools` enabled (the default), this will require rebuilding the cached state whenver you modify your [exposed entities][] and restart the app.

Rebuilding the cached state can take several minutes.


<!-- Links -->
[conversation]: https://www.home-assistant.io/integrations/conversation/
[gemma4]: https://deepmind.google/models/gemma/gemma-4/
[intents]: https://developers.home-assistant.io/docs/intent_builtin/
[llama.cpp]: https://github.com/ggml-org/llama.cpp
[quantized]: https://en.wikipedia.org/wiki/Large_language_model#Quantization
[quantization version]: https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF
[official model]: https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF
[tools]: https://developers.openai.com/api/docs/guides/function-calling
[exposed entities]: https://www.home-assistant.io/voice_control/voice_remote_expose_devices/
[aliases]: https://www.home-assistant.io/voice_control/aliases/
