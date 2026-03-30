# Default Agent

[Wyoming][wyoming] protocol [conversation agent][conversation] that replaces the default [Assist behavior][assist].

## Installation

After installing the app, a `default-agent` agent should automatically be discovered via the [wyoming][] integration.

For timers to work, you will need to install a [HACS integration][mike-voice-hacs] that adds the appropriate websocket commands.

## Usage

Once installed, create a voice assistant with the discovered conversation agent and ensure that "Prefer local intents" is off.

### Custom Sentences

Add your [custom sentences][] to `/share/default-agent/custom_sentences/<language>`.

<!-- Links -->
[wyoming]: https://www.home-assistant.io/integrations/wyoming/
[conversation]: https://www.home-assistant.io/integrations/conversation
[assist]: https://www.home-assistant.io/voice_control
[mike-voice-hacs]: https://github.com/synesthesiam/mike-voice-hacs
[custom sentences]: https://github.com/OHF-Voice/default-agent?tab=readme-ov-file#custom-sentences
