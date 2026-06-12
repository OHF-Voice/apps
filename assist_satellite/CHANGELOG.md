# Changelog

## 1.1.11

- Update to linux-voice-assistant [1.1.11](https://github.com/OHF-Voice/linux-voice-assistant/releases/tag/v1.1.11)
- Fix executable permissions [#4](https://github.com/OHF-Voice/apps/pull/4)

## 1.0.0

### Linux Voice Assistant App ([linux voice assistant](https://github.com/OHF-Voice/linux-voice-assistant))

- Initial release of the Linux Voice Assistant App
- Uses the ESPHome satellite protocol for seamless integration with Home Assistant
- Automatic discovery via ESPHome integration using built-in mDNS/zeroconf
- Local wake word detection using microWakeWord and openWakeWord models
- Dual assistant support with selectable wake words
- Configurable wake word model, audio devices, refractory period, and thinking sound
- Start and continue conversation features
- Stop word support for announcements and timers
- Automatic timer stop after the set duration (default: 15 mins)
- Persistent preferences (active wake word, volume) across restarts in `/share/assist_satellite/preferences.json`
- Support for custom wake word models downloaded from Home Assistant, stored in `/share/assist_satellite/local`
- Satellite name, mute switch, and thinking sound toggle exposed as entities in Home Assistant
- New media player entity with volume control for both satellite and media player

