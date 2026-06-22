# Home Assistant App: Assist Satellite

![Supports aarch64 Architecture][aarch64-shield] ![Supports amd64 Architecture][amd64-shield]

Home Assistant app (formerly known as add-on) that turns the host machine into a [voice satellite](https://www.home-assistant.io/voice_control/) with [media player](https://www.home-assistant.io/integrations/media_player/) using a connected microphone and speaker.

Depends on [Linux Voice Assistant](https://github.com/OHF-Voice/linux-voice-assistant) runtime via [ESPHome](https://esphome.io/) protocol.

Part of the [Year of Voice](https://www.home-assistant.io/blog/2022/12/20/year-of-voice/).

### Installation of Assist Satellite app on Home Assistant OS

> [!NOTE]
> For now you first have to add the [OHF-Voice apps](https://github.com/OHF-Voice/apps) repo manually to the App Store repositroy inside Home Assistant Operating System before you can install it.

Later you will be able to install it directly from the official add-on repository (but it is not yet published publicly there):

[![Add repository to your Home Assistant instance.](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https://github.com/home-assistant/addons)

Once installed, the satellite is automatically discovered by Home Assistant via the ESPHome integration.

[aarch64-shield]: https://img.shields.io/badge/aarch64-yes-green.svg
[amd64-shield]: https://img.shields.io/badge/amd64-yes-green.svg
