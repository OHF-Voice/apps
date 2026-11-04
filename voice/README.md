# Voice

A second opinion on the wake words your voice satellites hear.

A satellite's wake word detector is a small classifier running on every frame
of audio, on a microcontroller, tuned to fire cheaply — because a device that
misses its name is worse than one that occasionally answers the television.
This app is the other half of that trade: Home Assistant sends it the audio
the detector fired on, and it says whether that was really the wake word
before the command is acted on.

It has no interface and nothing to configure. Install it, and Home Assistant
finds it.

## What it needs from Home Assistant

Verification only happens when all of this is true, and nothing breaks when
it isn't:

* the satellite sends the audio it buffered before the detection — on ESPHome
  devices, wake word audio;
* the app has a model for that wake word (`GET /info` says which);
* the `voice_support` integration is set up, which the Supervisor arranges when
  the app starts.

Otherwise the detection stands exactly as the device reported it.

## How a verification goes

```
HA  -> text    {"type": "verify", "wake_word_phrase": "okay nabu", "wake_word_audio_ms": 1000, ...}
HA  -> binary  16 kHz 16-bit mono PCM
HA  -> text    {"type": "audio-end", "final": false}
app -> text    {"type": "verdict", "accepted": true, "wake_word_audio_ms": 940, ...}
```

One websocket per detection, on `/verify`. An app may answer `audio-end` with
`{"type": "more-audio", "ms": 300}` instead of a verdict, and Home Assistant
sends more of what the satellite is streaming. That exists because the useful
amount of audio *after* the detection is not a constant — see "The retry"
below.

`wake_word_audio_ms` is where the wake phrase really ends, and is sent only with
`--split kaldi`. By default it is absent and Home Assistant keeps the boundary the
satellite reported.

The alignment is real, but it is not yet a better boundary to cut at: the
experimental app measured Kaldi's end of the wake phrase landing a median
0.3-0.4 s after a CTC placement of the same audio, which is far enough to take
the first word of the command with it (76.2% of commands correct against 89.6%).
It is always reported in `detail.split_ms`, so a placement can be compared
against it without changing what the pipeline cuts at.

## The verifier

Kaldi, vendored in [voice/kaldi](voice/kaldi): four grammar-constrained
decodes of one acoustic model — the wake phrase, free background speech, the phrase minus its
last word, and the phrase's known confusables. The wake word is accepted when
the phrase beats background on its prefix *and* on its last word, and no
confusable beats it by much. Those are three separate decisions and they fail
separately: "okay now" loses on the last word, "okay naruto" on the confusable
margin.

`wake.yaml` is where the phrases, confusables, pronunciations and thresholds
live, with the measurements behind each number.

The code is a copy of [wyoming-wake-kaldi][]'s, not a dependency on it. That
project is a Wyoming server, and this app does not speak Wyoming to it -- it
reached past the server into internals nobody promised to keep. Dropping the
server left code that imports nothing but the standard library and PyYAML, so
there is no package to install and no Wyoming to carry. A copy under a name of
its own also cannot be shadowed by an installed or editable copy of the
original, which is a failure that looks like a wrong answer rather than an
import error. `voice/kaldi/__init__.py` says what was changed.

### The retry

A detector that fires before the speaker has finished leaves the last phoneme
outside the window, and the thresholds are then applied to a truncated phrase.
When the alignment puts the end of the phrase hard against the end of the
window, that is what happened, and the window slides forward over audio from
after the detection.

Only then. Measured across four speakers in the experimental app, sliding
*always* cost two of them (-1.4 and -2.8 points) what it gained the one being
truncated; sliding only on a pinned split gained +3.8 and +0.7 and took nothing
from anyone.

## Failing open

Every failure path accepts the detection: an app that is down, slow, confused,
or has no model for the wake word. Wrongly accepting costs one unwanted
command; wrongly refusing costs a device that no longer answers its name. Only
a verifier that ran and said no refuses.

## Building

The Kaldi and OpenFST binaries and the acoustic model are ~255 MB that no
package index serves, so they are staged before the image is built:

```bash
script/stage_build [path to a wyoming-wake-kaldi checkout]
```

The decoder daemon is built from the vendored source against those binaries.
Without it every decode spawns its own Kaldi process, which works: on one
window of "okay nabu", 42 ms with the daemon against roughly 200 ms without.

```bash
bash voice/kaldi/decoderd/build.sh "$PWD/local/wake-kaldi-tools"
```

An image built without any of this still builds and still runs. It reports no
wake words, and Home Assistant never asks it anything.

## Running it while working on it

The app is an ordinary server, so it does not have to be an app to be used.
Run it from a checkout:

```bash
python3 -m voice --port 10800 --wake-config wake.yaml --train-dir var/wake-train
```

and add it in Home Assistant from **Settings → Devices & services → Add
integration → Voice**, which asks for a host and port when there is no
Supervisor to install an app from. Without `local/` staged it will start and
report no wake words, which is enough to exercise everything but the decode.

For a development tree, `local/wake-kaldi-tools` and `local/wake-kaldi-models`
can be symlinks into a wyoming-wake-kaldi checkout rather than copies.
`script/stage_build` makes real copies, which is what a Docker build needs.

## Tests

```bash
python3 -m pytest
```

[wyoming-wake-kaldi]: https://github.com/OHF-Voice/wyoming-wake-kaldi
