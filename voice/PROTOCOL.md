# The protocol

What Home Assistant and this app say to each other.

Two things are asked of the app: what it can do (`GET /info`) and one run
(a websocket at `/run`). A run is one interaction — audio arrives from the
moment a satellite starts streaming until the stream ends — and the app is
sent that audio whether or not there is a wake word to verify, because
endpointing, and later speaker identification, need the same stream.

## `GET /info`

```json
{"name": "Voice", "version": "1.0", "api_version": 1,
 "wake_words": [{"id": "okay_nabu", "phrase": "okay nabu"}],
 "max_run_ms": 30000}
```

`wake_words` is a standing decision, not a per-detection one: a satellite told
to send wake word audio detects at a lower threshold, trusting that something
downstream will catch what that lets through, so Home Assistant has to know
before a detection whether anything will. An app that reports none is still
sent audio, and still endpoints.

`max_run_ms` is how much audio the app will hold. Home Assistant does not need
it to function, but a ceiling that only shows up as silence is a ceiling
nobody can debug.

## One run

```
HA  -> text    {"type": "run-start", ...}
HA  -> binary  16 kHz 16-bit mono PCM, continuously
HA  -> text    {"type": "audio-end"}
```

One websocket per run. Audio frames have nowhere to carry a run id, so a
shared connection would mean tagging every chunk of every satellite's audio;
a connection on the Supervisor's network costs a handshake, and a run that
wedges then takes nothing else with it.

`run-start`:

```json
{"type": "run-start",
 "satellite_id": "assist_satellite.kitchen",
 "wake_word_phrase": "okay nabu",
 "wake_word_audio_ms": 1500,
 "silence_seconds": 0.7}
```

The wake word fields are absent when there is no wake word stage — a button
press, or a satellite that starts at speech-to-text — and the app then only
endpoints. `silence_seconds` is the user's VAD sensitivity, already resolved
to seconds by Home Assistant; `null` means do not endpoint at all, which is
what a run with voice activity detection switched off looks like.

The audio format is fixed. There is no resampling in the app, and a format
field would only suggest otherwise.

## What the app says back

```json
{"type": "wake-word-end",
 "wake_word_output": {"wake_word_phrase": "okay nabu",
                      "wake_word_audio_ms": 1480}}

{"type": "wake-word-rejected",
 "wake_word_output": {"wake_word_phrase": "okay nabu", "reason": "last"}}

{"type": "stt-vad-start", "timestamp": 1840}

{"type": "stt-vad-end", "timestamp": 4120}

{"type": "error", "code": "...", "message": "..."}
```

These are Home Assistant's own pipeline events, with their own names and their
own fields, and they are passed through rather than translated. The split is
not a new idea needing a new message: `wake_word_audio_ms` in `wake-word-end`
is already where speech-to-text starts. `stt-vad-start` and `stt-vad-end` are
what satellites animate on today, so an app that endpoints without sending
them would leave devices looking dead while someone talks to them.

`timestamp` is milliseconds from the first audio byte of the run, which is the
same clock Home Assistant's chunks already carry.

A rejection ends the run. The app sends nothing else and Home Assistant closes
the connection.

The app does not send `stt-start` or `stt-end`. Those are the speech-to-text
stage, and the app is not running speech-to-text.

Ordering is the obvious one: `wake-word-end` before `stt-vad-start` before
`stt-vad-end`. Reaching `max_run_ms` ends the run with an ordinary
`stt-vad-end`, and the app writes a line to its log saying that is what
happened — Home Assistant is told where the boundary is, not why.

## Reading audio is not analysing it

Audio arrives for the whole run while a verification holds the decoder for
roughly 120 ms. Draining the socket and analysing what was drained have to be
separate tasks inside the app, into a buffer bounded by `max_run_ms`; an app
that reads only as fast as it decodes stalls the socket and grows a queue
behind it in Home Assistant. The app keeps draining to `audio-end` even after
it has stopped having opinions, so the connection always closes cleanly.

## Who ends the stream

The app is authoritative on endpointing while it is connected. Home Assistant
keeps its own segmenter constructed but idle, and the first end from any
source wins — the app, a speech-to-text engine that cuts off early, or the
input stream running out.

Every event has a deadline, and every deadline fails open. No wake word
verdict in time and the detection stands as the satellite reported it; no
`stt-vad-start` and the satellite's own `wake_word_audio_ms` is the boundary;
no `stt-vad-end` and Home Assistant's segmenter takes over. The app is in the
path of every run now, not only of detections, so one wedged app must not be
able to end every interaction. Wrongly accepting costs one unwanted command;
wrongly refusing costs an assistant that stops answering its name.

## Timestamps refer to the audio Home Assistant sent

That is what makes reporting boundaries enough. It stops being enough the day
the app cleans audio, because then nobody else has the audio the boundaries
describe — and at that point the app is responsible for the audio reaching
speech-to-text, which is a new message rather than a quiet change of meaning
here.

## Left out on purpose

Speaker identification, audio cleanup and the stream it would imply, and
anything from the audio-capable `conversation` entities. Also two channels of
audio: ESPHome devices with multi-channel audio send an enhanced and a
non-enhanced stream, and Home Assistant already throws one away before the
pipeline sees it, so giving the app both is a pipeline change rather than a
protocol one. Gain the app can do itself; noise reduction it cannot, and
whether that costs anything is a measurement nobody has made.

`api_version` is how any of those arrive.
