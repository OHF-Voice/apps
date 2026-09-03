#!/usr/bin/env python3
"""Generate one TTS FLAC for every Speech-to-Phrase template combination."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import language_test_common as common
import numpy as np
import soundfile as sf
from speech_to_phrase.audio import SAMPLE_RATE

DEFAULT_TTS_URL = "http://localhost:8000/api/text-to-speech"


def synthesize(url: str, language: str, text: str) -> np.ndarray:
    query = urllib.parse.urlencode({"language": language})
    request = urllib.request.Request(
        f"{url}?{query}",
        data=text.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        content_type = response.headers.get_content_type()
        if content_type != "audio/wav":
            raise RuntimeError(
                f"TTS returned {content_type!r} for {language}: {text!r}"
            )
        audio_bytes = response.read()

    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes), dtype="float32", always_2d=False
    )
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return common.resample_audio(audio, sample_rate, SAMPLE_RATE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        action="append",
        choices=common.languages(),
        help="Generate only this language (repeatable; defaults to all)",
    )
    parser.add_argument(
        "--case",
        action="append",
        metavar="INTENT/COMBO",
        help="Generate only this intent/combination (repeatable)",
    )
    parser.add_argument("--tts-url", default=DEFAULT_TTS_URL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected_languages = args.language or common.languages()
    for language in selected_languages:
        corpus = common.enumerate_examples(language)
        if args.case:
            requested = set(args.case)
            corpus = [
                item
                for item in corpus
                if f"{item['intent']}/{item['combo']}" in requested
            ]
            found = {f"{item['intent']}/{item['combo']}" for item in corpus}
            if missing := requested - found:
                parser.error(
                    f"{language} does not contain: {', '.join(sorted(missing))}"
                )
        output_dir = common.WAV_ROOT / language
        output_dir.mkdir(parents=True, exist_ok=True)
        pending = [
            item
            for item in corpus
            if args.force or not (output_dir / item["wav"]).exists()
        ]
        print(
            f"{language}: {len(corpus)} templates, "
            f"{len(pending)} FLACs to synthesize"
        )
        for index, item in enumerate(pending, start=1):
            tts_text = common.synthesis_text(item)
            audio = synthesize(args.tts_url, language, tts_text)
            output_path = output_dir / item["wav"]
            with tempfile.NamedTemporaryFile(
                dir=output_dir, suffix=".flac", delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
            try:
                sf.write(
                    temp_path,
                    audio,
                    SAMPLE_RATE,
                    format="FLAC",
                    subtype="PCM_16",
                )
                temp_path.replace(output_path)
            finally:
                temp_path.unlink(missing_ok=True)
            print(
                f"  [{index:02d}/{len(pending):02d}] "
                f"{item['intent']}/{item['combo']}: {item['text']}"
            )

        if not args.case:
            common.manifest_path(language).write_text(
                json.dumps(corpus, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
