from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import s2p_intents  # noqa: E402


class SentenceOverrideTests(unittest.TestCase):
    def test_loads_tagged_upstream_yaml_and_preserves_package_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            override = (
                Path(temp_dir) / "en" / "HassMediaPause" / "default.yaml"
            )
            override.parent.mkdir(parents=True)
            override.write_text(
                """
language: en
expansion_rules:
  action: "<verb> the music"
  verb: hold
data:
  - sentences:
      - rich package phrasing
  - sentences:
      - "<action>"
      - "<verb> playback <the>"
    expansion_rules:
      verb: stop
    response: ignored_override_response
    speech_to_phrase: true
""".strip()
            )

            loaded = s2p_intents._load_sentence_overrides(Path(temp_dir))
            original = s2p_intents.combo_blocks(
                "en", "HassMediaPause", "default"
            )
            self.assertTrue(original)

            old_overrides = s2p_intents._SENTENCE_OVERRIDES
            try:
                s2p_intents._SENTENCE_OVERRIDES = loaded
                s2p_intents._combo_map.cache_clear()
                patched = s2p_intents.combo_blocks(
                    "en", "HassMediaPause", "default"
                )
            finally:
                s2p_intents._SENTENCE_OVERRIDES = old_overrides
                s2p_intents._combo_map.cache_clear()

            self.assertEqual(
                patched[0]["sentences"],
                ["stop the music", "stop playback <the>"],
            )
            self.assertEqual(patched[0]["response"], original[0]["response"])
            self.assertEqual(
                patched[0]["context_area"], original[0]["context_area"]
            )

    def test_rejects_language_that_does_not_match_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            override = (
                Path(temp_dir) / "en" / "HassMediaPause" / "default.yaml"
            )
            override.parent.mkdir(parents=True)
            override.write_text(
                "language: de\ndata:\n  - sentences: [pause]\n"
            )

            with self.assertRaisesRegex(
                ValueError, "must declare language: en"
            ):
                s2p_intents._load_sentence_overrides(Path(temp_dir))

    def test_rejects_circular_expansion_rules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            override = (
                Path(temp_dir) / "en" / "HassMediaPause" / "default.yaml"
            )
            override.parent.mkdir(parents=True)
            override.write_text(
                """
language: en
expansion_rules:
  first: "<second>"
  second: "<first>"
data:
  - sentences: ["<first>"]
""".strip()
            )

            with self.assertRaisesRegex(ValueError, "Circular expansion rules"):
                s2p_intents._load_sentence_overrides(Path(temp_dir))

    def test_preserves_hassil_slots_and_escaped_literals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            override = (
                Path(temp_dir) / "en" / "HassMediaPause" / "default.yaml"
            )
            override.parent.mkdir(parents=True)
            override.write_text(
                r"""
language: en
expansion_rules:
  target: '{timer_hours:hours} \(remaining\)'
data:
  - sentences: ["pause <target> [now|please]"]
""".strip()
            )

            loaded = s2p_intents._load_sentence_overrides(Path(temp_dir))
            sentence = loaded["en"][("HassMediaPause", "default")][0][0]
            self.assertEqual(
                sentence,
                r"pause {timer_hours:hours} \(remaining\) [now|please]",
            )
            # Canonical output must remain valid input for Hassil.
            s2p_intents.parse_sentence(sentence)

    def test_ignores_empty_placeholder_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            override = (
                Path(temp_dir) / "en" / "HassMediaPause" / "default.yaml"
            )
            override.parent.mkdir(parents=True)
            override.touch()

            self.assertEqual(
                s2p_intents._load_sentence_overrides(Path(temp_dir)), {}
            )


if __name__ == "__main__":
    unittest.main()
