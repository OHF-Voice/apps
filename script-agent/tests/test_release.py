import argparse
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from app import _log_arguments
from const import APP_NAME, APP_SLUG, APP_VERSION
from gemma4_recognizer import DEFAULT_MAX_TOKENS


class ReleaseMetadataTests(unittest.TestCase):
    def setUp(self):
        self.project_dir = Path(__file__).parent.parent
        self.config = yaml.safe_load(
            (self.project_dir / "config.yaml").read_text("utf-8")
        )

    def test_config_identity_matches_runtime_identity(self):
        self.assertEqual(APP_NAME, self.config["name"])
        self.assertEqual(APP_SLUG, self.config["slug"])

    def test_config_version_matches_runtime_version(self):
        self.assertEqual(APP_VERSION, self.config["version"])

    def test_changelog_documents_the_current_version(self):
        # The version is written in three places: config.yaml, const.py, and the
        # changelog's newest heading. The first two are checked above; this is
        # the copy that is easy to leave behind when the number changes.
        changelog = (self.project_dir / "CHANGELOG.md").read_text("utf-8")
        versions = re.findall(r"^##\s+(\S+)", changelog, flags=re.MULTILINE)

        self.assertTrue(versions, "CHANGELOG.md has no version headings")
        self.assertEqual(APP_VERSION, versions[0])

    def test_flash_attention_is_enabled_by_default(self):
        self.assertIs(True, self.config["options"]["flash_attention"])

    def test_generation_limit_matches_runtime_default(self):
        self.assertEqual(DEFAULT_MAX_TOKENS, self.config["options"]["max_tokens"])

    def test_release_uses_current_repository(self):
        expected_url = "https://github.com/OHF-Voice/apps"
        self.assertTrue(self.config["url"].startswith(expected_url))

        for filename in ("README.md", "DOCS.md"):
            contents = (self.project_dir / filename).read_text("utf-8")
            self.assertIn(expected_url, contents)
            self.assertNotIn("OHF-Voice/apps-experimental", contents)

    def test_docker_stages_use_trixie(self):
        dockerfile = (self.project_dir / "Dockerfile").read_text("utf-8")
        from_lines = [
            line.split() for line in dockerfile.splitlines() if line.startswith("FROM ")
        ]

        self.assertEqual(2, len(from_lines))
        self.assertTrue(
            all(
                stage[1] == "ghcr.io/home-assistant/base-debian:trixie"
                for stage in from_lines
            )
        )
        self.assertIn("GGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod", dockerfile)

    def test_gpu_dockerfile_enables_cuda_and_exposes_cli_environment(self):
        dockerfile = (self.project_dir / "Dockerfile.gpu").read_text("utf-8")
        entrypoint = (self.project_dir / "docker-entrypoint").read_text("utf-8")
        image_sources = dockerfile + entrypoint

        self.assertIn("-DGGML_CUDA=ON", dockerfile)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=all", dockerfile)
        for variable in (
            "SCRIPT_AGENT_URI",
            "SCRIPT_AGENT_HTTP_HOST",
            "SCRIPT_AGENT_HTTP_PORT",
            "SCRIPT_AGENT_HASS_TOKEN",
            "SCRIPT_AGENT_HASS_API",
            "SCRIPT_AGENT_HF_REPO",
            "SCRIPT_AGENT_HF_FILENAME",
            "SCRIPT_AGENT_TOOL_CALL_CACHE_SIZE",
            "SCRIPT_AGENT_LLAMA_STATE",
            "SCRIPT_AGENT_N_CTX",
            "SCRIPT_AGENT_N_CTX_OVERHEAD",
            "SCRIPT_AGENT_N_THREADS",
            "SCRIPT_AGENT_N_GPU_LAYERS",
            "SCRIPT_AGENT_MAX_TOKENS",
            "SCRIPT_AGENT_FLASH_ATTENTION",
            "SCRIPT_AGENT_BENCHMARK_FIXTURE",
            "SCRIPT_AGENT_OVERRIDES",
            "SCRIPT_AGENT_DEBUG",
        ):
            self.assertTrue(
                (f"{variable}=" in image_sources)
                or (f"${{{variable}}}" in image_sources)
                or (f"${{{variable}:" in image_sources),
                variable,
            )

    def test_python_environment_matches_app_name(self):
        python_environment = (self.project_dir / ".python-version").read_text("utf-8")
        self.assertEqual("script-agent", python_environment.strip())

    def test_debug_arguments_redact_home_assistant_token(self):
        with patch("app._LOGGER") as logger:
            _log_arguments(argparse.Namespace(hass_token="super-secret", debug=True))

        logged = repr(logger.debug.call_args)
        self.assertNotIn("super-secret", logged)
        self.assertIn("<redacted>", logged)


if __name__ == "__main__":
    unittest.main()
