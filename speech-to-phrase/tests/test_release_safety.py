import hashlib
import os
import platform
import sys
import threading
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from vendored_lib import bind as _bind_vendored_lib  # noqa: E402

_bind_vendored_lib()

from speech_to_phrase.backends.coqui import CoquiModel  # noqa: E402

import app as web_app  # noqa: E402
import models  # noqa: E402
import settings  # noqa: E402


def test_default_english_model_is_bundled_parakeet():
    model_name = models.model_name_for("en", "nemo")

    assert model_name == "stt_en_parakeet_tdt_ctc_110m"
    assert models.default_max_score("nemo", model_name) == 3.8
    assert (
        models.default_max_score("nemo", ROOT / "local/models/parakeet-tdt-ctc-110m")
        == 3.8
    )
    assert models.default_max_score("nemo", "stt_de_citrinet_1024") == 5.0
    assert models.resolve_backend("en", "citrinet") == "nemo"
    assert models.model_name_for("en", "citrinet") == model_name
    assert models.default_token_bonus("citrinet") == 2.0
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert f"ARG BUNDLE_MODEL={model_name}" in dockerfile


def test_legacy_english_default_gate_is_migrated(tmp_path):
    settings.set_max_score(tmp_path, "en", 5.0)

    assert settings.migrate_max_score_default(
        tmp_path,
        "en",
        model_id=models.ENGLISH_MODEL,
        previous_model_ids=models.LEGACY_ENGLISH_MODELS,
        previous_default=5.0,
        new_default=3.8,
    )
    assert settings.load(tmp_path, "en") == {
        "max_score": 3.8,
        "max_score_model": models.ENGLISH_MODEL,
    }

    settings.set_max_score(
        tmp_path,
        "en",
        5.0,
        model_id=models.ENGLISH_MODEL,
    )
    assert not settings.migrate_max_score_default(
        tmp_path,
        "en",
        model_id=models.ENGLISH_MODEL,
        previous_model_ids=models.LEGACY_ENGLISH_MODELS,
        previous_default=5.0,
        new_default=3.8,
    )
    assert settings.get_max_score(tmp_path, "en", 3.8) == 5.0


def test_custom_legacy_gate_is_preserved_during_model_migration(tmp_path):
    settings.set_max_score(tmp_path, "en", 4.5)

    assert not settings.migrate_max_score_default(
        tmp_path,
        "en",
        model_id=models.ENGLISH_MODEL,
        previous_model_ids=models.LEGACY_ENGLISH_MODELS,
        previous_default=5.0,
        new_default=3.8,
    )
    assert settings.load(tmp_path, "en") == {
        "max_score": 4.5,
        "max_score_model": models.ENGLISH_MODEL,
    }


def _cfg(data: Path) -> Namespace:
    return Namespace(
        data=str(data),
        language="en",
        hass_api="http://home-assistant",
        hass_token="token",
        entities_file=None,
        slot_lists_file=None,
    )


def test_registry_fetches_use_last_known_good_cache(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    records = [{"name": "real light", "domain": "light"}]
    areas_floors = (["Kitchen"], ["Ground floor"])

    monkeypatch.setattr(
        web_app.training, "entity_records_from_hass", lambda *_: records
    )
    monkeypatch.setattr(
        web_app.training, "areas_floors_from_hass", lambda *_: areas_floors
    )
    assert web_app._raw_records(cfg) == records
    assert web_app._raw_slot_lists(cfg)["area"] == ["Kitchen"]

    def unavailable(*_args):
        raise OSError("Home Assistant unavailable")

    monkeypatch.setattr(web_app.training, "entity_records_from_hass", unavailable)
    monkeypatch.setattr(web_app.training, "areas_floors_from_hass", unavailable)
    assert web_app._raw_records(cfg) == records
    cached_lists = web_app._raw_slot_lists(cfg)
    assert cached_lists["area"] == ["Kitchen"]
    assert cached_lists["floor"] == ["Ground floor"]


def test_registry_fetch_does_not_fall_back_to_dev_fixtures(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)

    def unavailable(*_args):
        raise OSError("Home Assistant unavailable")

    monkeypatch.setattr(web_app.training, "entity_records_from_hass", unavailable)
    monkeypatch.setattr(web_app.training, "areas_floors_from_hass", unavailable)
    with pytest.raises(web_app.HomeAssistantUnavailable):
        web_app._raw_records(cfg)
    with pytest.raises(web_app.HomeAssistantUnavailable):
        web_app._raw_slot_lists(cfg)


class _FakeProcess:
    def __init__(self):
        self.stdin = self
        self.stdout = self
        self._guard = threading.Lock()
        self._owner = None
        self._line = 0

    def poll(self):
        return None

    def write(self, data):
        first_write = False
        owner = threading.get_ident()
        with self._guard:
            if self._owner is None:
                self._owner = owner
                first_write = True
            elif self._owner != owner:
                raise AssertionError("concurrent protocol transactions interleaved")
        if first_write:
            time.sleep(0.03)
        return len(data)

    def flush(self):
        return None

    def readline(self):
        with self._guard:
            if self._line == 0:
                self._line = 1
                return b"0.5 0.5\n"
            self._line = 0
            self._owner = None
            return b"\n"


def test_coqui_serializes_shared_subprocess_transactions():
    model = CoquiModel.__new__(CoquiModel)
    model._proc = _FakeProcess()
    model._num_classes = 2
    model._lock = threading.Lock()
    barrier = threading.Barrier(2)

    def recognize():
        barrier.wait()
        return model.log_probs(np.zeros(160, dtype=np.float32))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(recognize) for _ in range(2)]
        results = [future.result() for future in futures]
    assert all(result.shape == (1, 2) for result in results)


def test_stt_binary_is_pinned_and_verified(monkeypatch, tmp_path):
    name = "stt_onlyprobs.x86_64.bin"
    payload = b"verified test helper"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / name
    target.write_bytes(b"corrupt")

    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setitem(models.STT_BINARY_SHA256, name, digest)

    def download(url, destination):
        assert models.HF_REVISION in url
        Path(destination).write_bytes(payload)

    monkeypatch.setattr(models.urllib.request, "urlretrieve", download)
    assert models.ensure_stt_binary(tmp_path) == target
    assert target.read_bytes() == payload
    assert os.access(target, os.X_OK)
    assert os.environ["STT_ONLYPROBS"] == str(target)


def test_stt_binary_rejects_bad_digest(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        models.urllib.request,
        "urlretrieve",
        lambda _url, destination: Path(destination).write_bytes(b"tampered"),
    )
    with pytest.raises(RuntimeError, match="integrity check failed"):
        models.ensure_stt_binary(tmp_path)
    assert not (tmp_path / "stt_onlyprobs.x86_64.bin").exists()
