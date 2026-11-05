"""Relocatable view of a Kaldi acoustic model.

The shipped online-decoding configs carry absolute paths from the machine
that trained them, so the verifier regenerates them under its own work
directory with paths re-rooted at wherever the model actually lives. The
model tree itself is never modified.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

# --option=/some/abs/path
_OPTION_RE = re.compile(r"^(--[^=]+)=(/.*)$")
_ONLINE_MARKER = re.compile(r"(?:^|/)model/online/(.*)$")


@dataclass
class PreparedModel:
    """A model whose online-decoding config is usable from this machine."""

    id: str
    data_dir: Path
    """Model package directory (contains lexicon.db and model/)."""

    model_dir: Path
    """`<data_dir>/model`: conf/, phones/, model/, online/, extractor/."""

    final_mdl: Path
    online_conf: Path

    @property
    def phones_dir(self) -> Path:
        return self.model_dir / "phones"

    @property
    def am_dir(self) -> Path:
        """Directory holding final.mdl + tree, as mkgraph.sh expects."""
        return self.model_dir / "model"


def prepare_model(model_id: str, data_dir: Path, work_dir: Path) -> PreparedModel:
    """Validate a model tree and regenerate its online config in `work_dir`."""
    data_dir = Path(data_dir).absolute()
    model_dir = data_dir / "model"
    final_mdl = model_dir / "model" / "final.mdl"
    source_conf_dir = model_dir / "online" / "conf"

    for path in (model_dir, model_dir / "phones", source_conf_dir, final_mdl):
        if not path.exists():
            raise FileNotFoundError(f"Model {model_id!r} is incomplete: {path}")

    out_conf_dir = Path(work_dir).absolute() / model_id / "online_conf"
    out_conf_dir.mkdir(parents=True, exist_ok=True)

    online_dir = model_dir / "online"
    for conf_file in sorted(source_conf_dir.glob("*.conf")):
        target = out_conf_dir / conf_file.name
        text = conf_file.read_text(encoding="utf-8")
        rewritten = "\n".join(
            _rewrite_line(line, online_dir, out_conf_dir)
            for line in text.splitlines()
        )
        _write_if_changed(target, rewritten + "\n")

    # Non-.conf siblings (e.g. splice_opts) are copied verbatim.
    for other in sorted(source_conf_dir.iterdir()):
        if other.is_file() and other.suffix != ".conf":
            shutil.copy(other, out_conf_dir / other.name)

    online_conf = out_conf_dir / "online.conf"
    if not online_conf.is_file():
        raise FileNotFoundError(f"Model {model_id!r}: missing online.conf")

    _LOGGER.debug("Prepared model %s (online conf: %s)", model_id, online_conf)
    return PreparedModel(
        id=model_id,
        data_dir=data_dir,
        model_dir=model_dir,
        final_mdl=final_mdl,
        online_conf=online_conf,
    )


def _rewrite_line(line: str, online_dir: Path, out_conf_dir: Path) -> str:
    """Re-root an absolute `--option=/path` at this machine's model tree."""
    match = _OPTION_RE.match(line.strip())
    if match is None:
        return line

    option, value = match.group(1), match.group(2)
    # Strip any trailing inline comment Kaldi would not accept in a path.
    marker = _ONLINE_MARKER.search(value)
    if marker is None:
        return line

    relative = marker.group(1)
    if relative.startswith("conf/"):
        # conf files are the ones we regenerate; point at our copies
        new_value = out_conf_dir / Path(relative).name
    else:
        new_value = online_dir / relative

    return f"{option}={new_value}"


def _write_if_changed(path: Path, text: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.write_text(text, encoding="utf-8")
