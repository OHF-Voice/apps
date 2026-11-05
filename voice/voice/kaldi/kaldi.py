"""Kaldi tool discovery and grammar-constrained decoding.

Adapted from the voice project's `alignment.py`. One decode of a wav against
one HCLG graph yields the best-path cost (for likelihood ratios) and,
optionally, a word-level CTM (for the wake/command split point).
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

# rhasspy chain models: 3x frame subsampling -> 30 ms output frames.
# Passing the factor to the decoder makes it evaluate the nnet once per
# OUTPUT frame (~3x cheaper than the per-input-frame default) and keeps
# lattice frame indices in output-frame units, so CTM times use FRAME_SHIFT.
FRAME_SUBSAMPLING_FACTOR = 3
FRAME_SHIFT = 0.01 * FRAME_SUBSAMPLING_FACTOR

_BEST_COST_RE = re.compile(
    r"For utterance (\S+), best cost ([\d.eE+-]+) \+ ([\d.eE+-]+) = "
    r"([\d.eE+-]+) over (\d+) frames"
)

# (word, start_sec, end_sec)
CtmWord = tuple[str, float, float]
# (best_path_total_cost, n_output_frames)
BestCost = tuple[float, int]


@dataclass
class SpeechTools:
    """Local Kaldi/OpenFST tool tree."""

    tools_dir: Path
    kaldi_dir: Path
    openfst_dir: Path
    _extended_env: dict[str, str] | None = field(default=None, repr=False)

    @staticmethod
    def from_dir(tools_dir: str | Path) -> "SpeechTools":
        tools_dir = Path(tools_dir).absolute()
        return SpeechTools(
            tools_dir=tools_dir,
            kaldi_dir=tools_dir / "kaldi",
            openfst_dir=tools_dir / "openfst",
        )

    @property
    def egs_utils_dir(self) -> Path:
        return self.kaldi_dir / "utils"

    @property
    def egs_steps_dir(self) -> Path:
        return self.kaldi_dir / "steps"

    @property
    def extended_env(self) -> dict[str, str]:
        if self._extended_env is None:
            env = os.environ.copy()
            bin_dirs = [str(self.kaldi_dir / "bin"), str(self.egs_utils_dir)]
            lib_dirs = [str(self.kaldi_dir / "lib")]

            if self.openfst_dir:
                bin_dirs.append(str(self.openfst_dir / "bin"))
                lib_dirs.append(str(self.openfst_dir / "lib"))

            current_path = env.get("PATH")
            if current_path:
                bin_dirs.append(current_path)

            current_lib_path = env.get("LD_LIBRARY_PATH")
            if current_lib_path:
                lib_dirs.append(current_lib_path)

            env["PATH"] = os.pathsep.join(bin_dirs)
            env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs)
            # decodes run concurrently; keep BLAS single-threaded to avoid thrash
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["OMP_NUM_THREADS"] = "1"
            self._extended_env = env

        return self._extended_env

    def check(self) -> None:
        """Fail early (and clearly) when the tool tree is incomplete."""
        for path in (
            self.kaldi_dir / "bin" / "online2-wav-nnet3-latgen-faster",
            self.kaldi_dir / "bin" / "nbest-to-ctm",
            self.egs_utils_dir / "prepare_lang.sh",
            self.egs_utils_dir / "mkgraph.sh",
            self.egs_steps_dir / "online" / "nnet3" / "prepare_online_decoding.sh",
            self.openfst_dir / "bin" / "fstcompile",
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing Kaldi tool: {path}")

    def run(self, program: str, args: list[str], **kwargs) -> bytes:
        kwargs.setdefault("env", self.extended_env)
        kwargs.setdefault("stderr", subprocess.PIPE)

        with subprocess.Popen([program] + args, stdout=subprocess.PIPE, **kwargs) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                detail = (stderr or stdout or b"").decode(errors="replace")
                raise RuntimeError(
                    f"Command failed: {program} {shlex.join(args)}: {detail[-2000:]}"
                )
            return stdout

    def run_pipeline(self, *commands: list[str], **kwargs) -> bytes:
        kwargs.setdefault("env", self.extended_env)
        kwargs.setdefault("stderr", subprocess.PIPE)
        command_str = " | ".join(shlex.join(c) for c in commands)

        with subprocess.Popen(
            command_str, stdout=subprocess.PIPE, shell=True, **kwargs
        ) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                detail = (stderr or stdout or b"").decode(errors="replace")
                raise RuntimeError(
                    f"Command failed: {command_str}: {detail[-2000:]}"
                )
            return stdout


def read_words_txt(path: Path) -> dict[str, str]:
    """words.txt as {word_id: word}."""
    id_to_word: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        word, word_id = line.split()
        id_to_word[word_id] = word
    return id_to_word


def decode_best_path(
    tools: SpeechTools,
    model_file: Path,
    online_conf: Path,
    graph_dir: Path,
    lang_dir: Path | None,
    wav_path: Path,
    want_ctm: bool,
    search_args: list[str],
) -> tuple[BestCost | None, list[CtmWord]]:
    """One grammar-constrained decode of a wav.

    Returns (best-path cost, ctm) where ctm is non-empty only when `want_ctm`
    (which also requires `lang_dir`, for word boundaries).

    The word sequence/times come from the 1-BEST path (lattice-1best +
    nbest-to-ctm), NOT lattice-to-ctm-conf: MBR over a grammar lattice can
    prefer paths other than the best one.
    """
    graph_dir = Path(graph_dir).absolute()
    model_file = Path(model_file).absolute()
    online_conf = Path(online_conf).absolute()

    id_to_word: dict[str, str] = {}
    if want_ctm:
        id_to_word = read_words_txt(graph_dir / "words.txt")

    env = tools.extended_env

    with tempfile.TemporaryDirectory() as temp_dir_str:
        # own temp dir per decode: shared intermediate paths race
        temp_dir = Path(temp_dir_str)
        (temp_dir / "spk2utt").write_text("utt1 utt1", encoding="utf-8")
        (temp_dir / "wav.scp").write_text(
            f"utt1 {Path(wav_path).absolute()}", encoding="utf-8"
        )

        latgen = shlex.join(
            [
                "online2-wav-nnet3-latgen-faster",
                f"--config={online_conf}",
                "--online=true",
                "--acoustic-scale=1.0",
                # Search width (beam / lattice-beam / max-active) comes from
                # the config so the resident and subprocess paths agree. A
                # tight lattice-beam avoids determinizing large filler-path
                # equivalence classes; we only ever use the 1-best path.
                *search_args,
                "--do-endpointing=false",
                f"--frame-subsampling-factor={FRAME_SUBSAMPLING_FACTOR}",
                str(model_file),
                str(graph_dir / "HCLG.fst"),
                f"ark:{temp_dir / 'spk2utt'}",
                f"scp:{temp_dir / 'wav.scp'}",
                "ark:-",
            ]
        )
        onebest = "lattice-1best ark:- ark:-"
        if want_ctm:
            if lang_dir is None:
                raise ValueError("want_ctm requires lang_dir")
            align = shlex.join(
                [
                    "lattice-align-words",
                    str(Path(lang_dir).absolute() / "phones" / "word_boundary.int"),
                    str(model_file),
                    "ark:-",
                    "ark:-",
                ]
            )
            lats_path = temp_dir / "aligned.lats"
            cmd = (
                f"{latgen} | {onebest} | {align} | "
                f"tee {shlex.quote(str(lats_path))} | "
                f"lattice-best-path ark:- ark:/dev/null"
            )
        else:
            cmd = f"{latgen} | {onebest} | lattice-best-path ark:- ark:/dev/null"

        proc = subprocess.run(
            cmd,
            shell=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        stderr = proc.stderr.decode(errors="replace")
        if proc.returncode != 0:
            last_line = stderr.splitlines()[-1][:300] if stderr.strip() else "no output"
            raise RuntimeError(f"decode failed ({graph_dir.name}): {last_line}")

        cost: BestCost | None = None
        match = _BEST_COST_RE.search(stderr)
        if match:
            cost = (float(match.group(4)), int(match.group(5)))

        ctm_words: list[CtmWord] = []
        if want_ctm:
            ctm_out = subprocess.run(
                f"nbest-to-ctm --frame-shift={FRAME_SHIFT} "
                f"ark:{shlex.quote(str(lats_path))} -",
                shell=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            ).stdout.decode()
            for line in ctm_out.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                _utt, _ch, start_str, dur_str, word_id = parts[:5]
                start = float(start_str)
                ctm_words.append(
                    (id_to_word.get(word_id, word_id), start, start + float(dur_str))
                )

        return cost, ctm_words
