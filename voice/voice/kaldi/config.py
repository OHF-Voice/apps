"""YAML configuration: wake words, their accepted forms, and confusables.

Only wake words declared here are recognized, and only words the model can
pronounce may be used -- either because they are in the model's lexicon or
because the config supplies the pronunciation. Nothing is guessed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .lexicon import LexiconError, ModelLexicon

_LOGGER = logging.getLogger(__name__)

# Inline pronunciation inside a phrase: okay /n ˈæ b u/
_TOKEN_RE = re.compile(r"/([^/]+)/|(\S+)")

# Search width. These grammars are tiny and heavily constrained, so Kaldi's
# usual wide beam is mostly wasted work. See README "Performance".
DEFAULT_BEAM = 10.0
DEFAULT_MAX_ACTIVE = 700
DEFAULT_LATTICE_BEAM = 1.0

# How the four graphs are shared out between resident decoder processes.
#
# Every graph in one process shares that process's acoustic pass, which is
# identical for all of them; graphs in different processes run concurrently.
# So this trades duplicated neural-net work against parallelism, and the best
# split is not either extreme. Measured on a Pi 4 (4 cores, 1.4 s window):
#
#   1 process  x 4 graphs   473 ms    88 MB   nnet once, searches serialised
#   4 processes x 1 graph    382 ms   350 MB   nnet x4, searches parallel
#   2 processes x 2 graphs   359 ms   176 MB   <- this default
#
# Pairing matters a little: kw+bg | prefix+kwc and kw+kwc | bg+prefix both
# land at ~360 ms, kw+prefix | bg+kwc at 380 ms (it puts the two cheapest
# searches together and the two dearest together).
DEFAULT_GROUPS: tuple[tuple[str, ...], ...] = (("kw", "bg"), ("prefix", "kwc"))

# Graphs that must never have their active-state count capped.
#
# `max_active` keeps only the N best-scoring states per frame. In the forced
# graphs (kw, prefix) there is essentially one path, and in bg the best path
# through a free phone loop is found greedily, so capping changes nothing. The
# kwc graph is different: its whole job is to find a RIVAL hypothesis -- a
# confusable that beats the wake phrase -- and that rival is, by construction,
# scoring worse than the forced wake phrase until the moment it wins. Capping
# prunes it before it gets there, the margin collapses, and a false positive
# is silently accepted. Measured: with kwc capped at 700, three recordings of
# "okay natalie" were accepted as "okay nabu"; uncapped, all three are
# rejected as confusable.
UNCAPPED_GRAPHS = frozenset({"kwc"})

DEFAULT_THETA_PREFIX = 0.85
DEFAULT_THETA_LAST = 0.25
DEFAULT_THETA_COMPETITOR = 0.06


class ConfigError(Exception):
    """The configuration file is invalid."""


@dataclass(frozen=True)
class Thresholds:
    """Verifier decision thresholds (see verify.py for the statistics)."""

    theta_prefix: float = DEFAULT_THETA_PREFIX
    theta_last: float = DEFAULT_THETA_LAST
    theta_competitor: float = DEFAULT_THETA_COMPETITOR

    @staticmethod
    def from_dict(data: dict[str, Any], base: "Thresholds") -> "Thresholds":
        unknown = set(data) - {"theta_prefix", "theta_last", "theta_competitor"}
        if unknown:
            raise ConfigError(f"Unknown threshold keys: {sorted(unknown)}")
        return Thresholds(
            theta_prefix=float(data.get("theta_prefix", base.theta_prefix)),
            theta_last=float(data.get("theta_last", base.theta_last)),
            theta_competitor=float(
                data.get("theta_competitor", base.theta_competitor)
            ),
        )


@dataclass(frozen=True)
class DecoderConfig:
    """Kaldi search width, shared by the resident and subprocess decoders."""

    beam: float = DEFAULT_BEAM
    max_active: int = DEFAULT_MAX_ACTIVE
    lattice_beam: float = DEFAULT_LATTICE_BEAM
    groups: tuple[tuple[str, ...], ...] = DEFAULT_GROUPS

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DecoderConfig":
        unknown = set(data) - {"beam", "max_active", "lattice_beam", "groups"}
        if unknown:
            raise ConfigError(f"Unknown decoder keys: {sorted(unknown)}")
        decoder = DecoderConfig(
            beam=float(data.get("beam", DEFAULT_BEAM)),
            max_active=int(data.get("max_active", DEFAULT_MAX_ACTIVE)),
            lattice_beam=float(data.get("lattice_beam", DEFAULT_LATTICE_BEAM)),
            groups=_load_groups(data.get("groups")),
        )
        if decoder.beam <= 0 or decoder.lattice_beam <= 0 or decoder.max_active < 1:
            raise ConfigError(
                "decoder: beam and lattice_beam must be > 0, max_active >= 1"
            )
        return decoder

    def args(self, tag: str) -> list[str]:
        """Decoder flags for one graph.

        `max_active` is deliberately not applied to the confusable graph; see
        UNCAPPED_GRAPHS.
        """
        args = [f"--beam={self.beam}", f"--lattice-beam={self.lattice_beam}"]
        if tag not in UNCAPPED_GRAPHS:
            args.append(f"--max-active={self.max_active}")
        return args


def _load_groups(raw: Any) -> tuple[tuple[str, ...], ...]:
    """Validate that `groups` partitions the graphs exactly once each."""
    from .train import GRAPH_TAGS  # imported here to avoid a cycle

    if raw is None:
        return DEFAULT_GROUPS
    if not isinstance(raw, list) or not raw:
        raise ConfigError("decoder: `groups` must be a non-empty list of lists")

    groups: list[tuple[str, ...]] = []
    for group in raw:
        if isinstance(group, str):
            group = [group]
        if not isinstance(group, list) or not group:
            raise ConfigError("decoder: each entry of `groups` must be a non-empty list")
        groups.append(tuple(str(tag) for tag in group))

    flat = [tag for group in groups for tag in group]
    if sorted(flat) != sorted(GRAPH_TAGS):
        raise ConfigError(
            f"decoder: `groups` must name each of {sorted(GRAPH_TAGS)} exactly "
            f"once; got {flat}"
        )
    return tuple(groups)


@dataclass(frozen=True)
class Phrase:
    """One accepted (or confusable) form, as a sequence of lexicon words."""

    text: str
    """The phrase exactly as written in the config."""

    words: tuple[str, ...]
    """Lexicon word ids, in order."""


@dataclass
class WakeWordConfig:
    """One configured wake word."""

    id: str
    """Canonical id (the key in `wake_words`)."""

    model: str
    """Key in the `models` mapping."""

    names: tuple[str, ...]
    """Ids accepted in the Wyoming `detect` message."""

    phrases: tuple[Phrase, ...]
    """Accepted spoken forms."""

    confusables: tuple[Phrase, ...]
    """Phrases that must NOT be accepted as this wake word."""

    lexicon: dict[str, tuple[str, ...]]
    """word -> pronunciations, for every word used above."""

    thresholds: Thresholds

    description: str | None = None

    @property
    def phrase_text(self) -> str:
        return self.phrases[0].text if self.phrases else self.id


@dataclass
class ModelConfig:
    """One Kaldi acoustic model."""

    id: str
    directory: Path
    language: str = "en_US"


@dataclass
class ServiceConfig:
    """The whole configuration."""

    models: dict[str, ModelConfig]
    wake_words: dict[str, WakeWordConfig]
    by_name: dict[str, WakeWordConfig] = field(default_factory=dict)
    tools_dir: Path = Path("local/tools")
    train_dir: Path = Path("train")
    max_window_seconds: float = 1.4
    resident_decoders: bool = True
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def resolve(self, name: str) -> WakeWordConfig | None:
        """Look up a name from a `detect` message."""
        return self.by_name.get(name) or self.by_name.get(_normalize_name(name))


def _normalize_name(name: str) -> str:
    return re.sub(r"[\s_-]+", "_", name.strip().lower())


def tokenize_phrase(text: str) -> list[tuple[str, str | None]]:
    """Split a phrase into (word, inline_pronunciation | None) tokens."""
    tokens: list[tuple[str, str | None]] = []
    for match in _TOKEN_RE.finditer(text):
        inline, word = match.group(1), match.group(2)
        if inline is not None:
            tokens.append((f"_ipa_{len(tokens)}", " ".join(inline.split())))
        else:
            tokens.append((word.lower(), None))
    return tokens


def load_config(path: str | Path) -> ServiceConfig:
    """Load and fully validate a config file."""
    path = Path(path).absolute()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        raise ConfigError(f"Cannot read config {path}: {err}") from err

    if not isinstance(raw, dict):
        raise ConfigError(f"Config {path} must be a YAML mapping")

    base_dir = path.parent
    unknown = set(raw) - {
        "models",
        "defaults",
        "wake_words",
        "tools_dir",
        "train_dir",
        "max_window_seconds",
        "resident_decoders",
        "decoder",
    }
    if unknown:
        raise ConfigError(f"Unknown top-level keys: {sorted(unknown)}")

    models = _load_models(raw.get("models"), base_dir)

    raw_decoder = raw.get("decoder") or {}
    if not isinstance(raw_decoder, dict):
        raise ConfigError("`decoder` must be a mapping")
    decoder = DecoderConfig.from_dict(raw_decoder)
    defaults = Thresholds.from_dict(raw.get("defaults") or {}, Thresholds())

    raw_wake_words = raw.get("wake_words")
    if not isinstance(raw_wake_words, dict) or not raw_wake_words:
        raise ConfigError("`wake_words` must be a non-empty mapping")

    lexicons: dict[str, ModelLexicon] = {}
    wake_words: dict[str, WakeWordConfig] = {}
    by_name: dict[str, WakeWordConfig] = {}
    try:
        for wake_id, raw_wake in raw_wake_words.items():
            wake_word = _load_wake_word(
                str(wake_id), raw_wake or {}, models, defaults, lexicons
            )
            wake_words[wake_word.id] = wake_word
            for name in wake_word.names:
                existing = by_name.get(name)
                if existing is not None and existing.id != wake_word.id:
                    raise ConfigError(
                        f"Name {name!r} is claimed by both {existing.id!r} "
                        f"and {wake_word.id!r}"
                    )
                by_name[name] = wake_word
    finally:
        for lexicon in lexicons.values():
            lexicon.close()

    return ServiceConfig(
        models=models,
        wake_words=wake_words,
        by_name=by_name,
        tools_dir=_resolve_path(raw.get("tools_dir", "local/tools"), base_dir),
        train_dir=_resolve_path(raw.get("train_dir", "train"), base_dir),
        max_window_seconds=float(raw.get("max_window_seconds", 1.4)),
        resident_decoders=bool(raw.get("resident_decoders", True)),
        decoder=decoder,
    )


def _resolve_path(value: Any, base_dir: Path) -> Path:
    candidate = Path(str(value)).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _load_models(raw_models: Any, base_dir: Path) -> dict[str, ModelConfig]:
    if not isinstance(raw_models, dict) or not raw_models:
        raise ConfigError("`models` must be a non-empty mapping")

    models: dict[str, ModelConfig] = {}
    for model_id, raw_model in raw_models.items():
        model_id = str(model_id)
        raw_model = raw_model or {}
        if not isinstance(raw_model, dict):
            raise ConfigError(f"Model {model_id!r} must be a mapping")
        unknown = set(raw_model) - {"directory", "language"}
        if unknown:
            raise ConfigError(f"Model {model_id!r}: unknown keys {sorted(unknown)}")

        directory = _resolve_path(
            raw_model.get("directory", f"local/models/{model_id}"), base_dir
        )
        if not directory.is_dir():
            raise ConfigError(
                f"Model {model_id!r}: directory not found: {directory}"
            )
        models[model_id] = ModelConfig(
            id=model_id,
            directory=directory,
            language=str(raw_model.get("language", "en_US")),
        )
    return models


def _load_wake_word(
    wake_id: str,
    raw: Any,
    models: dict[str, ModelConfig],
    defaults: Thresholds,
    lexicons: dict[str, ModelLexicon],
) -> WakeWordConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"Wake word {wake_id!r} must be a mapping")
    unknown = set(raw) - {
        "model",
        "names",
        "phrases",
        "confusables",
        "pronunciations",
        "thresholds",
        "description",
    }
    if unknown:
        raise ConfigError(f"Wake word {wake_id!r}: unknown keys {sorted(unknown)}")

    model_id = str(raw.get("model") or "")
    if model_id not in models:
        raise ConfigError(
            f"Wake word {wake_id!r}: unknown model {model_id!r} "
            f"(known: {sorted(models)})"
        )

    lexicon = lexicons.get(model_id)
    if lexicon is None:
        try:
            lexicon = ModelLexicon(models[model_id].directory)
        except LexiconError as err:
            raise ConfigError(f"Wake word {wake_id!r}: {err}") from err
        lexicons[model_id] = lexicon

    # Config-supplied pronunciations, validated against the model's phones.
    overrides: dict[str, tuple[str, ...]] = {}
    raw_prons = raw.get("pronunciations") or {}
    if not isinstance(raw_prons, dict):
        raise ConfigError(f"Wake word {wake_id!r}: `pronunciations` must be a mapping")
    for word, prons in raw_prons.items():
        word = str(word).lower()
        if isinstance(prons, str):
            prons = [prons]
        if not isinstance(prons, list) or not prons:
            raise ConfigError(
                f"Wake word {wake_id!r}: pronunciations for {word!r} must be "
                f"a string or a non-empty list of strings"
            )
        validated = []
        for pron in prons:
            try:
                validated.append(" ".join(lexicon.validate_pronunciation(word, str(pron))))
            except LexiconError as err:
                raise ConfigError(f"Wake word {wake_id!r}: {err}") from err
        overrides[word] = tuple(validated)

    used: dict[str, tuple[str, ...]] = {}

    def resolve_phrase(text: str, kind: str) -> Phrase:
        tokens = tokenize_phrase(text)
        if not tokens:
            raise ConfigError(f"Wake word {wake_id!r}: empty {kind} phrase")

        words: list[str] = []
        for word, inline in tokens:
            if inline is not None:
                # Inline /ipa/: give it a stable id derived from the phonemes
                # so identical inline prons share one lexicon entry.
                try:
                    phonemes = lexicon.validate_pronunciation(text, inline)
                except LexiconError as err:
                    raise ConfigError(f"Wake word {wake_id!r}: {err}") from err
                word = "_ipa_" + "".join(phonemes)
                used[word] = (" ".join(phonemes),)
            elif word in overrides:
                used[word] = overrides[word]
            else:
                prons = lexicon.pronunciations(word)
                if not prons:
                    raise ConfigError(
                        f"Wake word {wake_id!r}: {kind} phrase {text!r} uses "
                        f"{word!r}, which is not in the {model_id} lexicon. "
                        f"Add a `pronunciations:` entry for it (the verifier "
                        f"never guesses pronunciations)."
                    )
                used[word] = tuple(prons)
            words.append(word)

        return Phrase(text=text, words=tuple(words))

    raw_phrases = raw.get("phrases")
    if isinstance(raw_phrases, str):
        raw_phrases = [raw_phrases]
    if not isinstance(raw_phrases, list) or not raw_phrases:
        raise ConfigError(
            f"Wake word {wake_id!r}: `phrases` must be a non-empty list of "
            f"accepted spoken forms"
        )
    phrases = tuple(resolve_phrase(str(p), "wake") for p in raw_phrases)

    raw_confusables = raw.get("confusables") or []
    if isinstance(raw_confusables, str):
        raw_confusables = [raw_confusables]
    if not isinstance(raw_confusables, list):
        raise ConfigError(f"Wake word {wake_id!r}: `confusables` must be a list")
    confusables = tuple(resolve_phrase(str(p), "confusable") for p in raw_confusables)

    accepted_sequences = {p.words for p in phrases}
    overlap = [c.text for c in confusables if c.words in accepted_sequences]
    if overlap:
        raise ConfigError(
            f"Wake word {wake_id!r}: {overlap} appear as both an accepted "
            f"phrase and a confusable"
        )

    unused = set(overrides) - set(used)
    if unused:
        _LOGGER.warning(
            "Wake word %s: pronunciations for %s are not used by any phrase",
            wake_id,
            ", ".join(sorted(unused)),
        )

    raw_names = raw.get("names") or [wake_id]
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    if not isinstance(raw_names, list):
        raise ConfigError(f"Wake word {wake_id!r}: `names` must be a list")
    names: list[str] = []
    for name in [wake_id, *raw_names]:
        for variant in (str(name), _normalize_name(str(name))):
            if variant not in names:
                names.append(variant)

    raw_thresholds = raw.get("thresholds") or {}
    if not isinstance(raw_thresholds, dict):
        raise ConfigError(f"Wake word {wake_id!r}: `thresholds` must be a mapping")
    try:
        thresholds = Thresholds.from_dict(raw_thresholds, defaults)
    except ConfigError as err:
        raise ConfigError(f"Wake word {wake_id!r}: {err}") from err

    return WakeWordConfig(
        id=wake_id,
        model=model_id,
        names=tuple(names),
        phrases=phrases,
        confusables=confusables,
        lexicon=used,
        thresholds=thresholds,
        description=(
            str(raw["description"]) if raw.get("description") is not None else None
        ),
    )
