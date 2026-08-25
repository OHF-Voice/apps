"""Which sentence source a transcript came from.

The constrained decoder can only emit a phrase the grammar contains, so every
transcript traces back to exactly one template -- but the FST returns tokens,
not the template it walked. Debug mode wants that provenance ("this was your
sentence trigger, not a built-in command"), so we reconstruct it: ask
``training.assemble_sources`` which template each source contributed, turn each
template back into a pattern over the same list values and number spellings the
trainer compiled, and see which one the transcript satisfies.

Parity is by construction: the templates and list values are the ones the
grammar was built from, and the numbers come from the library's own
``_spellout_variants`` -- the function the FST builder uses -- so "set the
brightness to fifty percent" is attributed rather than missed because we
guessed a different spelling of 50.

Attribution is only ever needed while someone is watching the debug view, one
utterance at a time, so this is built lazily and matched by a plain loop.
"""
import logging
import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

_LOGGER = logging.getLogger("speech-to-phrase.sources")

# Source label prefixes produced by training.assemble_sources.
BUILTIN = "builtin"
CUSTOM = "custom"

_REF_SPLIT_RE = re.compile(r"(\{[^{}]*\})")
_RANGE_RE = re.compile(r"^(-?\d+)\s*\.\.\s*(-?\d+)(?:\s*[,/]\s*(-?\d+))?$")


def normalize(text: str) -> str:
    """The library's template normalization (NFC + lowercase + collapsed
    whitespace), applied to both sides of a comparison."""
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


@lru_cache(maxsize=1)
def _spellout_fn():
    """The FST builder's own number spellout, so a language whose low numerals
    inflect (cs "dva"/"dvě") is matched in whichever form was actually said.

    ``_spellout_variants`` is the current entry point; ``_spellout_words`` is the
    single-reading one it replaced. Both are private, but borrowing them is the
    whole point -- a spellout of our own would disagree with the grammar's, and
    then "fifty" would go unattributed for no visible reason."""
    from speech_to_phrase import templates as t

    variants = getattr(t, "_spellout_variants", None)
    if variants is not None:
        return lambda num, lang: [" ".join(w) for w in variants(num, lang)]
    words = getattr(t, "_spellout_words", None)
    if words is not None:
        return lambda num, lang: [" ".join(words(num, lang))]
    return None


def _range_words(spec: str, lang: str) -> Optional[List[str]]:
    """Spoken forms of every number an inline ``{lo..hi[,step]}`` covers."""
    m = _RANGE_RE.match(spec)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    step = abs(int(m.group(3) or 1)) or 1
    if lo > hi:
        step = -step
    try:
        spellout = _spellout_fn()
    except Exception:  # noqa: BLE001
        spellout = None
    if spellout is None:
        _LOGGER.debug("no spellout available; numeric commands go unattributed")
        return None
    out: List[str] = []
    for num in range(lo, hi + step, step):
        out.extend(normalize(w) for w in spellout(num, lang))
    return list(dict.fromkeys(out))


def _ref_values(
    ref: str, list_values: Dict[str, Sequence[str]], lang: str
) -> Optional[List[str]]:
    ref = ref.strip()
    words = _range_words(ref, lang)
    if words is not None:
        return words
    values = list_values.get(ref.split(":", 1)[0].strip())
    if values is None:
        return None
    return [normalize(v) for v in values if v and v.strip()]


def _literal(text: str) -> str:
    """Escaped literal with runs of whitespace made flexible."""
    return "".join(
        r"\s+" if part.isspace() else re.escape(part)
        for part in re.split(r"(\s+)", text)
        if part
    )


def _pattern(
    template: str, list_values: Dict[str, Sequence[str]], lang: str
) -> Optional[str]:
    """Regex body for one template, or None if a reference can't be resolved (in
    which case the template is simply not attributable -- better than a pattern
    that matches the wrong thing)."""
    parts: List[str] = []
    for chunk in _REF_SPLIT_RE.split(template):
        if not chunk:
            continue
        if chunk.startswith("{") and chunk.endswith("}"):
            values = _ref_values(chunk[1:-1], list_values, lang)
            if not values:
                return None
            # Longest first: alternation is first-match, so "living room" must be
            # tried before "living" where both are values.
            alts = sorted(set(values), key=len, reverse=True)
            parts.append("(?:" + "|".join(_literal(v) for v in alts) + ")")
        else:
            parts.append(_literal(chunk))
    return "".join(parts) or None


class Attributor:
    """Compiled patterns for every template, tagged with its source."""

    def __init__(self, patterns: Sequence[Tuple[str, str, "re.Pattern"]]):
        self._patterns = list(patterns)

    def __len__(self) -> int:
        return len(self._patterns)

    def attribute(self, text: str) -> Optional[Dict[str, str]]:
        """``{"source": <label>, "template": <template>}`` for `text`, or None if
        nothing in the grammar produces it (which for a constrained decode means
        the grammar changed since -- or a reference we could not resolve)."""
        norm = normalize(text)
        if not norm:
            return None
        for source, template, rx in self._patterns:
            if rx.fullmatch(norm):
                return {"source": source, "template": display_template(template)}
        return None


def build(
    templates_by_source: Dict[str, List[str]],
    list_values: Dict[str, Sequence[str]],
    lang: str,
) -> Attributor:
    """An :class:`Attributor` over the assembled grammar.

    Literal templates are tried before templates with slots, so a sentence
    trigger reading "movie time" wins over a built-in whose ``{name}`` happens to
    include a device called "movie" -- the specific reading of an ambiguous
    transcript is the useful one.
    """
    plain: List[Tuple[str, str, "re.Pattern"]] = []
    slotted: List[Tuple[str, str, "re.Pattern"]] = []
    n_skipped = 0
    for source, templates in templates_by_source.items():
        for template in templates:
            body = _pattern(template, list_values, lang)
            if body is None:
                n_skipped += 1
                continue
            try:
                rx = re.compile(body)
            except re.error:
                n_skipped += 1
                continue
            entry = (source, template, rx)
            (slotted if "{" in template else plain).append(entry)
    if n_skipped:
        _LOGGER.debug("%d template(s) not attributable", n_skipped)
    return Attributor(plain + slotted)


_SCOPED_SLOT_RE = re.compile(r"\{(name|area|floor)__[^{}]*\}")


def display_template(template: str) -> str:
    """The matched template as a person would recognize it: the domain-scoped
    list names the grammar needs internally (``{name__light__brightness}``) mean
    nothing to a reader, so show the slot they stand for."""
    return _SCOPED_SLOT_RE.sub(lambda m: "{" + m.group(1) + "}", template)
