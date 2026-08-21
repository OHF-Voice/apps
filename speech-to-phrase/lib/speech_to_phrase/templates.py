"""Sentence-template parser and grammar-FST builder.

Templates use the familiar syntax:
  - ``[optional]``
  - ``(alt1|alt2)``
  - ``{1..100}`` / ``{1..24/2}`` / ``{1, 5, 10}`` numeric ranges (spelled out)
  - ``{list_name}`` references resolved from ``list_values``

The compiler turns templates into an :class:`Fst` whose arc labels are **integer
token ids** produced by a :class:`TokenizerLike` (SentencePiece subwords for
Citrinet, character ids for Coqui). Epsilon is the sentinel :data:`EPS` (``-1``);
the native FST layer maps token id ``t`` to OpenFST label ``t + 1`` and epsilon to
``0``.

This module is pure Python and has no dependency on the acoustic backend.
"""

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

from icu_rbnf import spellout


class TokenizerLike(Protocol):
    def text_to_ids(self, text: str) -> List[int]: ...  # noqa: E704


# Epsilon label. Token ids are >= 0, so a negative sentinel never collides.
EPS = -1

STATUS_EMPTY = 0
STATUS_WORD = 1


# -----------------------------
# Finite State Transducer (FST)
# -----------------------------


@dataclass
class FstArc:
    to_state: int
    in_label: int = EPS
    out_label: int = EPS


@dataclass
class Fst:
    arcs: Dict[int, List[FstArc]] = field(default_factory=lambda: defaultdict(list))
    states: Set[int] = field(default_factory=lambda: {0})
    final_states: Set[int] = field(default_factory=set)
    start: int = 0
    current_state: int = 0

    def next_state(self) -> int:
        self.states.add(self.current_state)
        self.current_state += 1
        return self.current_state

    def next_edge(
        self,
        from_state: int,
        in_label: Optional[int] = None,
        out_label: Optional[int] = None,
    ) -> int:
        to_state = self.next_state()
        self.add_edge(from_state, to_state, in_label, out_label)
        return to_state

    def add_edge(
        self,
        from_state: int,
        to_state: int,
        in_label: Optional[int] = None,
        out_label: Optional[int] = None,
    ) -> None:
        if in_label is None:
            in_label = EPS

        if out_label is None:
            out_label = in_label

        self.states.add(from_state)
        self.states.add(to_state)
        self.arcs[from_state].append(FstArc(to_state, in_label, out_label))

    def accept(self, state: int) -> None:
        self.states.add(state)
        self.final_states.add(state)

    def arc_tuples(self) -> List[Tuple[int, int, int, int]]:
        """Flat ``(from_state, to_state, in_label, out_label)`` list for the
        native grammar builder."""
        out: List[Tuple[int, int, int, int]] = []
        for from_state, arcs in self.arcs.items():
            for arc in arcs:
                out.append((from_state, arc.to_state, arc.in_label, arc.out_label))
        return out


# ----------------------------
# AST nodes
# ----------------------------


@dataclass
class Node:
    pass


@dataclass
class SequenceNode(Node):
    parts: List[Node]
    separators: List[bool]  # True if there was whitespace before the corresponding part


@dataclass
class LiteralNode(Node):
    text: str  # single literal chunk, no whitespace


@dataclass
class OptionalNode(Node):
    child: Node


@dataclass
class AlternativesNode(Node):
    options: List[Node]


@dataclass
class ListRefNode(Node):
    name: str


@dataclass
class NumberRangeNode(Node):
    items: List[Tuple[int, ...]]  # (number,) or (start, end, step)


# ----------------------------
# Parser
# ----------------------------


class TemplateParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def parse(self) -> Node:
        node = self._parse_sequence(stop_chars=set())
        self._consume_ws()
        if self.pos != len(self.text):
            raise ValueError(f"Unexpected trailing text at position {self.pos}")
        return node

    def _peek(self) -> Optional[str]:
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def _take(self) -> str:
        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of input")
        ch = self.text[self.pos]
        self.pos += 1
        return ch

    def _expect(self, expected: str) -> None:
        actual = self._take()
        if actual != expected:
            raise ValueError(
                f"Expected '{expected}' at position {self.pos - 1}, got '{actual}'"
            )

    def _consume_ws(self) -> bool:
        had_ws = False
        while True:
            peek = self._peek()
            if (peek is None) or (not peek.isspace()):
                break
            had_ws = True
            self.pos += 1
        return had_ws

    def _parse_sequence(self, stop_chars: Set[str]) -> Node:
        parts: List[Node] = []
        separators: List[bool] = []

        first = True
        while True:
            had_ws = self._consume_ws()
            ch = self._peek()

            if ch is None or ch in stop_chars:
                break

            if not first:
                separators.append(had_ws)

            if ch == "[":
                parts.append(self._parse_optional())
            elif ch == "(":
                parts.append(self._parse_alternatives())
            elif ch == "{":
                parts.append(self._parse_list_ref())
            else:
                parts.append(self._parse_literal())

            first = False

        if not parts:
            return SequenceNode([], [])

        if len(parts) == 1:
            return parts[0]

        return SequenceNode(parts, separators)

    def _parse_optional(self) -> Node:
        self._expect("[")
        child = self._parse_sequence(stop_chars={"]"})
        self._expect("]")
        return OptionalNode(child)

    def _parse_alternatives(self) -> Node:
        self._expect("(")
        options: List[Node] = []

        while True:
            option = self._parse_sequence(stop_chars={"|", ")"})
            options.append(option)

            ch = self._peek()
            if ch == "|":
                self._take()
                continue
            if ch == ")":
                break
            raise ValueError(f"Expected '|' or ')' at position {self.pos}")

        self._expect(")")
        return AlternativesNode(options)

    def _parse_list_ref(self) -> Node:
        self._expect("{")
        start_pos = self.pos

        while True:
            ch = self._peek()
            if ch is None:
                raise ValueError("Unterminated '{'")
            if ch == "}":
                break
            self.pos += 1

        content = self.text[start_pos : self.pos].strip()
        self._expect("}")

        if not content:
            raise ValueError("Empty list reference {} is not allowed")

        # Inline numeric range, hassil-compatible: {from..to[,step][:slot_name]}.
        # The step delimiter accepts ',' (hassil) or '/' (legacy). The optional
        # ':slot_name' suffix is how hassil binds the matched number to a slot;
        # it is irrelevant to the acoustic grammar, so we accept and ignore it.
        range_body = content
        if ":" in content:
            maybe_range, _, _slot = content.rpartition(":")
            if re.fullmatch(
                r"-?\d+\s*\.\.\s*-?\d+(?:\s*[,/]\s*-?\d+)?", maybe_range.strip()
            ):
                range_body = maybe_range.strip()
        m_single = re.fullmatch(
            r"(-?\d+)\s*\.\.\s*(-?\d+)(?:\s*[,/]\s*(-?\d+))?", range_body
        )
        if m_single:
            step = int(m_single.group(3)) if m_single.group(3) else 1
            if step == 0:
                raise ValueError(f"Step cannot be zero: {content}")
            return NumberRangeNode(
                [(int(m_single.group(1)), int(m_single.group(2)), step)]
            )

        # Otherwise a comma-separated numeric list ({1, 5, 10}) or a named list.
        items: List[Tuple[int, ...]] = []
        parts = [p.strip() for p in content.split(",")]
        is_numeric = True

        for part in parts:
            if not part:
                is_numeric = False
                break

            m_range = re.fullmatch(r"(-?\d+)\s*\.\.\s*(-?\d+)(/\s*(-?\d+))?", part)
            if m_range:
                start = int(m_range.group(1))
                end = int(m_range.group(2))
                step_str = m_range.group(4)
                step = int(step_str) if step_str else 1
                if step == 0:
                    raise ValueError(f"Step cannot be zero: {part}")
                items.append((start, end, step))
                continue

            m_num = re.fullmatch(r"(-?\d+)", part)
            if m_num:
                items.append((int(m_num.group(1)),))
                continue

            is_numeric = False
            break

        if is_numeric:
            return NumberRangeNode(items)

        # Named list reference, e.g. {device}.
        return ListRefNode(content)

    def _parse_literal(self) -> Node:
        start = self.pos
        while True:
            ch = self._peek()
            if ch is None or ch.isspace() or ch in "[](){}|":
                break
            self.pos += 1

        text = self.text[start : self.pos]
        if not text:
            raise ValueError(f"Expected literal at position {self.pos}")

        return LiteralNode(text)


# ----------------------------
# Analysis helpers
# ----------------------------


def _node_can_be_empty(node: Node) -> bool:
    if isinstance(node, SequenceNode):
        return all(_node_can_be_empty(part) for part in node.parts)
    if isinstance(node, LiteralNode):
        return False
    if isinstance(node, OptionalNode):
        return True
    if isinstance(node, AlternativesNode):
        return any(_node_can_be_empty(option) for option in node.options)
    if isinstance(node, (ListRefNode, NumberRangeNode)):
        return False
    raise TypeError(f"Unsupported node type: {type(node).__name__}")


def _word_fragments(node: Node) -> Optional[List[str]]:
    """Return every space-free string this node can expand to, or ``None`` if it
    cannot be a single-word fragment (contains whitespace or resolves to list /
    number-range values that must be tokenized as whole phrases).

    This is what lets optional and alternative pieces glued to neighboring text
    (``[un]locked``, ``cancel[l]ed``, ``minute[s]``) be expanded into full-word
    variants so a subword tokenizer sees each complete word at once.
    """
    if isinstance(node, LiteralNode):
        return [node.text]
    if isinstance(node, OptionalNode):
        child = _word_fragments(node.child)
        if child is None:
            return None
        return [""] + child
    if isinstance(node, AlternativesNode):
        out: List[str] = []
        for option in node.options:
            frags = _word_fragments(option)
            if frags is None:
                return None
            out.extend(frags)
        return out
    if isinstance(node, SequenceNode):
        # Only joinable into one word if no part is separated by whitespace.
        if any(node.separators):
            return None
        result = [""]
        for part in node.parts:
            frags = _word_fragments(part)
            if frags is None:
                return None
            result = [prefix + frag for prefix in result for frag in frags]
        return result
    # ListRefNode / NumberRangeNode expand to whole phrases, not word fragments.
    return None


def _run_word_variants(parts: Sequence[Node]) -> List[str]:
    """Cartesian concatenation of the fragments of a glued run of parts, with
    duplicates removed (every part must be fragmentable)."""
    result = [""]
    for part in parts:
        frags = _word_fragments(part)
        assert frags is not None  # caller guarantees fragmentable parts
        result = [prefix + frag for prefix in result for frag in frags]

    seen: Set[str] = set()
    variants: List[str] = []
    for variant in result:
        if variant not in seen:
            seen.add(variant)
            variants.append(variant)
    return variants


def _node_can_start_with_word(node: Node) -> bool:
    if isinstance(node, SequenceNode):
        for part in node.parts:
            if _node_can_start_with_word(part):
                return True
            if not _node_can_be_empty(part):
                return False
        return False
    if isinstance(node, LiteralNode):
        return True
    if isinstance(node, OptionalNode):
        return _node_can_start_with_word(node.child)
    if isinstance(node, AlternativesNode):
        return any(_node_can_start_with_word(option) for option in node.options)
    if isinstance(node, (ListRefNode, NumberRangeNode)):
        return True
    raise TypeError(f"Unsupported node type: {type(node).__name__}")


# ----------------------------
# Token helpers
# ----------------------------


def _add_token_ids(fst: Fst, state: int, token_ids: Sequence[int]) -> int:
    if not token_ids:
        raise ValueError("Tokenizer returned no ids")

    for token_id in token_ids:
        label = int(token_id)
        state = fst.next_edge(state, in_label=label, out_label=label)

    return state


def _add_token_lattice(
    fst: Fst, start_state: int, n_positions: int, edges: Sequence[Tuple[int, int, int]]
) -> int:
    """Splice a subword-segmentation lattice (see ``segment_lattice``) into the
    FST, so the grammar accepts *any* valid tokenization of a word rather than
    one hard-coded segmentation. Character position ``0`` maps to ``start_state``;
    returns the state for position ``n_positions`` (the word's end)."""
    if n_positions <= 0:
        raise ValueError("Empty token lattice")

    pos_state: Dict[int, int] = {0: start_state}

    def state_for(pos: int) -> int:
        state = pos_state.get(pos)
        if state is None:
            state = fst.next_state()
            pos_state[pos] = state
        return state

    for i, j, token_id in edges:
        label = int(token_id)
        fst.add_edge(state_for(i), state_for(j), in_label=label, out_label=label)

    return state_for(n_positions)


def _tokenize_words(tokenizer: TokenizerLike, words: Sequence[str]) -> List[int]:
    if not words:
        raise ValueError("Expected at least one word")

    for word in words:
        if not word:
            raise ValueError("Word cannot be empty")
        if any(ch.isspace() for ch in word):
            raise ValueError(f"Expected pre-split words, got: {word!r}")

    # Important: tokenize the complete phrase so word boundaries are seen
    # by the tokenizer normally, but only after we've assembled whole words.
    text = " ".join(words)
    token_ids = tokenizer.text_to_ids(text)
    if not token_ids:
        raise ValueError(f"Tokenizer returned no ids for text: {text!r}")

    return token_ids


def _expand_list_value(value: str) -> List[str]:
    words = value.strip().split()
    if not words:
        raise ValueError(f"List value must not be empty: {value!r}")
    return words


@lru_cache(maxsize=None)
def _spellout_words(num: int, locale: str) -> Tuple[str, ...]:
    """Spell out ``num`` for ``locale`` as a tuple of pronounceable words.

    icu inserts Unicode *format* characters as syllable hints in some locales
    (e.g. German "ein­und­zwanzig" with soft hyphens, Thai with
    zero-width spaces). These are not pronounced and are absent from acoustic
    alphabets, so a character tokenizer would reject them. Drop all format (Cf)
    characters, then treat a real hyphen as a word boundary (e.g. English
    "twenty-one").

    Memoized: the ICU spellout dominates FST build time and a range like
    ``{0..100}`` reused across hundreds of templates asks for the same numbers
    over and over. The result is treated as immutable by callers (each copies it
    via ``list(words)`` before use)."""
    text = spellout(num, locale)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return tuple(unicodedata.normalize("NFC", text).replace("-", " ").split())


# Numerals that inflect for the gender of the noun they count. ICU has separate
# rulesets for these (%spellout-cardinal-feminine and friends) but the icu_rbnf
# binding exposes no ruleset argument, so it always returns a single reading --
# the masculine. A grammar built from that has no path for the other forms:
# Czech "nastav časovač na dvě hodiny" (feminine "hodiny" needs "dvě", not
# "dva") cannot be decoded at all, however clearly it is spoken. Only the low
# numerals inflect; from five up the counted noun takes the genitive plural and
# the numeral stops changing.
_NUMBER_VARIANTS: Dict[str, Dict[int, Tuple[str, ...]]] = {
    "cs": {1: ("jeden", "jedna", "jedno"), 2: ("dva", "dvě")},
    "sk": {1: ("jeden", "jedna", "jedno"), 2: ("dva", "dve")},
    "ru": {1: ("один", "одна", "одно"), 2: ("два", "две")},
    "uk": {1: ("один", "одна", "одне"), 2: ("два", "дві")},
    "pl": {1: ("jeden", "jedna", "jedno"), 2: ("dwa", "dwie")},
    "hr": {1: ("jedan", "jedna", "jedno"), 2: ("dva", "dvije")},
    "sl": {1: ("en", "ena", "eno"), 2: ("dva", "dve")},
}


def _spellout_variants(num: int, locale: str) -> Tuple[Tuple[str, ...], ...]:
    """Every spoken form of ``num``, each as a tuple of words.

    Usually just what ICU gives. Where a language's low numerals agree in
    gender, every form is offered so the grammar accepts whichever one the
    speaker's noun requires; the FST just gains an alternative branch.
    """
    base = _spellout_words(num, locale)
    variants = _NUMBER_VARIANTS.get(
        locale.split("-")[0].split("_")[0].lower(), {}
    ).get(num)
    if not variants:
        return (base,)
    return tuple(dict.fromkeys((base, *((v,) for v in variants))))


def _expand_ref(
    node: Node, list_values: Mapping[str, Sequence[str]], locale: str
) -> List[Sequence[str]]:
    if isinstance(node, ListRefNode):
        if node.name not in list_values:
            raise KeyError(f"Missing values for list '{node.name}'")
        return [_expand_list_value(v) for v in list_values[node.name]]

    if isinstance(node, NumberRangeNode):
        range_words: List[Sequence[str]] = []
        for item in node.items:
            if len(item) == 1:
                range_words.extend(_spellout_variants(item[0], locale))
            else:
                start, end, step = item
                if start <= end:
                    step = abs(step)
                else:
                    step = -abs(step)

                for num in range(start, end + step, step):
                    range_words.extend(_spellout_variants(num, locale))

        return range_words

    raise TypeError(f"Unsupported reference node: {type(node).__name__}")


# ----------------------------
# Text expansion helpers
# ----------------------------

Config = Tuple[int, int]  # (state, status)


def _compile_text_words(
    fst: Fst,
    configs: Set[Config],
    words: Sequence[str],
    tokenizer: TokenizerLike,
    prepend_space_if_needed: bool,
) -> Set[Config]:
    if not words:
        return set(configs)

    out: Set[Config] = set()

    # Subword tokenizers expose a segmentation lattice so the grammar can accept
    # every valid tokenization of a word (e.g. both "percent" and "per cent"),
    # not just the one greedy/canonical segmentation -- otherwise the acoustic
    # model's natural output may fail to match and the constrained decode is
    # penalized. (The SPM-space prep in segment_lattice makes leading spacing a
    # no-op, exactly as it is for text_to_ids, so STATUS_WORD spacing needs no
    # special handling here.) Tokenizers without a lattice (e.g. the character
    # tokenizer) keep the single-segmentation path.
    lattice = getattr(tokenizer, "segment_lattice", None)

    for state, status in configs:
        actual_words = list(words)

        if lattice is not None:
            n_positions, edges = lattice(" ".join(actual_words))
            new_state = _add_token_lattice(fst, state, n_positions, edges)
            out.add((new_state, STATUS_WORD))
            continue

        # Preserve template whitespace semantics:
        # if the prior emitted thing ended in a word and template syntax
        # says a separator is needed, realize that as a normal space in the
        # text fed to the tokenizer.
        if prepend_space_if_needed and status == STATUS_WORD:
            token_ids = tokenizer.text_to_ids(" " + " ".join(actual_words))
            if not token_ids:
                raise ValueError(
                    f"Tokenizer returned no ids for text: {' '.join(actual_words)!r}"
                )
        else:
            token_ids = _tokenize_words(tokenizer, actual_words)

        new_state = _add_token_ids(fst, state, token_ids)
        out.add((new_state, STATUS_WORD))

    return out


# ----------------------------
# Compiler
# ----------------------------


def _merge_frontier(fst: Fst, configs: Set[Config]) -> Set[Config]:
    """Collapse the NFA frontier to a single state per status.

    Branching is carried in the frontier *set*, so a ``{list}``/``{range}`` node
    fans out one branch from *every* current config -- turning e.g.
    ``set {area} brightness to {0..100}`` into ``|area| * 101`` parallel paths
    (a Cartesian blow-up across the whole template). Adding an epsilon arc from
    each frontier state into one shared merge state factors that: the list then
    fans out from a single state (``V`` branches, not ``C * V``), and the tail
    after it is built once. The epsilon merge states are removed by
    ``RmEpsilon`` + minimization in the native grammar build, so the compiled
    grammar is unchanged -- only far cheaper to construct.

    Merges are per status: two paths that differ in whether a word was just
    emitted (``STATUS_WORD`` vs ``STATUS_EMPTY``) must not share a state, or the
    next word's spacing/tokenization would be wrong. A status group that already
    has a single state is left untouched (no useless epsilon)."""
    by_status: Dict[int, List[int]] = defaultdict(list)
    for state, status in configs:
        by_status[status].append(state)

    merged: Set[Config] = set()
    for status, states in by_status.items():
        if len(states) <= 1:
            merged.add((states[0], status))
            continue
        merge_state = fst.next_state()
        for state in states:
            fst.add_edge(state, merge_state, EPS, EPS)
        merged.add((merge_state, status))
    return merged


def _compile_node(
    fst: Fst,
    node: Node,
    configs: Set[Config],
    list_values: Mapping[str, Sequence[str]],
    locale: str,
    tokenizer: TokenizerLike,
    needs_leading_space: bool = False,
) -> Set[Config]:
    out: Set[Config]

    if isinstance(node, SequenceNode):
        current = set(configs)

        i = 0
        while i < len(node.parts):
            part = node.parts[i]

            # The first child inherits whether a separator is needed before the
            # whole sequence (e.g. a nested sequence inside "[... ...]"); later
            # children use the in-sequence separators. Char tokenizers depend on
            # this for word boundaries (SentencePiece masks it via its marker).
            needs_space = needs_leading_space if i == 0 else node.separators[i - 1]

            # --- Combine a maximal run of glued word fragments ---
            # Consecutive parts with no separator between them form a single
            # spoken word, where the optional/alternative pieces may appear at
            # the start ("[un]locked"), the middle ("cancel[l]ed") or the end
            # ("minute[s]"). Subword tokenizers must see each whole word at once,
            # so expand the pieces into full-word variants and tokenize each
            # variant as a unit instead of piece by piece.
            if _word_fragments(part) is not None:
                j = i
                while (
                    j + 1 < len(node.parts)
                    and not node.separators[j]  # no space before the next part
                    and _word_fragments(node.parts[j + 1]) is not None
                ):
                    j += 1

                if j > i:
                    new_configs: Set[Config] = set()
                    for variant in _run_word_variants(node.parts[i : j + 1]):
                        new_configs |= _compile_text_words(
                            fst,
                            current,
                            variant.split(),
                            tokenizer,
                            prepend_space_if_needed=needs_space,
                        )

                    current = new_configs
                    i = j + 1
                    continue

            # --- Normal path ---
            current = _compile_node(
                fst,
                part,
                current,
                list_values,
                locale,
                tokenizer,
                needs_leading_space=needs_space,
            )

            i += 1

        return current

    if isinstance(node, LiteralNode):
        return _compile_text_words(
            fst,
            configs,
            [node.text],
            tokenizer,
            prepend_space_if_needed=needs_leading_space,
        )

    if isinstance(node, OptionalNode):
        taken = _compile_node(
            fst,
            node.child,
            set(configs),
            list_values,
            locale,
            tokenizer,
            needs_leading_space=needs_leading_space,
        )
        return set(configs) | taken

    if isinstance(node, AlternativesNode):
        out = set()
        for option in node.options:
            out |= _compile_node(
                fst,
                option,
                set(configs),
                list_values,
                locale,
                tokenizer,
                needs_leading_space=needs_leading_space,
            )
        return out

    if isinstance(node, (ListRefNode, NumberRangeNode)):
        # Collapse the incoming frontier so the list fans out from one state
        # (C*V -> C+V), then collapse its own output so the tail after it is
        # built once instead of once per value. See _merge_frontier.
        configs = _merge_frontier(fst, configs)
        expansions = _expand_ref(node, list_values, locale)
        out = set()
        for words in expansions:
            out |= _compile_text_words(
                fst,
                configs,
                words,
                tokenizer,
                prepend_space_if_needed=needs_leading_space,
            )
        return _merge_frontier(fst, out)

    raise TypeError(f"Unsupported node type: {type(node).__name__}")


# ----------------------------
# Public API
# ----------------------------


def templates_to_fst(
    templates: Sequence[str],
    tokenizer: TokenizerLike,
    locale: str,
    list_values: Optional[Mapping[str, Sequence[str]]] = None,
) -> Fst:
    """Parse templates and compile them into an FST whose arc labels are
    integer tokenizer ids (epsilon is :data:`EPS`).

    Behavior:
    - literals are tokenized as complete words, never character-by-character
    - no explicit space arcs are emitted; required template whitespace is folded
      into the text passed to ``tokenizer.text_to_ids(...)`` so the tokenizer sees
      a normal word boundary
    - list values and spoken-out numbers are split into words first, then
      tokenized as full phrases
    """
    if list_values is None:
        list_values = {}

    fst = Fst()

    for template in templates:
        parser = TemplateParser(template)
        ast = parser.parse()

        template_state = fst.next_edge(fst.start, EPS, EPS)

        end_configs = _compile_node(
            fst,
            ast,
            {(template_state, STATUS_EMPTY)},
            list_values,
            locale,
            tokenizer,
        )

        for state, _status in end_configs:
            fst.accept(state)

    return fst
