import hashlib
import io
import json
import logging
import math
import os
import pickle
import re
import time
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp import __version__ as LLAMA_CPP_VERSION

from util import LRUCache

TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:([a-zA-Z0-9_]+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)
TOOL_ARGS = Dict[str, Any]
TOOL_CALL = Tuple[str, TOOL_ARGS]
MODEL_RESPONSE = Tuple[List[TOOL_CALL], str]
DEFAULT_MAX_TOKENS = 128


DEFAULT_REPO = "ggml-org/gemma-4-E2B-it-GGUF"
DEFAULT_FILENAME = "gemma-4-E2B-it-Q8_0.gguf"
# Both defaults are written already normalized (see ``normalize_prompt``), so
# restoring them in the web UI cannot leave the saved and running prompts one
# rebuild apart.
DEFAULT_SYSTEM_PROMPT = (
    "Call tools for the sentence below.\n"
    "If no tools are called, reply in the language below that you don't understand."
)
DEFAULT_USER_PROMPT = (
    'Sentence: "{text}"\nLanguage: "{language}"\nToday is {weekday}, {date}'
)

# The most a prompt may be, so an edit cannot quietly swallow the context.
MAX_PROMPT_CHARS = 2000

# What ``{...}`` in the user prompt may refer to. Anything the model needs in
# order to resolve "Saturday" or "tomorrow" belongs here rather than in the
# system prompt: the system prompt is the cached prefix, so a date in it would
# be rebuilt (minutes, on a Raspberry Pi) every time the day changed.
PROMPT_PLACEHOLDERS = {
    "text": "the sentence to recognize",
    "language": "the requested response language",
    "date": "the current date, as YYYY-MM-DD",
    "time": "the current time, as HH:MM",
    "datetime": "the current date and time, ISO 8601",
    "weekday": "the current day of the week",
}
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

_LOGGER = logging.getLogger(__name__)


class Gemma4Recognizer:
    def __init__(
        self,
        state_path: Union[str, Path],
        repo_id: str = DEFAULT_REPO,
        filename: str = DEFAULT_FILENAME,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
        cache_size: int = 0,
        n_ctx: Optional[int] = None,
        n_ctx_overhead: int = 128,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        flash_attn: bool = True,
        debug: bool = False,
    ) -> None:
        self.llm: Optional[Llama] = None
        # True once load() has finished, including building or restoring the
        # cached prefix state. `llm` alone is not enough: it is set as soon as
        # the model is constructed, while the prefix rebuild that follows is the
        # slow part (minutes on a Raspberry Pi).
        self.ready = False
        self.state_path = Path(state_path)
        self.repo_id = repo_id
        self.filename = filename
        self.user_prompt = user_prompt
        self.n_ctx = n_ctx
        self.n_ctx_overhead = n_ctx_overhead
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.flash_attn = flash_attn
        self.model_path: Optional[Path] = None
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 1
        self.enable_thinking = False
        self.tool_choice: str = "auto"
        self.tools: Optional[List[Dict[str, Any]]] = None
        self.debug = debug

        self.cache: Optional[LRUCache] = None
        if cache_size > 0:
            self.cache = LRUCache(cache_size)

        self.system_prompt = system_prompt
        self.system_message = {
            "role": "system",
            "content": system_prompt,
        }

    def required_n_ctx(self, tools: List[Dict[str, Any]]) -> int:
        """Context size needed for the fixed prompt plus one utterance.

        Assumes chars / 3 is a conservative upper bound on tokens; the sentence
        itself and the templating are covered by the overhead. Rounded up to a
        multiple of 64.
        """
        with io.StringIO() as prompt_file:
            print(self.system_prompt, file=prompt_file)
            print(self.user_prompt, file=prompt_file)
            json.dump(_sort_tools(tools), prompt_file)
            num_chars = len(prompt_file.getvalue())

        return (
            math.ceil(
                (math.ceil(num_chars / 3) + self.max_tokens + self.n_ctx_overhead) / 64
            )
            * 64
        )

    def _create_llm(self, n_ctx: int) -> None:
        """Download the model if needed and construct llama.cpp."""
        try:
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_files_only=True,
            )
        except OSError:
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_files_only=False,
            )
        _LOGGER.debug("Loading gemma4: %s", model_path)
        self.model_path = Path(model_path).resolve()
        self.llm = Llama(
            model_path=model_path,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
            n_ctx=n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            flash_attn=self.flash_attn,
            verbose=self.debug,
        )

    def _restore_or_build_state(self) -> None:
        """Restore the cached prefix for ``self.tools``, or build and save it."""
        assert self.llm, "Not loaded"
        assert self.tools is not None, "No tools"

        runtime_model_id = (
            f"llama-cpp-python/{LLAMA_CPP_VERSION};"
            f"n_ctx={self.llm.n_ctx()};"
            f"n_gpu_layers={self.n_gpu_layers};"
            f"flash_attn={int(self.flash_attn)};"
            f"model_path={self.model_path};"
            f"{self.repo_id}/{self.filename}"
        )
        actual_tools_hash = _get_tools_hash(
            runtime_model_id, self.tools, self.system_prompt
        )
        state_metadata_path = self.state_path.with_suffix(".sha256")
        if state_metadata_path.exists() and self.state_path.exists():
            expected_tools_hash = state_metadata_path.read_text(
                encoding="utf-8"
            ).strip()
            if expected_tools_hash == actual_tools_hash:
                _LOGGER.debug("Cache hit. Loading state: %s", self.state_path)
                try:
                    with open(self.state_path, "rb") as state_file:
                        state = pickle.load(state_file)

                    self.llm.load_state(state)
                    return
                except Exception as err:  # pylint: disable=broad-except
                    # State is an optimization, never a reason to keep the app
                    # from starting. Rebuild if the file is corrupt or the
                    # runtime rejects it despite matching metadata.
                    _LOGGER.warning(
                        "Could not restore cached llama.cpp state; rebuilding: %s",
                        err,
                    )

        _LOGGER.info("Retraining...")
        if self.cache is not None:
            self.cache.clear()

        start_time = time.monotonic()
        self.llm.create_chat_completion(
            messages=[self.system_message],  # type: ignore
            tools=self.tools,  # type: ignore
            max_tokens=0,
        )
        end_time = time.monotonic()
        _LOGGER.debug("Rebuilt state in %s second(s)", end_time - start_time)

        # Write to a temporary file and swap, so a failure part-way through
        # cannot leave a truncated state behind. The hash is written last: if
        # anything interrupts us, the mismatch just costs a rebuild next start.
        _LOGGER.debug("Saving state: %s", self.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_state_path = self.state_path.with_suffix(".tmp")
        with open(tmp_state_path, "wb") as state_file:
            state = self.llm.save_state()
            pickle.dump(state, state_file)

        os.replace(tmp_state_path, self.state_path)
        state_metadata_path.write_text(actual_tools_hash, encoding="utf-8")

    def load(self, tools: List[Dict[str, Any]]) -> None:
        self.tools = _sort_tools(tools)
        if self.llm is None:
            self._create_llm(
                self.n_ctx
                if self.n_ctx is not None
                else self.required_n_ctx(self.tools)
            )

        self._restore_or_build_state()
        self.ready = True

    def reload(self, tools: List[Dict[str, Any]]) -> None:
        """Swap in a new tool set, keeping the working one if the swap fails.

        Must be called on the same single thread that serves recognition.
        """
        assert self.llm, "Not loaded"
        assert self.tools is not None, "Not loaded"

        new_tools = _sort_tools(tools)
        if new_tools == self.tools:
            _LOGGER.debug("Tools unchanged, nothing to reload")
            return

        old_tools = self.tools

        # An auto-sized context has to grow with the tool set: the model is
        # constructed once, so a larger prompt would otherwise overflow it.
        needed_n_ctx = self.required_n_ctx(new_tools)
        grow_context = (self.n_ctx is None) and (needed_n_ctx > self.llm.n_ctx())

        # Keep both the live object and its prefix so even a failed context
        # growth can roll back without requiring an app restart.
        old_llm = self.llm
        live_state = old_llm.save_state()

        self.ready = False
        if self.cache is not None:
            # Cached responses were produced against the old tools.
            self.cache.clear()

        try:
            self.tools = new_tools
            if grow_context:
                _LOGGER.info(
                    "Growing context from %s to %s token(s) for %s tool(s)",
                    self.llm.n_ctx(),
                    needed_n_ctx,
                    len(new_tools),
                )
                self._create_llm(needed_n_ctx)

            self._restore_or_build_state()
        except Exception:
            self.tools = old_tools
            _LOGGER.exception(
                "Failed to load %s tool(s), keeping the previous %s",
                len(new_tools),
                len(old_tools),
            )
            self.llm = old_llm
            self.llm.load_state(live_state)
            self.ready = True
            raise

        self.ready = True
        _LOGGER.info("Now using %s tool(s)", len(new_tools))

    def set_prompts(self, system_prompt: str, user_prompt: str) -> None:
        """Change the prompts, rebuilding the cached prefix when it is affected.

        The system prompt is the cached prefix, so changing it costs a full
        rebuild -- the same wait as changing the tool set. The user prompt is
        evaluated per utterance, so changing it is free unless it no longer fits
        in the context.

        Must be called on the same single thread that serves recognition.
        """
        assert self.llm, "Not loaded"
        assert self.tools is not None, "Not loaded"

        system_prompt, user_prompt = validate_prompts(system_prompt, user_prompt)
        system_changed = normalize_prompt(self.system_prompt) != system_prompt
        user_changed = normalize_prompt(self.user_prompt) != user_prompt
        if not (system_changed or user_changed):
            return

        old_system_prompt = self.system_prompt
        old_user_prompt = self.user_prompt

        def restore_prompts() -> None:
            self.system_prompt = old_system_prompt
            self.system_message = {"role": "system", "content": old_system_prompt}
            self.user_prompt = old_user_prompt

        self.system_prompt = system_prompt
        self.system_message = {"role": "system", "content": system_prompt}
        self.user_prompt = user_prompt

        needed_n_ctx = self.required_n_ctx(self.tools)
        if (self.n_ctx is not None) and (needed_n_ctx > self.llm.n_ctx()):
            restore_prompts()
            raise ValueError(
                f"These prompts need at least {needed_n_ctx} context tokens, but "
                f"the fixed context is {self.llm.n_ctx()}"
            )

        grow_context = (self.n_ctx is None) and (needed_n_ctx > self.llm.n_ctx())
        if not (system_changed or grow_context):
            # Only the per-utterance prompt moved, and it still fits.
            if self.cache is not None:
                self.cache.clear()
            _LOGGER.info("User prompt updated")
            return

        old_llm = self.llm
        live_state = old_llm.save_state()
        self.ready = False
        try:
            if grow_context:
                _LOGGER.info(
                    "Growing context from %s to %s token(s) for the new prompts",
                    self.llm.n_ctx(),
                    needed_n_ctx,
                )
                self._create_llm(needed_n_ctx)

            self._restore_or_build_state()
        except Exception:
            restore_prompts()
            self.llm = old_llm
            self.llm.load_state(live_state)
            self.ready = True
            raise

        if self.cache is not None:
            self.cache.clear()
        self.ready = True
        _LOGGER.info("Prompts updated")

    def set_max_tokens(self, max_tokens: int) -> None:
        """Change the generation limit, growing an automatic context if needed.

        Must be called on the same single thread that serves recognition.
        """
        assert self.llm, "Not loaded"
        assert self.tools is not None, "Not loaded"
        if max_tokens <= 0:
            raise ValueError("Maximum tokens must be greater than zero")
        if max_tokens == self.max_tokens:
            return

        old_max_tokens = self.max_tokens
        self.max_tokens = max_tokens
        needed_n_ctx = self.required_n_ctx(self.tools)
        if (self.n_ctx is not None) and (needed_n_ctx > self.llm.n_ctx()):
            self.max_tokens = old_max_tokens
            raise ValueError(
                f"Maximum tokens {max_tokens} needs at least {needed_n_ctx} context "
                f"tokens, but the fixed context is {self.llm.n_ctx()}"
            )

        grow_context = (self.n_ctx is None) and (needed_n_ctx > self.llm.n_ctx())
        if not grow_context:
            if self.cache is not None:
                self.cache.clear()
            _LOGGER.info("Maximum generation tokens set to %s", max_tokens)
            return

        old_llm = self.llm
        live_state = old_llm.save_state()
        self.ready = False
        try:
            _LOGGER.info(
                "Growing context from %s to %s token(s) for max_tokens=%s",
                self.llm.n_ctx(),
                needed_n_ctx,
                max_tokens,
            )
            self._create_llm(needed_n_ctx)
            self._restore_or_build_state()
        except Exception:
            self.max_tokens = old_max_tokens
            self.llm = old_llm
            self.llm.load_state(live_state)
            self.ready = True
            raise

        if self.cache is not None:
            self.cache.clear()
        self.ready = True
        _LOGGER.info("Maximum generation tokens set to %s", max_tokens)

    def get_tool_calls(self, text: str, language: str = "en") -> MODEL_RESPONSE:
        assert self.llm, "Not loaded"

        text = text.strip()
        user_content = format_prompt(
            self.user_prompt, prompt_values(text, language, datetime.now())
        )

        # Keyed on the rendered prompt, not the sentence: a prompt that carries
        # the date must not answer tomorrow's "add it for Saturday" with the
        # datetime it worked out today.
        cache_key = f"{language}: {user_content}"
        if self.cache is not None:
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                _LOGGER.debug("Returning cached response (key='%s')", cache_key)
                return cached_response

        start_time = time.monotonic()
        response = cast(
            Dict[str, Any],
            self.llm.create_chat_completion(
                messages=[
                    self.system_message,  # type: ignore
                    {"role": "user", "content": user_content},
                ],
                tools=self.tools,  # type: ignore
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                tool_choice=self.tool_choice,  # type: ignore
            ),
        )
        end_time = time.monotonic()
        _LOGGER.debug("Response in %s second(s): %s", end_time - start_time, response)

        choice = response["choices"][0]
        content = choice["message"]["content"] or ""
        if choice.get("finish_reason") == "length":
            message = (
                f"Model response exceeded the {self.max_tokens}-token generation "
                "limit. Increase the Maximum tokens option for scripts with many "
                "arguments."
            )
            _LOGGER.warning("%s Partial response: %s", message, content)
            return [], message

        model_response = (_parse_tool_calls(content), content)
        if self.cache is not None:
            _LOGGER.debug("Caching '%s' -> %s", cache_key, model_response)
            self.cache.set(cache_key, model_response)

        return model_response

    def describe(self) -> Dict[str, Any]:
        """Return the effective runtime config (for benchmark reports)."""
        n_threads: Optional[int] = None
        n_ctx: Optional[int] = self.n_ctx
        draft_model: Optional[str] = None
        if self.llm is not None:
            n_threads = getattr(self.llm.context_params, "n_threads", None)
            n_ctx = self.llm.n_ctx()
            if getattr(self.llm, "draft_model", None) is not None:
                draft_model = type(self.llm.draft_model).__name__

        return {
            "repo_id": self.repo_id,
            "filename": self.filename,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "flash_attn": self.flash_attn,
            "draft_model": draft_model,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }

    def run_sentences(
        self,
        tools: List[Dict[str, Any]],
        sentences: List[Dict[str, Any]],
        passes: int = 3,
        default_language: str = "en",
    ) -> Dict[str, Any]:
        """Benchmark a fixed tool/sentence fixture without disturbing live state.

        Snapshots the running assistant's state, builds a temporary prefix from
        ``tools``, runs each sentence ``passes`` times (resetting to the warm
        prefix before each timed call so every measurement mirrors a fresh
        per-utterance start in production), then restores the live state. The
        response cache is bypassed entirely. Must be called on the same single
        thread that serves recognition (the model is not thread-safe).
        """
        assert self.llm, "Not loaded"
        passes = max(1, passes)
        sorted_tools = sorted(tools, key=lambda t: t["function"]["name"])

        # Snapshot live state so benchmarking never disturbs the running assistant.
        prod_state = self.llm.save_state()
        try:
            # Build the temporary static prefix for the benchmark tools.
            start_time = time.monotonic()
            self.llm.create_chat_completion(
                messages=[self.system_message],  # type: ignore
                tools=sorted_tools,  # type: ignore
                max_tokens=0,
            )
            rebuild_seconds = time.monotonic() - start_time

            # Snapshot the warm prefix so each sentence starts from the same point.
            prefix_state = self.llm.save_state()

            results: List[Dict[str, Any]] = []
            for sentence in sentences:
                text = str(sentence["text"]).strip()
                language = str(sentence.get("language") or default_language)
                sentence_passes: List[Dict[str, Any]] = []
                content = ""

                for _ in range(passes):
                    # Reset to warm prefix (untimed): mirrors a production
                    # per-utterance start where only the user turn is evaluated.
                    self.llm.load_state(prefix_state)

                    start_time = time.monotonic()
                    response = cast(
                        Dict[str, Any],
                        self.llm.create_chat_completion(
                            messages=[
                                self.system_message,  # type: ignore
                                {
                                    "role": "user",
                                    "content": format_prompt(
                                        self.user_prompt,
                                        prompt_values(text, language, datetime.now()),
                                    ),
                                },
                            ],
                            tools=sorted_tools,  # type: ignore
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=self.top_k,
                            max_tokens=self.max_tokens,
                            tool_choice=self.tool_choice,  # type: ignore
                        ),
                    )
                    latency = time.monotonic() - start_time

                    content = response["choices"][0]["message"]["content"] or ""
                    usage = response.get("usage") or {}
                    sentence_passes.append(
                        {
                            "latency": latency,
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                        }
                    )

                tool_calls = _parse_tool_calls(content)
                results.append(
                    {
                        "text": text,
                        "language": language,
                        "passes": sentence_passes,
                        "content": content,
                        "tool_calls": [
                            {"name": name, "args": args} for name, args in tool_calls
                        ],
                    }
                )
        finally:
            # Always restore the live assistant's state.
            self.llm.load_state(prod_state)

        return {
            "rebuild_seconds": rebuild_seconds,
            "passes": passes,
            "num_tools": len(sorted_tools),
            "config": self.describe(),
            "sentences": results,
        }


# -----------------------------------------------------------------------------


def normalize_prompt(prompt: str) -> str:
    """A prompt with insignificant whitespace removed.

    Prompts are edited in a browser textarea, which brings back line endings and
    trailing spaces of its own. Comparing normalized prompts keeps a form
    round-trip from counting as a change, which would rebuild the prompt cache.
    """
    lines = prompt.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    return "\n".join(line.rstrip() for line in lines)


def unknown_placeholders(prompt: str) -> List[str]:
    """``{...}`` names in a prompt that nothing will ever fill in."""
    return sorted(
        {
            name
            for name in _PLACEHOLDER_RE.findall(prompt)
            if name not in PROMPT_PLACEHOLDERS
        }
    )


def validate_prompts(system_prompt: str, user_prompt: str) -> Tuple[str, str]:
    """Return both prompts normalized, or raise ValueError saying what is wrong."""
    system_prompt = normalize_prompt(system_prompt)
    user_prompt = normalize_prompt(user_prompt)

    for label, prompt in (
        ("System prompt", system_prompt),
        ("User prompt", user_prompt),
    ):
        if not prompt:
            raise ValueError(f"{label} cannot be empty")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(
                f"{label} cannot be longer than {MAX_PROMPT_CHARS} characters"
            )

        unknown = unknown_placeholders(prompt)
        if unknown:
            raise ValueError(
                f"{label} uses unknown placeholder(s) "
                f"{', '.join('{' + name + '}' for name in unknown)}. Available: "
                f"{', '.join('{' + name + '}' for name in sorted(PROMPT_PLACEHOLDERS))}"
            )

    # Without the sentence the model is being asked to recognize nothing.
    if "{text}" not in user_prompt:
        raise ValueError("User prompt must include {text}")

    return system_prompt, user_prompt


def format_prompt(template: str, values: Dict[str, str]) -> str:
    """Fill in the ``{...}`` placeholders a prompt is allowed to use.

    Deliberately not ``str.format``: prompts are user-editable, and a stray
    brace must not turn every command into a KeyError.
    """
    return _PLACEHOLDER_RE.sub(
        lambda match: values.get(match.group(1), match.group(0)), template
    )


def prompt_values(text: str, language: str, now: datetime) -> Dict[str, str]:
    """What the user prompt's placeholders stand for, for one utterance."""
    return {
        "text": text,
        "language": language,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "datetime": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "weekday": now.strftime("%A"),
    }


def _parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    text = _normalize_gemma_tool_text(text)
    calls = []

    for match in TOOL_CALL_RE.finditer(text):
        name = match.group(1)
        raw_args = match.group(2).strip()
        args = {}

        if raw_args:
            valid = True
            for part in _split_args(raw_args):
                if ":" not in part:
                    _LOGGER.warning("Ignoring malformed tool call argument: %r", part)
                    valid = False
                    break
                key, value = part.split(":", 1)
                args[key.strip()] = _parse_value(value)

            if not valid:
                continue

        calls.append((name, args))

    return calls


def _normalize_gemma_tool_text(text: str) -> str:
    return text.replace('<|"|>', '"').replace("<|'|>", "'")


def _parse_value(value: str) -> Any:
    value = _normalize_gemma_tool_text(value.strip())

    # JSON-ish arrays/objects
    if value.startswith(("[", "{")) and value.endswith(("]", "}")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return literal_eval(value)
            except (SyntaxError, ValueError):
                pass

    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]

    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        try:
            return literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]

    if re.fullmatch(r"-?\d+", value):
        return int(value)

    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    return value


def _split_args(raw_args: str) -> List[str]:
    parts = []
    buf = []
    quote = None
    escape = False
    depth = 0

    for ch in raw_args:
        if escape:
            buf.append(ch)
            escape = False
            continue

        if ch == "\\":
            buf.append(ch)
            escape = True
            continue

        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue

        if ch in ('"', "'"):
            buf.append(ch)
            quote = ch
            continue

        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1

        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf.clear()
            continue

        buf.append(ch)

    part = "".join(buf).strip()
    if part:
        parts.append(part)

    return parts


def _sort_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tools in a stable order, so the prefix hash does not depend on ordering."""
    return sorted(tools, key=lambda t: t["function"]["name"])


def _get_tools_hash(
    model_id: str, tools: List[Dict[str, Any]], system_prompt: str
) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_id.encode())
    hasher.update(system_prompt.encode())
    hasher.update(json.dumps(tools).encode())
    return hasher.hexdigest()
