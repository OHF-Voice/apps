#!/usr/bin/env python3
"""Speech-to-Phrase add-on web UI.

Runnable locally:

    .venv/bin/python src/app.py --data ./data --port 8099
    # then open http://localhost:8099

Lets the user, per language:
  * enable/disable built-in slot-combinations,
  * write their own (speech-to-text only) commands, and
  * switch a device/area/floor off for voice entirely.
On save it persists the choice and (if a model is configured) retrains the
grammar. Works behind Home Assistant ingress and standalone.

Scope note: this is the speech-to-text half of the add-on. The intent
recognizer (``intent_server.py``) is present but not started unless
``--intent`` is passed -- Home Assistant handles the transcript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix

import custom_commands as cc
import debug_log
import extra_sentences as ex
import hass_sentences as hs
import models
import numeric_ranges
import overrides
import presets as bi
import s2p_intents
import settings
import training
import wyoming_server

_LOGGER = logging.getLogger("speech-to-phrase")
ADDON_ROOT = Path(__file__).resolve().parent.parent

if TYPE_CHECKING:
    import sources

JsonDict = Dict[str, Any]
EntityRecords = List[JsonDict]
RouteResult = Union[Response, Tuple[Response, int]]
StartResponse = Callable[..., Any]
WSGIApp = Callable[[Dict[str, Any], StartResponse], Iterable[bytes]]
_ENTITY_CACHE = ".ha-entity-records.json"
_AREA_FLOOR_CACHE = ".ha-area-floor-lists.json"


class HomeAssistantUnavailable(RuntimeError):
    """Live registry data and a last-known-good cache are both unavailable."""


@lru_cache(maxsize=1)
def supported_languages() -> Tuple[str, ...]:
    """Languages that ship Speech-to-Phrase templates. Cached: it is the
    allowlist every request validates against, and it cannot change without a
    restart (it comes from the installed home-assistant-intents package)."""
    return tuple(bi.languages())


def _known_lang(lang: str) -> bool:
    """Whether `lang` is a language we could serve.

    ``<data>/<lang>/`` is a path join, so a value like "../../x" reaches outside
    the data directory; the set of real languages is small, closed and known, so
    an allowlist is the whole defence. The web routes have a stricter test still
    (it must be *the* configured language), which leaves this for the watch loop,
    where the candidates are directory names found on disk."""
    return lang in supported_languages()


def _check_language(language: str) -> None:
    """Refuse to start on a language the add-on cannot serve, and say why.

    Without this the failure surfaced as ``ValueError: No sentence templates
    provided`` from deep inside training, which took the container down on every
    start. Five of the languages models.py maps a model for (zh, ru, hr, hi, sl)
    have no templates in the package yet, so this is a configuration a user can
    reach from the add-on options page."""
    langs = supported_languages()
    if language in langs:
        return
    if language in models.MODEL_NAMES:
        detail = (
            f"'{language}' has an acoustic model but no Speech-to-Phrase "
            f"sentence templates yet, so there is nothing it could "
            f"recognize."
        )
    else:
        detail = f"'{language}' is not a Speech-to-Phrase language."
    raise SystemExit(
        f"{detail} Set 'language' in the add-on options to one of: "
        f"{', '.join(langs)}."
    )


class IngressPrefixMiddleware:
    """Strip Home Assistant's X-Ingress-Path prefix so url_for/fetch work both
    behind ingress and standalone."""

    def __init__(self, app: WSGIApp) -> None:
        self.app = app

    def __call__(
        self, environ: Dict[str, Any], start_response: StartResponse
    ) -> Iterable[bytes]:
        prefix = environ.get("HTTP_X_INGRESS_PATH", "")
        if prefix:
            environ["SCRIPT_NAME"] = prefix
            path = environ.get("PATH_INFO", "")
            if path.startswith(prefix):
                environ["PATH_INFO"] = path[len(prefix) :] or "/"
        return self.app(environ, start_response)


def create_app(cfg: argparse.Namespace) -> Flask:
    # Resolve auto selection and the former "citrinet" spelling to a canonical
    # backend before model/default lookup and grammar fingerprinting.
    cfg.backend = models.resolve_backend(cfg.language, cfg.backend)
    # Gate default depends on the selected model/backend. Only applied when the
    # user left it unset. The word-insertion reward remains per-backend.
    if getattr(cfg, "max_score", None) is None:
        configured_model = cfg.model or models.model_name_for(cfg.language, cfg.backend)
        cfg.max_score = models.default_max_score(cfg.backend, configured_model)
    if getattr(cfg, "token_bonus", None) is None:
        cfg.token_bonus = models.default_token_bonus(cfg.backend)
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    app.wsgi_app = ProxyFix(  # type: ignore[method-assign]
        app.wsgi_app, x_proto=1, x_host=1
    )
    app.wsgi_app = IngressPrefixMiddleware(app.wsgi_app)  # type: ignore[method-assign]

    meta = bi.load_intents_meta()
    data_dir = Path(cfg.data)

    # Resolve / download the acoustic model. None => UI-only (saves persist but
    # don't retrain).
    try:
        resolved = models.resolve(
            cfg.model, Path(cfg.models_dir), cfg.language, cfg.backend
        )
        cfg.model = str(resolved) if resolved else None
    except Exception:  # noqa: BLE001
        _LOGGER.exception("model provisioning failed; UI will run without retraining")
        cfg.model = None

    if (
        cfg.language == "en"
        and cfg.model
        and Path(cfg.model).name in models.ENGLISH_MODEL_ALIASES
    ):
        settings.migrate_max_score_default(
            cfg.data,
            cfg.language,
            model_id=Path(cfg.model).name,
            previous_model_ids=models.LEGACY_ENGLISH_MODELS,
            previous_default=5.0,
            new_default=models.default_max_score(cfg.backend, cfg.model),
        )

    # Train the configured language now if its inputs changed (first boot,
    # entity/area/floor renames, config edits), then watch for further changes.
    if cfg.model:
        try:
            records = _current_records(cfg)
            slot_lists = _current_slot_lists(cfg)
        except HomeAssistantUnavailable:
            grammar = data_dir / cfg.language / "grammar.fst"
            if not grammar.exists():
                raise
            _LOGGER.warning(
                "Home Assistant registry unavailable; serving the existing "
                "grammar until the next successful refresh"
            )
        else:
            _ensure_trained(
                cfg,
                cfg.language,
                meta,
                records,
                slot_lists,
                data_dir,
            )
        _start_watch(cfg, meta, data_dir)

    # ---- per-language persistence -------------------------------------------
    def lang_dir(lang: str) -> Path:
        d = data_dir / lang
        d.mkdir(parents=True, exist_ok=True)
        return d

    def read_enabled(lang: str, combos: List[JsonDict]) -> List[List[Any]]:
        f = lang_dir(lang) / "enabled.json"
        if f.exists():
            return json.loads(f.read_text())
        return bi.default_enabled(combos, cfg.default_importance)  # first-run default

    # ---- routes --------------------------------------------------------------
    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    def wrong_lang(lang: str) -> Tuple[Response, int]:
        """409 for a request about some language other than the configured one.

        Only reachable from a page loaded before the `language` option changed:
        the UI edits whatever `--language` says and offers no choice. Refusing is
        better than quietly applying those edits to the current language --
        they were made against a different set of devices and commands."""
        _LOGGER.warning(
            "Rejected request for '%s'; this add-on is configured for '%s'",
            lang,
            cfg.language,
        )
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"this add-on is set to '{cfg.language}', not {lang!r}. "
                    f"Reload the page.",
                }
            ),
            409,
        )

    @app.route("/api/state")
    def api_state() -> RouteResult:
        # The one language the recognizer runs. A `lang` parameter is honoured
        # only to reject a stale page (see wrong_lang); there is nothing to
        # choose between.
        lang = request.args.get("lang") or cfg.language
        if lang != cfg.language:
            return wrong_lang(lang)
        combos = bi.available_combos(lang, meta)
        amap = training.enabled_domain_map(read_enabled(lang, combos))
        ov = _overrides(cfg, lang)
        range_definitions = s2p_intents.range_list_definitions(lang)
        range_overrides = numeric_ranges.load(data_dir, lang, range_definitions)
        records = _current_records(cfg, lang)
        slot_lists = _current_slot_lists(cfg, lang)
        for c in combos:
            key = (c["intent"], c["combo"])
            ce = bi.combo_examples(
                ADDON_ROOT, lang, c["intent"], c["combo"], records, slot_lists
            )
            c["domains"] = ce["domains"]
            c["examples"] = ce["examples"]
            c["shapes"] = ce["shapes"]
            c["examples_by_domain"] = ce["by_domain"]
            c["examples_by_domain_full"] = ce["by_domain_full"]
            c["group"] = bi.intent_group(c["intent"])
            # Grammar cost, per card: one number per targeted domain (the UI
            # splits a combo into one card per domain), or a single number for
            # combos that target none ("what time is it").
            c["cost"] = training.combo_cost(
                lang,
                c["intent"],
                c["combo"],
                None,
                records,
                slot_lists,
                ov=ov,
                range_overrides=range_overrides,
            )
            c["cost_by_domain"] = {
                d: training.combo_cost(
                    lang,
                    c["intent"],
                    c["combo"],
                    d,
                    records,
                    slot_lists,
                    ov=ov,
                    range_overrides=range_overrides,
                )
                for d in c["domains"]
            }
            if key in amap:
                c["enabled"] = True
                allowed = amap[key]
                c["enabled_domains"] = (
                    ce["domains"]
                    if allowed is None
                    else [d for d in ce["domains"] if d in allowed]
                )
            else:
                c["enabled"] = False
                c["enabled_domains"] = []
        present = {c["group"] for c in combos}
        groups = [g for g in bi.group_order() if g in present]
        commands = cc.load(data_dir, lang)
        # Devices/areas/floors are listed *unfiltered* here: the lists tab is
        # where the user switches one off for voice, so a switched-off one still
        # has to be visible (greyed) rather than vanishing.
        raw_lists = _raw_slot_lists(cfg)
        # One row per spoken *name*, carrying the entities behind it. A name is
        # not unique: two devices can share one, and every alias is a name of its
        # own. The grammar and the voice switch are keyed by name, so the entity
        # ids go underneath the row rather than becoming rows of their own.
        by_domain: Dict[str, Dict[str, List[JsonDict]]] = {}
        for rec in sorted(
            _raw_records(cfg),
            key=lambda r: (r["name"].lower(), r.get("entity_id") or ""),
        ):
            sources = by_domain.setdefault(rec["domain"], {}).setdefault(
                rec["name"], []
            )
            src = {"entity_id": rec.get("entity_id"), "alias_of": rec.get("alias_of")}
            if src not in sources:
                sources.append(src)

        # "How are devices/areas/floors used?" -- which commands consume each list.
        area_used, floor_used, name_used = _usage(lang, set(amap), commands)
        devices = {
            d: {
                "values": [
                    {"name": n, "sources": s} for n, s in by_domain.get(d, {}).items()
                ],
                "used_by": name_used.get(d, []),
            }
            for d in sorted(set(by_domain) | set(name_used))
        }
        return jsonify(
            {
                "lang": lang,
                "combos": combos,
                "groups": groups,
                "commands": commands,
                "trainable": bool(cfg.model),
                "max_score": settings.get_max_score(data_dir, lang, cfg.max_score),
                "max_score_default": cfg.max_score,
                "devices_by_domain": devices,
                # Phrases Home Assistant is already listening for, per source,
                # priced so the two switches show what they cost.
                "hass_sentences": _hass_sentences_state(
                    cfg, lang, records, slot_lists, ov
                ),
                "areas": {"values": raw_lists.get("area", []), "used_by": area_used},
                "floors": {"values": raw_lists.get("floor", []), "used_by": floor_used},
                "numeric_ranges": _numeric_ranges_state(
                    lang, set(amap), range_definitions, range_overrides
                ),
                # Voice targeting, round-tripped by the UI. The UI only edits the
                # global on/off switches; anything else already in the document
                # (aliases, per-command exclusions) rides along untouched.
                "overrides": overrides.load_doc(data_dir, lang),
                # Session state, not a setting: off after every restart.
                "debug_mode": debug_log.enabled(),
            }
        )

    @app.route("/api/debug_mode", methods=["POST"])
    def api_debug_mode() -> Response:
        """Toggle debug mode. Applied immediately -- it changes only how the STT
        server reports and whether it answers Home Assistant, not the grammar, so
        making the user save (and retrain) for it would be a lie about the cost.

        Nothing is persisted: while debug mode is on the add-on answers Home
        Assistant with an empty transcript, and a diagnostic that survived a
        restart would leave someone with a mute assistant and no way to guess
        why."""
        body = request.get_json(force=True)
        enabled = debug_log.set_enabled(bool(body.get("enabled")))
        _LOGGER.info(
            "Debug mode %s%s",
            "on" if enabled else "off",
            (
                " -- Home Assistant will receive an empty transcript for every "
                "utterance until it is switched off"
                if enabled
                else ""
            ),
        )
        return jsonify({"ok": True, "debug_mode": enabled})

    @app.route("/api/transcriptions")
    def api_transcriptions() -> RouteResult:
        """Recognitions since `since`, each tagged with the sentence source that
        produced it. Polled by the UI while debug mode is on."""
        lang = request.args.get("lang") or cfg.language
        if lang != cfg.language:
            return wrong_lang(lang)
        try:
            since = int(request.args.get("since", 0))
        except (TypeError, ValueError):
            since = 0
        entries = debug_log.entries(since=since)
        if entries:
            attributor = _attributor(cfg, lang)
            for entry in entries:
                entry["origin"] = (
                    attributor.attribute(entry["text"])
                    if (attributor and entry["text"])
                    else None
                )
        return jsonify(
            {
                "entries": entries,
                "last_id": debug_log.last_id(),
                "debug_mode": debug_log.enabled(),
            }
        )

    @app.route("/api/save", methods=["POST"])
    def api_save() -> RouteResult:
        body = request.get_json(force=True)
        lang = body.get("lang") or cfg.language
        if lang != cfg.language:
            return wrong_lang(lang)
        enabled = [list(e) for e in body.get("enabled", [])]
        commands = body.get("commands", [])
        range_definitions = s2p_intents.range_list_definitions(lang)
        parsed_ranges = None
        if "numeric_ranges" in body:
            try:
                parsed_range_choices = numeric_ranges.parse_payload(
                    body["numeric_ranges"], range_definitions
                )
            except numeric_ranges.NumericRangeError as err:
                return jsonify({"ok": False, "error": str(err)}), 400
            parsed_ranges = numeric_ranges.active_selections(
                parsed_range_choices, range_definitions
            )
        else:
            parsed_range_choices = None

        d = lang_dir(lang)
        (d / "enabled.json").write_text(json.dumps(enabled, indent=2))
        cc.save(data_dir, lang, commands)
        # The UI no longer edits extra phrasings, so only write them when the
        # client actually sent some -- otherwise an existing file would be wiped
        # on every save.
        if body.get("extra_sentences") is not None:
            ex.save(data_dir, lang, body["extra_sentences"])
        if body.get("overrides") is not None:
            overrides.save(data_dir, lang, body["overrides"])
        # Score gate: persisted per-language and hot-reloaded by the STT server
        # (no retrain needed — it only affects runtime gating, not the grammar).
        if body.get("max_score") is not None:
            settings.set_max_score(
                data_dir,
                lang,
                body["max_score"],
                model_id=_model_id(cfg, lang),
            )
        # Home Assistant sentence sources: these change the grammar, so the
        # retrain below is what makes them take effect.
        for source in hs.SOURCES:
            if body.get(source) is not None:
                settings.set_bool(data_dir, lang, source, body[source])
        range_overrides = (
            parsed_ranges
            if parsed_ranges is not None
            else numeric_ranges.load(data_dir, lang, range_definitions)
        )

        entities = _current_records(cfg, lang)
        slot_lists = _current_slot_lists(cfg, lang)
        templates, _ = training.assemble(
            lang,
            enabled,
            commands,
            entities,
            slot_lists,
            extra_sentences=ex.load(data_dir, lang),
            ov=_overrides(cfg, lang),
            hass_sentences=_hass_sentences(cfg, lang),
            range_overrides=range_overrides,
        )
        resp: JsonDict = {
            "ok": True,
            "n_templates": len(templates),
            "trained": False,
        }
        if _model_dir_for(cfg, lang) is not None:
            try:
                # force=True, so a False return means there was nothing to
                # compile. Say that instead of reporting a retrain that did not
                # happen: with every command switched off the grammar on disk is
                # the previous one, and voice keeps answering to it.
                trained = _ensure_trained(
                    cfg,
                    lang,
                    meta,
                    entities,
                    slot_lists,
                    data_dir,
                    force=True,
                    range_overrides=range_overrides,
                )
                if parsed_range_choices is not None:
                    numeric_ranges.save(data_dir, lang, parsed_range_choices)
                resp["trained"] = trained
                resp["message"] = (
                    f"Saved and retrained ({len(templates)} sentences)."
                    if trained
                    else "Saved, but there are no sentences to train — enable at "
                    "least one command, or the previous grammar stays in use."
                )
            except Exception as e:  # noqa: BLE001
                _LOGGER.exception("training failed")
                message = f"Saved, but training failed: {e}"
                if parsed_ranges is not None:
                    message = (
                        "Saved other changes, but kept the previous numeric ranges "
                        f"because training failed: {e}"
                    )
                resp.update(ok=False, message=message)
        else:
            if parsed_range_choices is not None:
                numeric_ranges.save(data_dir, lang, parsed_range_choices)
            resp["message"] = (
                f"Saved ({len(templates)} sentences). No speech model for "
                f"'{lang}' — skipped retrain."
            )
        return jsonify(resp)

    return app


def _load_json(path: Optional[Union[str, Path]], default: Any) -> Any:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as tmp:
            json.dump(value, tmp)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _current_records(
    cfg: argparse.Namespace, lang: Optional[str] = None
) -> EntityRecords:
    """Live enriched entity records (name/domain/device_class/features/area/
    floor): HA registry in the container, fixture/dev otherwise. Re-fetched at
    each training event so renames/adds/feature changes are picked up. Drives the
    entity-aware gating in training/intent_matcher.

    Entities the user switched off for voice are removed here, so gating, the
    examples and the UI all see the same set."""
    return _overrides(cfg, lang).filter_records(_raw_records(cfg))


def _overrides(
    cfg: argparse.Namespace, lang: Optional[str] = None
) -> overrides.Overrides:
    return overrides.load(Path(cfg.data), lang or cfg.language)


def _raw_records(cfg: argparse.Namespace) -> EntityRecords:
    """Entity records straight from Home Assistant / the fixture, before the
    user's voice-targeting overrides are applied."""
    if cfg.hass_token:
        cache_path = Path(cfg.data) / _ENTITY_CACHE
        try:
            recs = training.entity_records_from_hass(cfg.hass_api, cfg.hass_token)
            _LOGGER.debug("Loaded %d entity records from Home Assistant", len(recs))
            _write_json(cache_path, recs)
            return recs
        except Exception as err:  # noqa: BLE001
            cached = _load_json(cache_path, None)
            if not isinstance(cached, list):
                raise HomeAssistantUnavailable(
                    "entity registry fetch failed and no valid cache exists"
                ) from err
            _LOGGER.exception("entity fetch failed; using last-known-good cache")
            return cached
    data = _load_json(cfg.entities_file, None)
    if data is None:
        return training.DEV_ENTITY_RECORDS
    if isinstance(data, dict):  # legacy {name: domain} fixture
        return [{"name": n, "domain": d} for n, d in data.items()]
    return data


def _hass_flags(cfg: argparse.Namespace, lang: str) -> Dict[str, bool]:
    """Whether each Home-Assistant sentence source is on for `lang`: the add-on
    option is the default for every language, the web UI overrides one."""
    return {
        source: settings.get_bool(
            cfg.data, lang, source, bool(getattr(cfg, source, False))
        )
        for source in hs.SOURCES
    }


def _hass_sentences_grouped(cfg: argparse.Namespace, lang: str) -> Dict[str, List[str]]:
    """Phrases Home Assistant is already listening for, by source, honouring the
    two switches. A switched-off source is fetched from neither HA nor cache, so
    turning both off costs nothing.

    Called on every training pass and on every UI load. hass_sentences caches for
    a minute, so a multi-language pass costs one HA round-trip, not one per
    language."""
    flags = _hass_flags(cfg, lang)
    return hs.fetch_grouped(
        cfg.hass_api,
        cfg.hass_token,
        triggers=flags[hs.TRIGGERS],
        answers=flags[hs.ANSWERS],
    )


def _hass_sentences(cfg: argparse.Namespace, lang: str) -> List[str]:
    grouped = _hass_sentences_grouped(cfg, lang)
    return [s for source in hs.SOURCES for s in grouped[source]]


# Source attributor per (language, grammar fingerprint). Built only when the
# debug view asks for one, and thrown away when the grammar is retrained -- an
# attributor from the previous grammar would name a source for a phrase the
# recognizer can no longer produce.
_attributors: Dict[Tuple[str, Optional[str]], Optional[sources.Attributor]] = {}


def _attributor(cfg: argparse.Namespace, lang: str) -> Optional[sources.Attributor]:
    """A sources.Attributor for `lang`'s current grammar, or None if the grammar
    can't be assembled. Cached on the fingerprint the trainer recorded."""
    import sources

    data_dir = Path(cfg.data)
    fp = None
    meta_path = data_dir / lang / "grammar.meta.json"
    if meta_path.exists():
        try:
            fp = json.loads(meta_path.read_text()).get("fingerprint")
        except Exception:  # noqa: BLE001
            fp = None
    key = (lang, fp)
    if key in _attributors:
        return _attributors[key]
    try:
        combos = bi.available_combos(lang, bi.load_intents_meta())
        by_source, list_values = training.assemble_sources(
            lang,
            _read_enabled(data_dir, lang, combos, cfg.default_importance),
            cc.load(data_dir, lang),
            _current_records(cfg, lang),
            _current_slot_lists(cfg, lang),
            extra_sentences=ex.load(data_dir, lang),
            ov=_overrides(cfg, lang),
            hass_sentences=_hass_sentences_grouped(cfg, lang),
            range_overrides=numeric_ranges.load(
                data_dir, lang, s2p_intents.range_list_definitions(lang)
            ),
        )
        attributor = sources.build(by_source, list_values, lang)
    except Exception:  # noqa: BLE001 -- the debug view must not 500
        _LOGGER.exception("could not build the source index for '%s'", lang)
        attributor = None
    _attributors.clear()  # only the current grammar is ever of interest
    _attributors[key] = attributor
    return attributor


def _hass_sentences_state(
    cfg: argparse.Namespace,
    lang: str,
    records: training.EntityInput,
    slot_lists: Dict[str, List[str]],
    ov: overrides.Overrides,
) -> JsonDict:
    """The Home-Assistant sentence sources for the UI: each switch's state, the
    phrases it currently contributes, and what they cost.

    Only a switched-*on* source has phrases to show -- the point of switching one
    off is not to go asking Home Assistant for them."""
    grouped = _hass_sentences_grouped(cfg, lang)
    flags = _hass_flags(cfg, lang)
    out: Dict[str, Any] = {"sources": {}, "phrases": 0}
    for source in hs.SOURCES:
        costs = training.hass_sentence_costs(
            lang, grouped[source], records, slot_lists, ov=ov
        )
        out["sources"][source] = {
            "enabled": flags[source],
            "default": bool(getattr(cfg, source, False)),
            "sentences": costs,
            "phrases": sum(int(c["phrases"]) for c in costs),
        }
        out["phrases"] += out["sources"][source]["phrases"]
    out["available"] = bool(cfg.hass_token)
    return out


def _current_slot_lists(
    cfg: argparse.Namespace, lang: Optional[str] = None
) -> Dict[str, List[str]]:
    """Slot value lists for training/display. area + floor come live from the HA
    registries (all of them); language lists (color, brightness_level, ...) come
    from the fixture/file. Re-fetched per training event so registry edits are
    picked up (and change the fingerprint -> retrain). Areas/floors the user
    switched off for voice are removed."""
    return _overrides(cfg, lang).filter_slot_lists(_raw_slot_lists(cfg))


def _raw_slot_lists(cfg: argparse.Namespace) -> Dict[str, List[str]]:
    lists = {
        k: list(v)
        for k, v in _load_json(cfg.slot_lists_file, training.DEV_SLOT_LISTS).items()
    }
    if cfg.hass_token:
        cache_path = Path(cfg.data) / _AREA_FLOOR_CACHE
        try:
            areas, floors = training.areas_floors_from_hass(
                cfg.hass_api, cfg.hass_token
            )
            lists["area"], lists["floor"] = areas, floors
            _LOGGER.debug("Loaded %d areas, %d floors from HA", len(areas), len(floors))
            _write_json(cache_path, {"area": areas, "floor": floors})
        except Exception as err:  # noqa: BLE001
            cached = _load_json(cache_path, None)
            if not (
                isinstance(cached, dict)
                and isinstance(cached.get("area"), list)
                and isinstance(cached.get("floor"), list)
                and all(isinstance(v, str) for v in cached["area"])
                and all(isinstance(v, str) for v in cached["floor"])
            ):
                raise HomeAssistantUnavailable(
                    "area/floor registry fetch failed and no valid cache exists"
                ) from err
            _LOGGER.exception("area/floor fetch failed; using last-known-good cache")
            lists["area"] = list(cached["area"])
            lists["floor"] = list(cached["floor"])
    return lists


def _read_enabled(
    data_dir: Path,
    lang: str,
    combos: List[JsonDict],
    default_importance: str,
) -> List[List[Any]]:
    f = data_dir / lang / "enabled.json"
    if f.exists():
        return json.loads(f.read_text())
    return bi.default_enabled(combos, default_importance)


def _usage(
    lang: str,
    enabled_set: Set[Tuple[str, str]],
    commands: Sequence[JsonDict],
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Which commands consume each slot list, for the Names view.
    Returns (area_used_by, floor_used_by, {domain: name_used_by})."""
    import re

    def refs(sentence: str) -> Set[str]:
        return set(re.findall(r"\{([^}]+)\}", sentence))

    area_used: List[str] = []
    floor_used: List[str] = []
    name_used: Dict[str, List[str]] = {}

    for intent, combo in sorted(enabled_set):
        blocks = s2p_intents.combo_blocks(lang, intent, combo)
        if not blocks:
            continue
        label = f"{intent}/{combo}"
        ua = uf = False
        domains: set = set()
        for ss in blocks:
            if ss.get("context_area"):
                ua = True
            nd = ss.get("name_domains") or []
            for sentence in ss.get("sentences", []):
                r = refs(sentence)
                ua = ua or ("area" in r)
                uf = uf or ("floor" in r)
                if "name" in r:
                    domains.update(nd)
        if ua:
            area_used.append(label)
        if uf:
            floor_used.append(label)
        for d in domains:
            name_used.setdefault(d, []).append(label)

    for cmd in commands:
        ss_list = cmd.get("sentences") or []
        label = "✎ " + (ss_list[0] if ss_list else "(custom)")
        nd = cmd.get("name_domains") or ["(any)"]
        ua = uf = False
        domains = set()
        for sentence in ss_list:
            r = refs(sentence)
            ua = ua or ("area" in r)
            uf = uf or ("floor" in r)
            if "name" in r:
                domains.update(nd)
        if ua:
            area_used.append(label)
        if uf:
            floor_used.append(label)
        for d in domains:
            name_used.setdefault(d, []).append(label)

    return area_used, floor_used, name_used


def _numeric_ranges_state(
    lang: str,
    enabled_set: Set[Tuple[str, str]],
    definitions: numeric_ranges.Definitions,
    selections: numeric_ranges.Selections,
) -> List[JsonDict]:
    """Numeric package lists used by Speech-to-Phrase commands."""
    used_by: Dict[str, List[str]] = {}
    all_used: Set[str] = set()
    for intent, combo in s2p_intents.combos(lang):
        referenced = s2p_intents.combo_range_lists(lang, intent, combo)
        all_used.update(referenced)
        if (intent, combo) in enabled_set:
            label = f"{intent}/{combo}"
            for name in referenced:
                used_by.setdefault(name, []).append(label)

    rows: List[JsonDict] = []
    for name in sorted(all_used):
        definition = definitions[name]
        selection = selections.get(name)
        package = numeric_ranges.package_values(definition)
        rows.append(
            {
                "name": name,
                "label": name.replace("_", " ").title(),
                "minimum": definition[0],
                "maximum": definition[1],
                "step": definition[2],
                "package_count": len(package),
                "expression": selection.expression if selection else None,
                "count": len(selection.values) if selection else len(package),
                "presets": numeric_ranges.presets(name, definition),
                "used_by": sorted(used_by.get(name, [])),
            }
        )
    return rows


def _custom_commands(data_dir: Path, lang: str) -> list:
    return cc.load(data_dir, lang)


def _model_id(cfg: argparse.Namespace, lang: str) -> str:
    """Which acoustic model `lang` will be trained against, by name.

    Part of the grammar fingerprint, and cheap on purpose: a name lookup, never
    a download, because it is computed on every staleness check.
    """
    if lang == cfg.language and cfg.model:
        return Path(cfg.model).name
    return models.model_name_for(lang, cfg.backend) or ""


def _fingerprint(
    templates: Sequence[str],
    list_values: Mapping[str, Sequence[str]],
    backend: str,
    model_id: str,
) -> str:
    """Hash of everything that determines the grammar -- templates, the
    name/area/floor/list values, AND the model it is compiled for. Changes here
    mean the grammar is stale.

    ``model_id`` matters because a grammar.fst is not portable between models:
    its arc labels are token ids from that model's vocabulary. Swapping the
    model for a language while backend and templates stay put (as the French
    Citrinet -> Conformer change did, 1024 tokens -> 128) left a fingerprint
    match, so no retrain fired and the recognizer returned an empty transcript
    for every utterance, silently and forever."""
    blob = json.dumps(
        {
            "backend": backend,
            "model": model_id,
            "templates": sorted(templates),
            "lists": {k: sorted(v) for k, v in list_values.items()},
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _model_dir_for(cfg: argparse.Namespace, lang: str) -> Optional[Path]:
    if lang == cfg.language and cfg.model:
        return Path(cfg.model)
    try:
        return models.resolve(None, Path(cfg.models_dir), lang, cfg.backend)
    except Exception:  # noqa: BLE001
        _LOGGER.exception("model resolve failed for '%s'", lang)
        return None


def _ensure_trained(
    cfg: argparse.Namespace,
    lang: str,
    meta: JsonDict,
    entities: training.EntityInput,
    slot_lists: Dict[str, List[str]],
    data_dir: Path,
    force: bool = False,
    range_overrides: Optional[numeric_ranges.Selections] = None,
) -> bool:
    """(Re)train `lang` iff the grammar is missing or its input fingerprint
    changed (templates, enabled combos, custom text, or entity/area/floor lists).
    Returns True if it (re)trained. Logged at INFO so retrains are visible."""
    combos = bi.available_combos(lang, meta)
    enabled = _read_enabled(data_dir, lang, combos, cfg.default_importance)
    commands = _custom_commands(data_dir, lang)
    templates, list_values = training.assemble(
        lang,
        enabled,
        commands,
        entities,
        slot_lists,
        extra_sentences=ex.load(data_dir, lang),
        ov=overrides.load(data_dir, lang),
        hass_sentences=_hass_sentences(cfg, lang),
        range_overrides=(
            range_overrides
            if range_overrides is not None
            else numeric_ranges.load(
                data_dir, lang, s2p_intents.range_list_definitions(lang)
            )
        ),
    )
    if not templates:
        # Nothing to compile. The recognition library rejects an empty grammar,
        # and letting that ValueError out of here took the whole add-on down on
        # every start (a restart loop whose only clue was the traceback). A
        # language with no templates is a configuration problem to report, not a
        # crash: main() checks the configured language up front, and this covers
        # the rest -- a stray <data>/<lang>/ directory, or every command switched
        # off in the web UI.
        _LOGGER.warning(
            "No sentences to train for '%s': the grammar was left unchanged. "
            "Enable some commands in the web UI, or check that this language "
            "has Speech-to-Phrase templates.",
            lang,
        )
        return False
    fp = _fingerprint(templates, list_values, cfg.backend, _model_id(cfg, lang))

    d = data_dir / lang
    grammar, meta_path = d / "grammar.fst", d / "grammar.meta.json"
    prev = None
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text()).get("fingerprint")
        except Exception:  # noqa: BLE001
            prev = None
    if grammar.exists() and prev == fp and not force:
        return False  # up to date

    model_dir = _model_dir_for(cfg, lang)
    if model_dir is None:
        _LOGGER.warning("Cannot (re)train '%s': no acoustic model available", lang)
        return False

    reason = (
        "first boot"
        if not grammar.exists()
        else "save" if force else "inputs changed (entities/areas/floors/config)"
    )
    _LOGGER.info(
        "Training grammar for '%s' (%s): %d sentences. This may take a moment.",
        lang,
        reason,
        len(templates),
    )
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "enabled.json").exists():
        (d / "enabled.json").write_text(json.dumps(enabled, indent=2))
    training.train(cfg.backend, model_dir, lang, templates, list_values, grammar)
    meta_path.write_text(
        json.dumps(
            {"fingerprint": fp, "backend": cfg.backend, "n_templates": len(templates)}
        )
    )
    _LOGGER.info("Trained grammar for '%s' -> %s", lang, grammar)
    return True


def _start_watch(cfg: argparse.Namespace, meta: JsonDict, data_dir: Path) -> None:
    """Background timer: periodically re-check the live registry and retrain any
    set-up language whose inputs changed (entity/area/floor renames, adds, …)."""
    interval = cfg.refresh_interval
    if interval <= 0:
        return

    def loop() -> None:
        while True:
            time.sleep(interval)
            try:
                langs = {cfg.language} | {
                    p.name
                    for p in data_dir.iterdir()
                    if p.is_dir()
                    and (p / "grammar.fst").exists()
                    and _known_lang(p.name)
                }
                # Fetch the registry once per pass, then apply each language's
                # own overrides to it: the filtering is per-language, but the
                # Home Assistant round-trip must not be.
                raw_records = _raw_records(cfg)
                raw_lists = _raw_slot_lists(cfg)
                for lang in sorted(langs):
                    ov = overrides.load(data_dir, lang)
                    if _ensure_trained(
                        cfg,
                        lang,
                        meta,
                        ov.filter_records(raw_records),
                        ov.filter_slot_lists(raw_lists),
                        data_dir,
                    ):
                        _LOGGER.info(
                            "Auto-retrained '%s' after a registry/config change", lang
                        )
            except Exception:  # noqa: BLE001
                _LOGGER.exception("watch loop iteration failed")

    threading.Thread(target=loop, name="s2p-watch", daemon=True).start()
    _LOGGER.info("Watching for entity/area/floor changes every %ds", interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATA_DIR", "./data"))
    ap.add_argument("--language", default="en")
    # "auto" resolves to a backend that has a model for the language, which is
    # the only sensible production answer -- Czech ships Coqui and nothing else,
    # so a hardcoded "nemo" here meant no model at all. There is no add-on
    # option for this: nobody configuring a voice assistant wants to choose a
    # CTC topology, and the wrong choice only ever produced a model that would
    # not load. Still switchable on the command line for testing one backend
    # against the other.
    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "nemo", "citrinet", "coqui"],
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("MODEL_DIR"),
        help="model dir (dev) or HuggingFace model name; "
        "if unset, derived from language+backend",
    )
    ap.add_argument(
        "--models-dir", default=os.environ.get("MODELS_DIR", "/data/models")
    )
    ap.add_argument("--entities-file", default=os.environ.get("ENTITIES_FILE"))
    ap.add_argument("--slot-lists-file", default=os.environ.get("SLOT_LISTS_FILE"))
    ap.add_argument(
        "--hass-api", default=os.environ.get("HASS_API", "http://supervisor/core/api")
    )
    ap.add_argument("--hass-token", default=os.environ.get("SUPERVISOR_TOKEN"))
    # Which importance buckets are on the first time a language is set up.
    # "usable" is a deliberate middle -- 24 of 46 combos on German -- because the
    # whole catalogue is a much larger grammar and a larger grammar has more ways
    # to mishear; the rest are one click away in the web UI, per command and per
    # domain. This is a first-run seed, not a setting: once enabled.json exists
    # it is never consulted again, which is exactly why it is not an add-on
    # option (changing one that silently does nothing is worse than not having
    # it). Pass "optional" to exercise every combo the package ships, which is
    # what the round-trip checks in tools/ do when validating a language.
    ap.add_argument(
        "--default-importance",
        default="usable",
        choices=["required", "usable", "complete", "optional"],
    )
    # Phrases Home Assistant is already listening for. On, because a trigger or
    # question answer that isn't in the grammar can never be transcribed and the
    # automation would never fire. Each one widens the grammar and the answer
    # crawl costs a websocket round-trip per automation/script, so both can be
    # switched off -- per language, in the web UI, where the cost is shown next
    # to the switch.
    ap.add_argument(
        "--no-sentence-triggers",
        dest="sentence_triggers",
        action="store_false",
        help="don't add automation sentence-trigger phrases to the grammar",
    )
    ap.add_argument(
        "--no-question-answers",
        dest="question_answers",
        action="store_false",
        help="don't add assist_satellite.ask_question answers to the grammar",
    )
    ap.add_argument(
        "--refresh-interval",
        type=int,
        default=int(os.environ.get("REFRESH_INTERVAL", "600")),
        help="seconds between registry-change checks (0 disables)",
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument(
        "--wyoming-uri", default=os.environ.get("WYOMING_URI", "tcp://0.0.0.0:10300")
    )
    ap.add_argument(
        "--intent-uri", default=os.environ.get("INTENT_URI", "tcp://0.0.0.0:10500")
    )
    ap.add_argument(
        "--max-score",
        type=float,
        default=None,
        help="score gate; if unset, a model/backend default is used "
        "(English Parakeet 3.8, other NeMo CTC 5.0, Coqui 2.0)",
    )
    # Offsets the decoder's bias toward short paths (see models.DEFAULT_TOKEN_BONUS).
    # Unset => the per-backend default (nemo 2.0, coqui 0.0).
    ap.add_argument(
        "--token-bonus",
        type=float,
        default=None,
        help="word-insertion reward per emitted token (0 = off); "
        "if unset, a per-backend default is used "
        "(nemo 2.0, coqui 0.0)",
    )
    ap.add_argument(
        "--no-wyoming", action="store_true", help="UI only (don't serve Wyoming STT)"
    )
    # Off by default: the add-on ships as speech-to-text only, and Home Assistant
    # handles the transcript with its own conversation agent.
    ap.add_argument(
        "--intent",
        action="store_true",
        help="also serve the Wyoming intent service (experimental)",
    )
    ap.add_argument("--debug", action="store_true")
    cfg = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if cfg.debug else logging.INFO)
    _check_language(cfg.language)
    app = create_app(cfg)

    # Serve Wyoming STT alongside the web UI in this same process. It reads the
    # grammar.fst the UI/watch-thread writes (hot-reloaded on change).
    if cfg.model and not cfg.no_wyoming:
        grammar = Path(cfg.data) / cfg.language / "grammar.fst"
        wyoming_server.start_background(
            cfg.wyoming_uri,
            cfg.backend,
            cfg.model,
            cfg.language,
            grammar,
            cfg.max_score,
            cfg.token_bonus,
        )
    elif cfg.no_wyoming:
        _LOGGER.info("Wyoming server disabled (--no-wyoming)")
    else:
        _LOGGER.warning("No model — Wyoming server not started")

    # Optionally serve the Wyoming intent service alongside. It is independent of
    # the acoustic model (text in -> intent out), so it runs even UI-only, but
    # it is off unless asked for: speech-to-text is what this add-on ships.
    if cfg.intent:
        import intent_server

        intent_server.start_background(
            cfg.intent_uri,
            cfg.language,
            Path(cfg.data),
            get_entities=lambda: _current_records(cfg),
            get_slot_lists=lambda: _current_slot_lists(cfg),
            api_url=cfg.hass_api,
            token=cfg.hass_token,
            ttl=max(cfg.refresh_interval, 60),
        )
    else:
        _LOGGER.info(
            "Speech-to-text only; intent service not started (--intent enables it)"
        )

    _LOGGER.info(
        "Speech-to-Phrase UI on http://%s:%s (data=%s, model=%s)",
        cfg.host,
        cfg.port,
        cfg.data,
        cfg.model or "<none>",
    )
    app.run(host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
