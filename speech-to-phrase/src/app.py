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
import argparse
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, jsonify, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix

import custom_commands as cc
import extra_sentences as ex
import models
import overrides
import presets as bi
import settings
import training
import wyoming_server

_LOGGER = logging.getLogger("speech-to-phrase")
ADDON_ROOT = Path(__file__).resolve().parent.parent


class IngressPrefixMiddleware:
    """Strip Home Assistant's X-Ingress-Path prefix so url_for/fetch work both
    behind ingress and standalone."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        prefix = environ.get("HTTP_X_INGRESS_PATH", "")
        if prefix:
            environ["SCRIPT_NAME"] = prefix
            path = environ.get("PATH_INFO", "")
            if path.startswith(prefix):
                environ["PATH_INFO"] = path[len(prefix):] or "/"
        return self.app(environ, start_response)


def create_app(cfg) -> Flask:
    if cfg.backend == "auto":
        # Pick a backend that actually has a model for this language (Citrinet
        # preferred; Coqui for coqui-only languages like sl/nl/cs).
        cfg.backend = models.resolve_backend(cfg.language, "auto")
    # Gate default depends on the (now-resolved) backend: Citrinet and Coqui use
    # different penalty scales. Only applied when the user left it unset. Same
    # for the word-insertion reward, whose scale is likewise per-backend.
    if getattr(cfg, "max_score", None) is None:
        cfg.max_score = models.default_max_score(cfg.backend)
    if getattr(cfg, "token_bonus", None) is None:
        cfg.token_bonus = models.default_token_bonus(cfg.backend)
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    app.wsgi_app = IngressPrefixMiddleware(app.wsgi_app)

    meta = bi.load_intents_meta()
    data_dir = Path(cfg.data)

    # Resolve / download the acoustic model. None => UI-only (saves persist but
    # don't retrain).
    try:
        resolved = models.resolve(cfg.model, Path(cfg.models_dir), cfg.language, cfg.backend)
        cfg.model = str(resolved) if resolved else None
    except Exception:  # noqa: BLE001
        _LOGGER.exception("model provisioning failed; UI will run without retraining")
        cfg.model = None

    # Train the configured language now if its inputs changed (first boot,
    # entity/area/floor renames, config edits), then watch for further changes.
    if cfg.model:
        _ensure_trained(cfg, cfg.language, meta, _current_records(cfg),
                        _current_slot_lists(cfg), data_dir)
        _start_watch(cfg, meta, data_dir)

    # ---- per-language persistence -------------------------------------------
    def lang_dir(lang: str) -> Path:
        d = data_dir / lang
        d.mkdir(parents=True, exist_ok=True)
        return d

    def read_enabled(lang: str, combos: List[dict]) -> List[List[str]]:
        f = lang_dir(lang) / "enabled.json"
        if f.exists():
            return json.loads(f.read_text())
        return bi.default_enabled(combos, cfg.default_importance)  # first-run default

    # ---- routes --------------------------------------------------------------
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/languages")
    def api_languages():
        langs = bi.languages(ADDON_ROOT)
        return jsonify({"languages": langs, "default": cfg.language if cfg.language in langs else (langs[0] if langs else None)})

    @app.route("/api/state")
    def api_state():
        lang = request.args.get("lang", cfg.language)
        combos = bi.available_combos(ADDON_ROOT, lang, meta)
        amap = training.enabled_domain_map(read_enabled(lang, combos))
        ov = _overrides(cfg, lang)
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
                ADDON_ROOT, lang, c["intent"], c["combo"], None, records,
                slot_lists, ov=ov
            )
            c["cost_by_domain"] = {
                d: training.combo_cost(
                    ADDON_ROOT, lang, c["intent"], c["combo"], d, records,
                    slot_lists, ov=ov
                )
                for d in c["domains"]
            }
            if key in amap:
                c["enabled"] = True
                allowed = amap[key]
                c["enabled_domains"] = (
                    ce["domains"] if allowed is None
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
        by_domain: Dict[str, Dict[str, List[dict]]] = {}
        for rec in sorted(_raw_records(cfg),
                          key=lambda r: (r["name"].lower(), r.get("entity_id") or "")):
            sources = by_domain.setdefault(rec["domain"], {}).setdefault(rec["name"], [])
            src = {"entity_id": rec.get("entity_id"), "alias_of": rec.get("alias_of")}
            if src not in sources:
                sources.append(src)

        # "How are devices/areas/floors used?" -- which commands consume each list.
        area_used, floor_used, name_used = _usage(lang, set(amap), commands)
        devices = {
            d: {
                "values": [{"name": n, "sources": s}
                           for n, s in by_domain.get(d, {}).items()],
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
                "areas": {"values": raw_lists.get("area", []), "used_by": area_used},
                "floors": {"values": raw_lists.get("floor", []), "used_by": floor_used},
                # Voice targeting, round-tripped by the UI. The UI only edits the
                # global on/off switches; anything else already in the document
                # (aliases, per-command exclusions) rides along untouched.
                "overrides": overrides.load_doc(data_dir, lang),
            }
        )

    @app.route("/api/save", methods=["POST"])
    def api_save():
        body = request.get_json(force=True)
        lang = body["lang"]
        enabled = [list(e) for e in body.get("enabled", [])]
        commands = body.get("commands", [])
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
            settings.set_max_score(data_dir, lang, body["max_score"])

        entities = _current_records(cfg, lang)
        slot_lists = _current_slot_lists(cfg, lang)
        templates, _ = training.assemble(
            ADDON_ROOT, lang, enabled, commands, entities, slot_lists,
            extra_sentences=ex.load(data_dir, lang), ov=_overrides(cfg, lang),
        )
        resp = {"ok": True, "n_templates": len(templates), "trained": False}
        if _model_dir_for(cfg, lang) is not None:
            try:
                _ensure_trained(cfg, lang, meta, entities, slot_lists, data_dir, force=True)
                resp["trained"] = True
                resp["message"] = f"Saved and retrained ({len(templates)} sentences)."
            except Exception as e:  # noqa: BLE001
                _LOGGER.exception("training failed")
                resp.update(ok=False, message=f"Saved, but training failed: {e}")
        else:
            resp["message"] = (
                f"Saved ({len(templates)} sentences). No speech model for "
                f"'{lang}' — skipped retrain."
            )
        return jsonify(resp)

    return app


def _load_json(path, default):
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return default


def _current_records(cfg, lang: Optional[str] = None) -> list:
    """Live enriched entity records (name/domain/device_class/features/area/
    floor): HA registry in the container, fixture/dev otherwise. Re-fetched at
    each training event so renames/adds/feature changes are picked up. Drives the
    entity-aware gating in training/intent_matcher.

    Entities the user switched off for voice are removed here, so gating, the
    examples and the UI all see the same set."""
    return _overrides(cfg, lang).filter_records(_raw_records(cfg))


def _overrides(cfg, lang: Optional[str] = None):
    return overrides.load(Path(cfg.data), lang or cfg.language)


def _raw_records(cfg) -> list:
    """Entity records straight from Home Assistant / the fixture, before the
    user's voice-targeting overrides are applied."""
    if cfg.hass_token:
        try:
            recs = training.entity_records_from_hass(cfg.hass_api, cfg.hass_token)
            _LOGGER.debug("Loaded %d entity records from Home Assistant", len(recs))
            return recs
        except Exception:  # noqa: BLE001
            _LOGGER.exception("entity fetch failed; falling back to fixture/dev")
    data = _load_json(cfg.entities_file, None)
    if data is None:
        return training.DEV_ENTITY_RECORDS
    if isinstance(data, dict):  # legacy {name: domain} fixture
        return [{"name": n, "domain": d} for n, d in data.items()]
    return data


def _current_slot_lists(cfg, lang: Optional[str] = None) -> Dict[str, List[str]]:
    """Slot value lists for training/display. area + floor come live from the HA
    registries (all of them); language lists (color, brightness_level, ...) come
    from the fixture/file. Re-fetched per training event so registry edits are
    picked up (and change the fingerprint -> retrain). Areas/floors the user
    switched off for voice are removed."""
    return _overrides(cfg, lang).filter_slot_lists(_raw_slot_lists(cfg))


def _raw_slot_lists(cfg) -> Dict[str, List[str]]:
    lists = {k: list(v) for k, v in
             _load_json(cfg.slot_lists_file, training.DEV_SLOT_LISTS).items()}
    if cfg.hass_token:
        try:
            areas, floors = training.areas_floors_from_hass(cfg.hass_api, cfg.hass_token)
            lists["area"], lists["floor"] = areas, floors
            _LOGGER.debug("Loaded %d areas, %d floors from HA", len(areas), len(floors))
        except Exception:  # noqa: BLE001
            _LOGGER.exception("area/floor fetch failed; keeping fixture areas/floors")
    return lists


def _read_enabled(data_dir: Path, lang: str, combos, default_importance) -> list:
    f = data_dir / lang / "enabled.json"
    if f.exists():
        return json.loads(f.read_text())
    return bi.default_enabled(combos, default_importance)


def _usage(lang: str, enabled_set, commands: list):
    """Which commands consume each slot list, for the Devices & Lists view.
    Returns (area_used_by, floor_used_by, {domain: name_used_by})."""
    import re

    def refs(sentence: str):
        return set(re.findall(r"\{([^}]+)\}", sentence))

    import s2p_intents

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


def _custom_commands(data_dir: Path, lang: str) -> list:
    return cc.load(data_dir, lang)


def _fingerprint(templates, list_values, backend) -> str:
    """Hash of everything that determines the grammar -- templates AND the
    name/area/floor/list values. Changes here mean the grammar is stale."""
    blob = json.dumps(
        {"backend": backend, "templates": sorted(templates),
         "lists": {k: sorted(v) for k, v in list_values.items()}},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _model_dir_for(cfg, lang: str) -> Optional[Path]:
    if lang == cfg.language and cfg.model:
        return Path(cfg.model)
    try:
        return models.resolve(None, Path(cfg.models_dir), lang, cfg.backend)
    except Exception:  # noqa: BLE001
        _LOGGER.exception("model resolve failed for '%s'", lang)
        return None


def _ensure_trained(cfg, lang, meta, entities, slot_lists, data_dir: Path,
                    force: bool = False) -> bool:
    """(Re)train `lang` iff the grammar is missing or its input fingerprint
    changed (templates, enabled combos, custom text, or entity/area/floor lists).
    Returns True if it (re)trained. Logged at INFO so retrains are visible."""
    combos = bi.available_combos(ADDON_ROOT, lang, meta)
    enabled = _read_enabled(data_dir, lang, combos, cfg.default_importance)
    commands = _custom_commands(data_dir, lang)
    templates, list_values = training.assemble(
        ADDON_ROOT, lang, enabled, commands, entities, slot_lists,
        extra_sentences=ex.load(data_dir, lang),
        ov=overrides.load(data_dir, lang),
    )
    fp = _fingerprint(templates, list_values, cfg.backend)

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

    reason = ("first boot" if not grammar.exists()
              else "save" if force else "inputs changed (entities/areas/floors/config)")
    _LOGGER.info("Training grammar for '%s' (%s): %d sentences. This may take a moment.",
                 lang, reason, len(templates))
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "enabled.json").exists():
        (d / "enabled.json").write_text(json.dumps(enabled, indent=2))
    training.train(cfg.backend, model_dir, lang, templates, list_values, grammar)
    meta_path.write_text(json.dumps(
        {"fingerprint": fp, "backend": cfg.backend, "n_templates": len(templates)}
    ))
    _LOGGER.info("Trained grammar for '%s' -> %s", lang, grammar)
    return True


def _start_watch(cfg, meta, data_dir: Path) -> None:
    """Background timer: periodically re-check the live registry and retrain any
    set-up language whose inputs changed (entity/area/floor renames, adds, …)."""
    interval = cfg.refresh_interval
    if interval <= 0:
        return

    def loop():
        while True:
            time.sleep(interval)
            try:
                langs = {cfg.language} | {
                    p.name for p in data_dir.iterdir()
                    if p.is_dir() and (p / "grammar.fst").exists()
                }
                # Fetch the registry once per pass, then apply each language's
                # own overrides to it: the filtering is per-language, but the
                # Home Assistant round-trip must not be.
                raw_records = _raw_records(cfg)
                raw_lists = _raw_slot_lists(cfg)
                for lang in sorted(langs):
                    ov = overrides.load(data_dir, lang)
                    if _ensure_trained(
                        cfg, lang, meta, ov.filter_records(raw_records),
                        ov.filter_slot_lists(raw_lists), data_dir
                    ):
                        _LOGGER.info("Auto-retrained '%s' after a registry/config change", lang)
            except Exception:  # noqa: BLE001
                _LOGGER.exception("watch loop iteration failed")

    threading.Thread(target=loop, name="s2p-watch", daemon=True).start()
    _LOGGER.info("Watching for entity/area/floor changes every %ds", interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATA_DIR", "./data"))
    ap.add_argument("--language", default="en")
    ap.add_argument("--backend", default="citrinet")
    ap.add_argument("--model", default=os.environ.get("MODEL_DIR"),
                    help="model dir (dev) or HuggingFace model name; "
                         "if unset, derived from language+backend")
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "/data/models"))
    ap.add_argument("--entities-file", default=os.environ.get("ENTITIES_FILE"))
    ap.add_argument("--slot-lists-file", default=os.environ.get("SLOT_LISTS_FILE"))
    ap.add_argument("--hass-api", default=os.environ.get("HASS_API", "http://supervisor/core/api"))
    ap.add_argument("--hass-token", default=os.environ.get("SUPERVISOR_TOKEN"))
    # We only ship curated sentence files for intents we want supported, so by
    # default enable all of them ("optional" is the most permissive bucket).
    ap.add_argument("--default-importance", default="optional")
    ap.add_argument("--refresh-interval", type=int,
                    default=int(os.environ.get("REFRESH_INTERVAL", "600")),
                    help="seconds between registry-change checks (0 disables)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--wyoming-uri", default=os.environ.get("WYOMING_URI", "tcp://0.0.0.0:10300"))
    ap.add_argument("--intent-uri", default=os.environ.get("INTENT_URI", "tcp://0.0.0.0:10500"))
    ap.add_argument("--max-score", type=float, default=None,
                    help="score gate; if unset, a per-backend default is used "
                         "(citrinet 5.0, coqui 2.0)")
    # Offsets the decoder's bias toward short paths (see models.DEFAULT_TOKEN_BONUS).
    # Unset => the per-backend default (citrinet 2.0, coqui 0.0).
    ap.add_argument("--token-bonus", type=float, default=None,
                    help="word-insertion reward per emitted token (0 = off); "
                         "if unset, a per-backend default is used "
                         "(citrinet 2.0, coqui 0.0)")
    ap.add_argument("--no-wyoming", action="store_true", help="UI only (don't serve Wyoming STT)")
    # Off by default: the add-on ships as speech-to-text only, and Home Assistant
    # handles the transcript with its own conversation agent.
    ap.add_argument("--intent", action="store_true",
                    help="also serve the Wyoming intent service (experimental)")
    ap.add_argument("--debug", action="store_true")
    cfg = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if cfg.debug else logging.INFO)
    # numba (pulled in by librosa) floods DEBUG with JIT traces.
    logging.getLogger("numba").setLevel(logging.INFO)
    app = create_app(cfg)

    # Serve Wyoming STT alongside the web UI in this same process. It reads the
    # grammar.fst the UI/watch-thread writes (hot-reloaded on change).
    if cfg.model and not cfg.no_wyoming:
        grammar = Path(cfg.data) / cfg.language / "grammar.fst"
        wyoming_server.start_background(
            cfg.wyoming_uri, cfg.backend, cfg.model, cfg.language, grammar,
            cfg.max_score, cfg.token_bonus
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
            cfg.intent_uri, cfg.language, Path(cfg.data), ADDON_ROOT,
            get_entities=lambda: _current_records(cfg),
            get_slot_lists=lambda: _current_slot_lists(cfg),
            api_url=cfg.hass_api, token=cfg.hass_token,
            ttl=max(cfg.refresh_interval, 60),
        )
    else:
        _LOGGER.info("Speech-to-text only; intent service not started (--intent enables it)")

    _LOGGER.info("Speech-to-Phrase UI on http://%s:%s (data=%s, model=%s)",
                 cfg.host, cfg.port, cfg.data, cfg.model or "<none>")
    app.run(host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
