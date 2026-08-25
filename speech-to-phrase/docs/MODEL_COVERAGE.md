# Speech-to-Phrase — Model Coverage Report

**Goal:** support (at least) every language Home Assistant ships translations/UI for.
**Date:** 2026-08-25

A language needs *two* things to be usable: an **acoustic model**, and a set of
**Speech-to-Phrase sentence templates** in `home-assistant-intents`. Either
alone recognizes nothing, and the two are gated by completely different work —
model training versus sentence authoring — so this report counts them
separately.

Three parts:

1. **Coverage gap** — which of HA's 66 languages Speech-to-Phrase can serve.
2. **Empirical round-trip test** — TTS→STT over every command each language
   ships, per model.
3. **Datasets for the gaps** — permissively-licensed corpora to train new
   Citrinet/NeMo models for the missing languages.

---

## 1. Coverage gap

Home Assistant ships **66** translation/UI languages (top-level entries in
`intent-sentences/languages.yaml`). Speech-to-Phrase maps a model for **13 base
languages** (`src/models.py:MODEL_NAMES`) but can only *serve* the **8** that
also have sentence templates, covering **9** of the 66 HA codes (regional
variants share a base model):

| Base lang | Citrinet model | Coqui model | Templates? | Covers HA codes |
|-----------|----------------|-------------|:----------:|-----------------|
| en | `stt_en_citrinet_512` | `en_US-coqui` | ✅ | en (US/GB) |
| de | `stt_de_citrinet_1024` | `de_DE-coqui` | ✅ | de, de-CH |
| es | `stt_es_citrinet_512` | — | ✅ | es (ES/MX) |
| fr | `stt_fr_conformer_ctc_large` | `fr_FR-rhasspy` | ✅ | fr |
| it | `stt_it_conformer_ctc_large` | `it_IT-coqui` | ✅ | it |
| nl | `stt_nl_citrinet_256` | `nl_NL-coqui` | ✅ | nl (NL/BE) |
| ca | `stt_ca_conformer_ctc_large` | `ca_ES-coqui` | ✅ | ca |
| cs | — | `cs_CZ-coqui` | ✅ | cs |
| zh | `stt_zh_citrinet_512` | — | ❌ | zh-CN, zh-HK, zh-TW |
| ru | `stt_ru_conformer_ctc_large` | — | ❌ | ru |
| hr | `stt_hr_conformer_ctc_large` | — | ❌ | hr |
| hi | `stt_hi_conformer_ctc_medium` | — | ❌ | hi |
| sl | — | `sl_SL-coqui` | ❌ | sl |

Two entries in that table are constrained rather than merely chosen, and should
not be changed without re-testing:

- **Spanish is Citrinet-only.** `es_ES-coqui` does not load: its alphabet has 36
  symbols where the `stt_onlyprobs` decode path expects 30.
- **French uses the Conformer**, not `stt_fr_citrinet_1024_gamma_0_25`. Of the
  four French models in the dataset it is the one that reliably separates
  `verrouille` from `déverrouille`; the Citrinet decodes "unlock the front door"
  as "lock the front door" confidently enough that the score gate lets it
  through, which is the one error direction that must not happen on a lock.

### Three distinct gaps

The 66 HA codes split **9 served + 7 model-but-no-sentences + 50 no model**.

**Gap A — a model but no sentences: `zh`, `ru`, `hr`, `hi`, `sl` (7 HA codes).**
The cheapest coverage win by far, and *not* a modelling problem: the acoustic
side is done and validated, and what's missing is `speech_to_phrase`-tagged
blocks in `intent-sentences/sentences/<lang>/` — roughly 50–60 per language,
authored the way `de`/`ca`/`cs` are (PRs #4130–#4136 are the pattern). Slovenian
is closest to ready: a draft of `sentences/sl/` exists and round-tripped 20/20,
but is not upstream and needs rebasing onto the current block shape.

**Gap B — Coqui-only (no Citrinet): `cs`.**
Czech works, but on the character-level Coqui backend, which needs the
`stt_onlyprobs` binary and runs on its own score scale.

**Gap C — no model at all: 50 HA languages.** Prioritized by HA's own support signals:

- **Tier 1 — HA already offers Whisper STT (proven demand), 7 langs:**
  `bg` Bulgarian, `da` Danish, `pl` Polish, `pt` Portuguese, `pt-BR`
  Portuguese (Brazil), `sk` Slovak, `uk` Ukrainian.
- **Tier 2 — HA offers Piper TTS but no Whisper, 15 langs:**
  `ar`, `el`, `fa`, `fi`, `hu`, `is`, `ka`, `lb`, `lv`, `ne`, `ro`, `sr`,
  `sv`, `tr`, `vi`.
- **Tier 3 — translations only, no HA STT/TTS backend yet, 28 langs:**
  `af, bn, cy, et, eu, ga, gl, gu, he, hy, id, ja, kn, ko, kw, lt, ml, mn, mr,
  ms, nb, pa, sr-Latn, sw, ta, te, th, ur`.

Note `sk` (Slovak) is a special case: it is closely related to `cs` (Czech),
which already has a Coqui model — a Slovak model is high-value and low-risk.

---

## 2. Empirical round-trip test

**Method.** `tools/lang_check.py` runs the production path per language —
package templates → `training.assemble` → FST grammar → Recognizer, against
**every** `speech_to_phrase`-tagged example that language ships, each spoken by
its own HA Cloud TTS voice. The grammar is built over the whole entity fixture
set from `intent-sentences`, so it is a realistic install (dozens of devices per
domain), not a handful of hand-picked phrases.

What it scores is whether Home Assistant would end up running **the command
that was spoken**, not whether the transcript matched character for character:
the decoder legitimately picks a different in-grammar realization of the same
command (Dutch drops an article, French normalizes a mis-agreeing example), so
each decode is re-matched through hassil and compared by (intent, slot
combination). `accepted` additionally requires the score to be at or under the
gate — above it the utterance is handed to the cloud rather than guessed at.

> ⚠️ **Caveat:** TTS is not human speech, and one clip per block is a small
> sample. This catches what matters when adding a language — templates that do
> not transpile, a grammar that will not compile, commands the model cannot hear
> at all — and is **not** a real-world accuracy or robustness number.

### Results (2026-08-25, home-assistant-intents 2026.8.25)

| Lang | Backend / model | Same command | Accepted | Exact string |
|------|-----------------|:------------:|:--------:|:------------:|
| de | citrinet `stt_de_citrinet_1024` | **59/59** | 59/59 | 59/59 |
| ca | citrinet `stt_ca_conformer_ctc_large` | **56/56** | 56/56 | 46/56 |
| cs | coqui `cs_CZ-coqui` | **59/59** | 59/59 | 56/59 |
| es | citrinet `stt_es_citrinet_512` | **55/55** | 55/55 | 50/55 |
| fr | citrinet `stt_fr_conformer_ctc_large` | **56/56** | 56/56 | 35/56 |
| it | citrinet `stt_it_conformer_ctc_large` | **50/50** | 50/50 | 44/50 |
| nl | citrinet `stt_nl_citrinet_256` | **53/53** | 53/53 | 53/53 |
| en | citrinet `stt_en_citrinet_512` | **20/20** | 20/20 | 19/20 |

**Every supported language resolves every command it ships, all within the score
gate.** The gap between "same command" and "exact string" is concentrated in
languages whose `example:` fields interpolate slots without fixing agreement —
French is the extreme (`règle la luminosité de le Lampe A`, `ajoute 5 minute au
minuteur`), and the decoder returns the grammatical in-grammar form instead.
That is the examples needing work upstream, not the recognizer.

English is the least-covered language *by this test*: only 20 of its 54 tagged
blocks carry an `example:`, so 34 go unmeasured. Every other language annotates
all of them.

Reproduce: `python3 tools/lang_check.py --language <lang> --model <dir>
--intents-repo <intent-sentences>` — models under `speech-to-phrase-lib/local/`,
`--json-out` for per-utterance scores, `HA_TOKEN` for the TTS, and clips cache
under `tests/wav/.tts_cache` so a re-run makes no TTS calls.

### What this test cannot tell us

- **The five Gap A languages are unmeasured**, because there is nothing to
  build a grammar from. Their acoustic models do round-trip hand-built
  command sets — including `zh` (Han) and `hi` (Devanagari) — which is the basis
  for calling Gap A a sentence-authoring job rather than a modelling one, but
  that is a weaker result than the table above and does not substitute for it.
- **Nothing here is measured on human speech.** Real-world accuracy, noise
  robustness and false-accept rates against non-command audio are
  `tools/audio_test.py`'s job (TTS → room impulse response + noise sweep, plus
  out-of-grammar false-accept checks), not this one's.

---

## 3. Datasets to train the missing Citrinet models

Speech-to-Phrase's existing models are NVIDIA **NeMo CTC** nets (Citrinet /
Conformer, 16 kHz). To add Citrinet where it's missing, we need transcribed
speech under a **permissive** license. Clean licenses for a training pipeline:
**CC0**, **CC-BY-4.0**, **Apache-2.0**, **MIT**. **CC-BY-SA** is usable to train
on (the copyleft attaches to any redistributed *corpus*, generally not to the
trained model). **Anything CC-BY-NC / -ND / custom-academic is off the table.**

Common Voice figures are validated hours from **CV 26.0 (2026-06)**.

### The workhorse corpora

| Corpus | License | Coverage for our gaps |
|--------|---------|-----------------------|
| **Mozilla Common Voice 26.0** | ✅ CC0 | The universal baseline — the only clean-license source covering **all** target langs. Read speech, matches command-and-control well. |
| **Multilingual LibriSpeech / CML-TTS** ([SLR94](https://openslr.org/94/), [SLR146](https://openslr.org/146/)) | ✅ CC-BY-4.0 | Large, clean read speech, but **Germanic/Romance only**: nl 1,554h, pt 161h, pl 104h. |
| **VoxPopuli** (transcribed) | ✅ CC0 | EU-Parliament ASR, tens of hours each for pl, ro, hu, cs, nl, sk, fi, sl. Formal register. |
| **National parliamentary corpora** | mixed (CC-BY / CC-BY-SA) | The big win for Slavic/Nordic: **RixVox** sv 5,493h, **SloPalSpeech** sk 2,806h, **ParlaSpeech-CZ** cs 1,218h, **ARTUR** sl 884h. |
| **FLEURS**, **CoVoST 2** | ✅ CC-BY-4.0 / CC0 | ~10h/lang — **eval / few-shot only**, not training corpora. |

### Priority recommendations

**Czech is the only Coqui-only language (Gap B), and it round-trips 59/59, so a
Citrinet for it is a nice-to-have rather than a fix:**

| Lang | Recommended training data | License | ~Permissive hrs | Verdict |
|------|---------------------------|---------|:---------------:|---------|
| **cs** Czech | ParlaSpeech-CZ 1,218h + LINDAT 444h + CV 82h + VoxPopuli 62h | CC-BY-SA / CC-BY / CC0 / CC0 | **~1,800** | **Ample** — would drop the `stt_onlyprobs` binary requirement and put Czech on the same score scale as everything else |

The Gap A languages need no data at all: `sl` has `sl_SL-coqui` and the other
four have Citrinet/Conformer models. Sentences are the blocker, not corpora.

**Tier 1 gap languages (HA already ships Whisper — proven demand):**

| Lang | Recommended data | License | ~Hrs | Verdict |
|------|------------------|---------|:----:|---------|
| **pl** Polish | CV 177 + VoxPopuli 111 + MLS 108 | CC0 / CC0 / CC-BY | ~400 | **Ample** |
| **pt / pt-BR** Portuguese | CV 187 + MLS 161 + CML-TTS 68 | CC0 / CC-BY | ~415 | **Ample** |
| **sk** Slovak | **SloPalSpeech 2,806** + CV 57 + VoxPopuli 35 | CC-BY / CC0 | ~2,900 | **Ample** (also: reuse the cs pipeline — closely related) |
| **uk** Ukrainian | **speech-uk/openstt ~1,789** + CV 103 | CC-BY / CC0 | ~1,800 | **Ample** (review YODAS2/broadcast provenance) |
| **da** Danish | **NST Danish 390** + dictation 54 + CV 13 | CC0 | ~440 | **Ample** (CV alone is only 13h — NST is what makes it viable) |
| **bg** Bulgarian | CV 17 only (BG-PARLAMA / BulPhonC are NC) | CC0 | ~17 | **Scarce** ⚠️ — the one Tier-1 lang without enough permissive data |

**Tier 2 (Piper TTS, no Whisper) — where permissive data already suffices:**
`ja` (CV 372h, ample), `th` (CV 173h, ample), `hu` (CV+VoxPopuli ~200h, ample),
`tr` (~140h, marginal), `ro` (VoxPopuli+CV ~114h, moderate), `ar` (CV 92h +
Apache/MIT sets ~115h, moderate). **Data-scarce (defer / need collection):**
`fi` (~43h), `ko` (~55h), `el` (~20h), `id` (~34h), `he` (~5h ⚠️), `vi` (~8h ⚠️).

### Not suitable (do **not** put in a permissive pipeline)

- **Non-commercial (NC):** SOFES (sl), BulPhonC / BG-PARLAMA (bg), CORAA-ASR
  (pt-BR), Multilingual TEDx ([SLR100](https://openslr.org/100/), pt/el/ar),
  SASPEECH ([SLR134](https://openslr.org/134/), he), VIVOS (vi), most OpenSLR
  Korean sets (SLR58/97/113).
- **No-derivatives (ND):** CORAA, mTEDx.
- **Custom/restricted:** CGN (nl, paid), FT Speech (da, no redistribution),
  CoRal (da, OpenRAIL-D use restrictions).

### Bottom line

- **Do first, and it isn't a model:** write Speech-to-Phrase sentence blocks for
  `zh`, `ru`, `hr`, `hi`, `sl`. Five languages already have a validated acoustic
  model and cannot be used for want of ~50 templates each — no training, no
  data licensing, no GPU. Nothing else on this page is close in value per hour.
- **9 of 10 Tier-1 languages (all but Bulgarian) have >100h of permissively-licensed
  speech today** — for those, the blocker is training/packaging effort, not data.
- **Then train Citrinet for:** `pl, pt, sk, uk, da` (proven HA demand, ample
  data), and `cs` if the Coqui-only backend becomes a maintenance problem.
- **Needs data collection before a model is feasible:** bg, he, vi, and the
  scarcer Tier-2/Tier-3 languages — good candidates for a Common Voice campaign.

---

## Appendix — open action items

1. Author `speech_to_phrase` blocks for `zh`, `ru`, `hr`, `hi`, `sl` in
   `intent-sentences` (Gap A). The `sentences/sl/` draft needs rebasing onto the
   current block shape and upstreaming; the other four start from scratch.
2. Add `example:` fields to the 34 English tagged blocks that have none, so
   English is measured as thoroughly as every other language.
3. Re-check `fr`, `it` and `ca` with **human speech** — their gap between
   "same command" and "exact string" is the widest, and TTS prosody is the
   obvious confound.
4. Consider a Citrinet for `cs` to retire the `stt_onlyprobs` dependency
   (§3), and for the Tier-1 languages in priority order.

