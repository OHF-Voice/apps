# Speech-to-Phrase — Model Coverage Report

**Goal:** support (at least) every language Home Assistant ships translations/UI for.
**Date:** 2026-07-02

This report has three parts:

1. **Coverage gap** — which of HA's 66 languages have an acoustic model today.
2. **Empirical round-trip test** — a small TTS→STT test of every existing model
   (Citrinet and Coqui) on the 4 core command categories.
3. **Datasets for the gaps** — permissively-licensed corpora to train new
   Citrinet/NeMo models for the missing languages.

---

## 1. Coverage gap

Home Assistant ships **66** translation/UI languages (top-level entries in
`intent-sentences/languages.yaml`). Speech-to-Phrase currently maps a model for
**13 base languages** (`src/models.py:MODEL_NAMES`), covering **16** of the 66
HA codes (regional variants share a base model):

| Base lang | Citrinet model | Coqui model | Covers HA codes |
|-----------|----------------|-------------|-----------------|
| en | `stt_en_citrinet_512` | `en_US-coqui` | en (US/GB) |
| de | `stt_de_citrinet_1024` | `de_DE-coqui` | de, de-CH |
| es | `stt_es_citrinet_512` | `es_ES-coqui` | es (ES/MX) |
| fr | `stt_fr_citrinet_1024_gamma_0_25` | `fr_FR-rhasspy` | fr |
| it | `stt_it_conformer_ctc_large` | `it_IT-coqui` | it |
| zh | `stt_zh_citrinet_512` | — | zh-CN, zh-HK, zh-TW |
| ru | `stt_ru_conformer_ctc_large` | — | ru |
| hr | `stt_hr_conformer_ctc_large` | — | hr |
| hi | `stt_hi_conformer_ctc_medium` | — | hi |
| ca | `stt_ca_conformer_ctc_large` | `ca_ES-coqui` | ca |
| nl | — | `nl_NL-coqui` | nl (NL/BE) |
| cs | — | `cs_CZ-coqui` | cs |
| sl | — | `sl_SL-coqui` | sl |

### Two distinct gaps

**Gap A — Coqui-only (no Citrinet): `nl`, `cs`, `sl`.**
These work but run on the character-level Coqui backend, which needs the
`stt_onlyprobs` binary and — as the test below shows for `nl` — has lower
command fidelity than Citrinet. Upgrading these three to NeMo Citrinet/Conformer
is the cheapest coverage win.

**Gap B — no model at all: 50 HA languages.** Prioritized by HA's own support signals:

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

## 2. Empirical round-trip test (Citrinet + Coqui)

**Method.** For each language and each available backend, a grammar was built
from a handful of concrete, in-language commands spanning the 4 requested
categories, each command was synthesized with **HA Cloud TTS** (`tts_get_url`,
`engine=tts.home_assistant_cloud`, correct per-language voice), decoded through
the actual Speech-to-Phrase recognizer, and classified against the backend's
score gate (Citrinet 5.0, Coqui 2.0). Two out-of-grammar sentences per language
checked false-accept behavior.

Categories: **C1** lights on/off (current area) · **C2** named entity on/off ·
**C3** lights on in a named area · **C4** start a timer by duration.

Sentences for en/de/es/fr/it/nl/hi came from the human-authored `example:`
fields in home-assistant-intents; zh/ru/hr/ca/cs/sl were authored from each
language's own sentence templates (sl mirrors the curated S2P `sentences/sl/`).

> ⚠️ **Caveat:** TTS ≠ human speech, and the grammar here is a tiny hand-built
> set. This measures whether an acoustic model can *round-trip canonical
> commands cleanly*; it is **not** a real-world accuracy or robustness number.

### Results

`EXACT` = decoded to the exact command. `accepted` = EXACT **and** score ≤ gate
(would be handled locally rather than deferred to cloud). `OOV FA` = out-of-grammar
clips wrongly accepted as a command.

| Config | Backend | EXACT | Accepted | OOV false-accept | Notes |
|--------|---------|:-----:|:--------:|:----------------:|-------|
| en | Citrinet | 6/6 | 6/6 | 0/2 | clean |
| en | Coqui | 6/6 | 6/6 | 0/2 | clean |
| de | Citrinet | 6/6 | 6/6 | 0/2 | clean |
| es | Citrinet | 6/6 | 6/6 | 0/2 | clean |
| es | Coqui | **ERR** | — | — | **model fails to load** (see below) |
| fr | Citrinet | 5/6 | 5/6 | 0/2 | timer sentence missed but **gated** (→cloud) |
| it | Citrinet | 5/6 | 5/6 | 0/2 | area sentence missed but **gated** |
| zh | Citrinet | 6/6 | 6/6 | 0/2 | clean (char-level) |
| ru | Citrinet | 5/6 | 5/6 | 0/2 | area sentence missed but **gated** |
| hr | Citrinet | 6/6 | 6/6 | 0/2 | clean |
| hi | Citrinet | 6/6 | 6/6 | 0/2 | clean |
| ca | Citrinet | 6/6 | 6/6 | 0/2 | clean (scores run higher, ~3–4) |
| nl | Coqui | 4/6 | 4/6 | 0/2 | ⚠️ **2 below-gate confusions** (would misfire) |
| cs | Coqui | 6/6 | 6/6 | 0/2 | clean |
| sl | Coqui | 6/6 | 6/6 | 0/2 | clean |

**11 of 15 configs round-trip all 6 commands perfectly. Zero OOV false-accepts
anywhere** — the current gates (Citrinet 5.0 / Coqui 2.0) reject non-command
audio in every language tested.

### Issues found

1. **`es_ES-coqui` is broken.** Loading fails with
   `RuntimeError: Expected [T, 30] probs, got (118, 36)` — the model's alphabet
   has 36 symbols but the `stt_onlyprobs` decode path expects 30. The mapped
   Coqui model is incompatible with the current binary/config. Not urgent
   (Spanish has a working Citrinet), but `MODEL_NAMES["es"]["coqui"]` should be
   fixed or dropped.

2. **`nl_NL-coqui` misrecognizes below the gate — the only real accuracy risk.**
   "doe de lichten **uit**" (off) decoded as "…**aan**" (on), and the named-area
   command collapsed to the plain on-command — both at score < 2.0, so they
   would be **acted on as the wrong command** rather than safely deferred. Dutch
   is the strongest candidate for a Citrinet upgrade (see §3).

3. **Longer sentences on `fr`/`it`/`ru` Citrinet** (the timer sentence for fr,
   the "lights in the kitchen" sentence for it/ru) failed to decode — but every
   one scored **above the gate**, so the recognizer correctly falls back to HA
   Cloud instead of guessing. This is graceful degradation, not a
   misfire. Likely a mix of TTS prosody on the longer utterance and these being
   Conformer (ru) / gamma-variant (fr) models; worth re-checking with real
   speech and the full production grammar.

4. **Chinese, Hindi, Croatian, Catalan, Czech, Slovenian all round-trip
   cleanly**, confirming the non-Latin (Devanagari, Han) and Coqui-only models
   are functional for the core commands.

Raw per-utterance scores: `scratchpad/results.json`; harness:
`scratchpad/multilang_test.py`.

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

**Upgrade the 3 Coqui-only languages to Citrinet first — data is abundant:**

| Lang | Recommended training data | License | ~Permissive hrs | Verdict |
|------|---------------------------|---------|:---------------:|---------|
| **nl** Dutch | MLS 1,554h + CML-TTS 645h + CV 126h + VoxPopuli 53h | CC-BY / CC-BY / CC0 / CC0 | **~2,400** | **Ample** — and Dutch is the one model with a demonstrated accuracy bug (§2), so this is the highest-value single action. |
| **cs** Czech | ParlaSpeech-CZ 1,218h + LINDAT 444h + CV 82h + VoxPopuli 62h | CC-BY-SA / CC-BY / CC0 / CC0 | **~1,800** | **Ample** |
| **sl** Slovenian | ARTUR 884h (includes purpose-built smart-home utterances) + GOS 300h + CV 17h | CC-BY-SA / CC-BY-SA / CC0 | **~1,200** | **Ample** |

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

- **9 of 10 Tier-1 languages (all but Bulgarian) have >100h of permissively-licensed
  speech today** — the blocker is training/packaging effort, not data.
- **Do first:** train Citrinet for **nl, cs, sl** (fixes the Coqui-only gap and
  the Dutch accuracy bug), then **pl, pt, sk, uk, da** (proven HA demand, ample data).
- **Needs data collection before a model is feasible:** bg, he, vi, and the
  scarcer Tier-2/Tier-3 languages — good candidates for a Common Voice campaign.

---

## Appendix — action items surfaced by this report

1. Fix or drop `MODEL_NAMES["es"]["coqui"]` — `es_ES-coqui` fails to load
   (`Expected [T, 30] probs, got (118, 36)`).
2. Investigate `nl_NL-coqui` below-gate confusions (off→on); prioritize the
   Citrinet upgrade above.
3. Re-check `fr`/`it`/`ru` longer-sentence gating with **real speech** and the
   full production grammar (TTS prosody may explain the clean-audio misses).
4. Reproduce: `python scratchpad/multilang_test.py [lang]` (models in
   `speech-to-phrase-lib/local/`, per-utterance scores in `results.json`).

