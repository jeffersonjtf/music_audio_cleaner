# Architecture Overview

This document describes the two pipelines in `lyrics_cleaner.py` and their shared infrastructure.

---

## Pipeline A — Melody-Clean (`--keep-melody-remove-target`)

The simplest and best-sounding approach. No AI synthesis involved.

```
MP3 Input
  │
  ▼
[1] Transcription (Whisper)
    word text + timestamps
  │
  ▼
[2] Vocal Separation (Demucs htdemucs)
    vocals.wav + no_vocals.wav
  │
  ▼
[3] Onset snapping
    Whisper timestamps snapped ±200ms to true energy onsets
  │
  ▼
[4] Mute + musical fades
    ├─ Fade-out on vocals leading into gap  (1000 ms equal-power)
    ├─ Silence inserted for muted zone      (word duration + 80ms pad each side)
    └─ Fade-in on vocals coming out of gap  (1000 ms equal-power)
  │
  ▼
[5] Remix
    Modified vocals overlaid onto no_vocals.wav
    → *_melody_clean.mp3
```

**Fast path:** if `_timestamps.json` + demucs cache exist, skips Whisper entirely (~5 s total).

---

## Pipeline B — Word Replacement (full pipeline, default)

Replaces individual words with voice-cloned synthesis. More complex, results vary.

> **Quality note:** The voice-clone re-sing (`--resing`) pipeline was fully implemented but the output quality is poor — the cloned voice bears little resemblance to the original artist. The word-replacement pipeline is more targeted and can produce acceptable results on short words with good phonetic matches.

```
MP3 Input
  │
  ▼
[1] Transcription (Whisper)
    word text + timestamps
  │
  ▼
[2] Vocal Separation (Demucs htdemucs)
    vocals.wav + no_vocals.wav
  │
  ▼
[3] Voice clone reference
    extract_voice_sample() → 30s clean vocals from 4 song regions
    cached at cloned_voice/voice_reference.wav
  │
  ▼
[4] Phonetic replacement selection
    CMU Pronouncing Dictionary scoring (0–3):
    syllable count + stress pattern + stressed vowel
    child-friendly filter (~70-word blocklist)
  │
  ▼
[5] Per-word replacement loop
    │
    ├─ Onset snapping (correct Whisper ±50–200ms drift)
    ├─ XTTS-v2 TTS synthesis (voice-cloned text)
    ├─ SEED-VC voice conversion (30 diffusion steps)
    ├─ F0 pitch correction (match original word's key, ±12st clamp)
    ├─ Spectral / EQ matching (per-bin gain, ±20dB clamp)
    ├─ Singability scoring (open vowel + resonant coda bonus)
    ├─ Vowel-nucleus time stretch (stretch only nucleus, keep consonants ≤80ms)
    ├─ Harmonic bed blending (accompaniment at -24dB into replacement)
    └─ Auto-correction loop (up to 5 rounds, best-of-N)
  │
  ▼
[6] Seamless blending
    Equal-power S-curve crossfades (30ms) + -24dB duck zone + onset-aligned overlay
  │
  ▼
[7] 9-metric self-verification
    RMS, spectral centroid, rolloff, splice continuity ×2,
    ZCR, MFCC cosine distance, F0 semitones, chroma distance
    → PASS / WARN / FAIL verdict
  │
  ▼
[8] Remix
    Modified vocals overlaid onto no_vocals.wav
    → *_clean.mp3  +  *_vocals_clean.wav sidecar
  │
  ▼
[9] Melody-clean also produced (Step B always runs Step A too)
    → *_melody_clean.mp3
```

---

## Pipeline C — Re-sing (`--resing`, experimental)

Synthesizes the entire song line-by-line in the singer's cloned voice.

> **Status: Not recommended.** The output quality is, in the project team's words, *"it's horrible"*. The cloned voice does not convincingly match the original artist. This pipeline remains in the codebase for future improvement but should not be used for production output.

```
MP3 Input
  │
  (all Pipeline B caches must exist)
  │
  ▼
[1] Load _cleaned.txt (primary text source — clean words already baked in)
    Fallback: official_lyrics.txt → Whisper transcript
  │
  ▼
[2] Align text to Whisper line groups (difflib SequenceMatcher, ratio ≥ 0.6)
  │
  ▼
[3] Per-line synthesis loop
    │
    ├─ XTTS-v2 TTS (cloned voice)
    ├─ SEED-VC voice conversion
    ├─ F0 contour transfer (spline-fitted from original vocals)
    └─ Spectral matching
  │
  ▼
[4] Remix
    Synthesized vocal lines overlaid onto no_vocals.wav
    → resing/*_resing_vocals.wav + resing/*_resing_mix.mp3
```

---

## Shared Infrastructure

### Onset snapping (`_snap_to_onset`)
Searches ±250ms around each Whisper timestamp for the nearest energy onset in the separated vocals. Clamps to ±200ms. Eliminates the systematic 50–200ms Whisper timestamp drift that causes audible boundary artifacts.

### Equal-power crossfades (`_apply_eq_power_fade`, `_apply_crossfade`)
Uses sin(t)/cos(t) curves so `sin²(t) + cos²(t) = 1` — total power stays constant through the crossover. Avoids the level-dip artefact of linear fades.

### Logging (`_setup_logging`, `_PrintToLogger`)
All output (print statements) is redirected through Python's `logging` module. Every line gets a `[HH:MM:SS]` timestamp on both the terminal and the log file (`logs/terminal_DATE_TIME.log`). The file handler is UTF-8; the console handler wraps `sys.__stdout__.buffer` in a UTF-8 `TextIOWrapper` to handle Unicode symbols on Windows cp1252 terminals.

### Song folder layout
```
library/
  Artist/
    Album/
      Song/
        song.mp3
        targetwords.txt          ← per-song word list (optional)
        official_lyrics.txt      ← reference lyrics (optional)
        song_original.txt        ← Whisper transcript (generated)
        song_cleaned.txt         ← clean lyrics (generated)
        song_timestamps.json     ← word timestamps cache (generated)
        song_clean.mp3           ← word-replacement output (generated)
        song_melody_clean.mp3    ← melody-clean output (generated)
        song_vocals_clean.wav    ← clean vocals sidecar (generated)
        demucs_output/
          song/
            vocals.wav
            no_vocals.wav
        cloned_voice/
          voice_reference.wav    ← 30s voice clone reference (generated)
        resing/
          song_resing_vocals.wav
          song_resing_mix.mp3
```

---

## Key Functions

| Function | Purpose |
|---|---|
| `transcribe_with_timestamps()` | Whisper ASR with word-level timestamps |
| `separate_vocals()` | Demucs vocal/instrumental separation |
| `extract_voice_sample()` | Collect ~30s of clean singer audio across 4 regions |
| `load_target_words()` | Parse targetwords.txt + phonetic validation + auto-selection |
| `find_target_words()` | Match target phrases in Whisper word list |
| `clone_voice_word()` | Full TTS → VC → F0 → spectral chain for one word |
| `build_clean_audio()` | Orchestrates word replacement, correction loop, blending |
| `build_melody_clean()` | Mute target words + musical fades + remix (no synthesis) |
| `resing_song()` | Full-song re-synthesis in cloned voice (experimental) |
| `_correct_f0()` | Pitch-shift replacement to match original word's key |
| `_spectral_match()` | EQ/timbre alignment to original recording |
| `_snap_to_onset()` | Snap Whisper timestamps to true energy onsets |
| `_apply_crossfade()` | Equal-power S-curve fade in + fade out |
| `_apply_eq_power_fade()` | Per-channel equal-power fade using numpy |
| `_blend_harmonic_bed()` | Overlay accompaniment at low gain into replacement |
| `_auto_correct()` | 5-round correction loop for RMS/F0/spectral metrics |
| `_verify_replacement()` | 9-metric quality check per replaced word |
| `_singability_score()` | Score word for open vowels + resonant coda |
| `_vowel_stretch()` | Time-stretch only the vowel nucleus of a word |
| `_group_words_into_lines()` | Group Whisper words into sung lines by gap |
| `_align_official_lyrics()` | difflib alignment of reference text to Whisper timing |
| `verify_only()` | Re-run verification against existing clean file |

---

## Model Dependencies

| Model | Library | Purpose |
|---|---|---|
| Whisper `base` | `openai-whisper` | Speech-to-text transcription |
| htdemucs | `demucs` | Vocal/instrumental source separation |
| XTTS-v2 | `TTS` (Coqui) | Voice-cloned speech synthesis |
| SEED-VC v2 | `seed-vc` | Zero-shot voice conversion (30 diffusion steps) |

> `transformers==4.44.2` is pinned because XTTS-v2 requires `BeamSearchScorer` which was removed in transformers 5.x.
