# Architecture Overview

This document describes the full pipeline that `lyrics_cleaner.py` runs to replace a target word in a song while keeping it sounding like the original artist.

---

## Pipeline Stages

```
MP3 Input
  │
  ▼
[1] Transcription
    Whisper ASR → word text + timestamps
  │
  ▼
[2] Vocal Separation
    Demucs htdemucs → vocals.wav + no_vocals.wav
  │
  ▼
[3] Voice Sample Extraction
    ~30s of clean vocals (avoids target words) → voice_sample.wav
  │
  ▼
[4] Per-word replacement loop
    │
    ├─ Onset snapping (correct Whisper drift)
    ├─ XTTS-v2 TTS synthesis
    ├─ SEED-VC voice conversion (30 diffusion steps)
    ├─ F0 pitch correction
    ├─ Spectral matching
    └─ Single time-stretch to target duration
  │
  ▼
[5] Seamless blending
    Equal-power crossfades + -24 dB duck zone + onset-aligned overlay
  │
  ▼
[6] Remix
    Modified vocals overlaid onto accompaniment → cleaned MP3
```

---

## Key Functions

| Function | Purpose |
|---|---|
| `transcribe_with_timestamps()` | Whisper ASR with word-level timestamps |
| `separate_vocals()` | Demucs vocal/instrumental separation |
| `extract_voice_sample()` | Collect ~30s of clean singer audio across 4 song regions |
| `clone_voice_word()` | Full TTS → VC → F0 → spectral chain for one word |
| `_correct_f0()` | Pitch-shift replacement to match original word's key |
| `_spectral_match()` | EQ/timbre alignment to original recording |
| `_snap_to_onset()` | Snap Whisper timestamps to true energy onsets |
| `_apply_crossfade()` | Equal-power S-curve fade in + fade out |
| `_apply_eq_power_fade()` | Per-channel equal-power fade using numpy |
| `build_clean_audio()` | Orchestrates blending of all replacements |

---

## Model Dependencies

| Model | Library | Purpose |
|---|---|---|
| Whisper `base` | `openai-whisper` | Speech-to-text transcription |
| htdemucs | `demucs` | Vocal/instrumental source separation |
| XTTS-v2 | `TTS` (Coqui) | Voice-cloned speech synthesis |
| SEED-VC v2 | `seed-vc` | Zero-shot voice conversion |

> `transformers==4.44.2` is pinned because XTTS-v2 requires `BeamSearchScorer` which was removed in transformers 5.x.
