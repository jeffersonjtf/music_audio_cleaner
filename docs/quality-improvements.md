# Quality Improvements: Making Replacements Sound Seamless

This document explains the eight improvements made to address the unnatural sound of replaced words, what each one fixes, and why it works.

---

## Problem Summary

The original implementation produced replacements that sounded out-of-place due to several compounding issues:

- Wrong pitch — the replacement word sang at the wrong note
- Wrong timbre — the replacement had a different tonal "color" than the recording
- Acoustic holes — the muted zone removed all room tone, making the gap audible
- Edit click/sweep — abrupt level changes at splice boundaries were audible
- Misaligned cuts — Whisper timestamps drifted up to 200ms from the true word onset
- Phase artifacts — two sequential time-stretch passes compounded distortion

---

## Improvement 1 — F0 Pitch Correction

**Function:** `_correct_f0()`  
**Stage:** After SEED-VC voice conversion

### What it does
Extracts the mean fundamental frequency (F0) from both the original sung word and the voice-converted replacement using `librosa.pyin()`. Computes the interval in semitones and applies a pitch shift to the replacement so its pitch center matches the original.

### Why this matters
XTTS-v2 synthesizes speech at its own natural prosody — typically flat and conversational. When the singer holds a note on a word, the replacement arrives at a completely different pitch. This is the single most audible artifact. Correcting even the mean F0 brings the replacement into the same harmonic space as the surrounding melody.

### Bounds
Correction is clamped to ±12 semitones and skipped if the difference is less than 0.5 semitones to avoid over-processing.

---

## Improvement 2 — Spectral / EQ Matching

**Function:** `_spectral_match()`  
**Stage:** After F0 correction

### What it does
Computes the mean magnitude spectrum (via STFT) of both the original word segment and the replacement. Derives per-frequency-bin gain factors that would make the replacement's spectrum match the original's. Applies those gains after smoothing (20-bin uniform filter) and clamping (0.1× – 10×, i.e. ±20 dB max).

### Why this matters
The replacement is synthesized in a neutral acoustic environment. The original recording has mic coloration, room reflections, compression, and mixing EQ baked in. Without spectral matching, the replacement sounds like it came from a different room — because it did. This step gives it the same tonal signature as the rest of the track.

---

## Improvement 3 — Onset Snapping

**Function:** `_snap_to_onset()`  
**Stage:** Before replacement generation in `build_clean_audio()`

### What it does
For each Whisper word timestamp, searches a ±250ms window in the separated vocals track for the nearest energy onset using `librosa.onset.onset_detect()` with `backtrack=True`. Replaces the Whisper timestamp with the snapped position, clamped to ±200ms of the original.

### Why this matters
Whisper's word timestamps can be off by 50–200ms. If the mute window starts too early, it clips the preceding word. If it starts too late, the first phoneme of the target word leaks through before the replacement begins. Both are immediately obvious to the ear. Snapping to the true acoustic onset eliminates these boundary artifacts.

---

## Improvement 4 — Remove Pre-VC Time Stretch (Single Stretch Pass)

**Stage:** `clone_voice_word()` restructuring

### What it does
Previously the TTS output was time-stretched to match the target duration *before* voice conversion, then stretched again *after* voice conversion to correct any remaining length mismatch. Now the TTS is passed to SEED-VC without any pre-stretch. Only one time-stretch is applied — after voice conversion.

### Why this matters
`librosa.effects.time_stretch()` uses a phase vocoder. Each pass introduces phase smearing and warbling. Two sequential passes on the same audio compound these artifacts noticeably. A single post-VC stretch produces a far cleaner result.

---

## Improvement 5 — SEED-VC Diffusion Steps 10 → 30

**Parameter:** `diffusion_steps` in `clone_voice_word()`

### What it does
Increases the number of diffusion sampling steps from 10 to 30 during SEED-VC voice conversion.

### Why this matters
Diffusion models iteratively refine their output. At 10 steps the voice conversion converges to a rough approximation with unstable pitch (warbling) and blurred formants. At 30 steps the speaker characteristics are significantly more stable and detailed. The trade-off is processing time — roughly 3× slower — but the quality improvement is substantial.

---

## Improvement 6 — Loosen Trim Threshold (top_db 30 → 50)

**Parameter:** `top_db` in `librosa.effects.trim()` calls in `clone_voice_word()`

### What it does
Reduces the silence threshold for trimming from -30 dB to -50 dB (relative to peak).

### Why this matters
At -30 dB, `librosa.effects.trim()` was aggressively cutting the natural reverb decay at the end of each synthesized word, leaving it sounding dry and disconnected from the acoustic space. At -50 dB, the tail is preserved. This subtle breathiness after the word helps it feel embedded in the mix rather than pasted on top.

---

## Improvement 7 — Duck Zone -24 dB Instead of Silence

**Stage:** `build_clean_audio()` mute zone construction

### What it does
The zone where the original word is suppressed is now set to -24 dB gain (from `original_vocals`) rather than complete silence (`AudioSegment.silent()`).

### Why this matters
Complete silence in a mute zone is acoustically unnatural — the room tone, reverb, and background noise floor all vanish for that instant. Even if the replacement fills the gap, the crossfade edges expose the discontinuity. At -24 dB the original vocal is inaudible under the replacement but the room's acoustic character is maintained throughout, making the edit invisible to casual listening.

---

## Improvement 8 — Equal-Power S-Curve Crossfades

**Functions:** `_apply_crossfade()`, `_apply_eq_power_fade()`, `_eq_power_curve()`  
**Crossfade window:** 100 ms → 30 ms

### What it does
Replaces pydub's built-in `fade_in()` / `fade_out()` (which use an exponential curve and are applied to the whole clip) with a custom numpy-based equal-power (sin/cos) S-curve fade applied per channel. The crossfade window is also reduced from 100 ms to 30 ms.

### Why this matters
Standard linear or exponential fades cause an audible level dip at the crossover point because the combined power of the two signals drops at the midpoint. An **equal-power** crossfade uses `sin(t)` for the fade-in and `cos(t)` for the fade-out so that `sin²(t) + cos²(t) = 1` — the total power stays constant throughout the transition. This is the standard used in professional DAWs. The shorter 30 ms window also means less smearing of the consonant attacks at the word boundaries.

---

## Summary

| Change | Where in code | Impact |
|---|---|---|
| F0 pitch correction | `_correct_f0()` called in `clone_voice_word()` after SEED-VC | Replacement word sings at the correct note instead of the wrong key |
| Spectral / EQ matching | `_spectral_match()` called in `clone_voice_word()` after F0 fix | Timbre and tonal color match the original recording's mic/room character |
| Onset snapping | `_snap_to_onset()` called in `build_clean_audio()` per word | Mute window aligns to the true word attack, eliminating Whisper's ±200ms drift |
| Remove pre-VC stretch | `clone_voice_word()` restructured — no stretch before SEED-VC | Eliminates double phase-vocoder pass and its compounding artifacts |
| `diffusion_steps` 10 → 30 | `clone_voice_word()`, `wrapper.convert_voice()` call | Much more stable and detailed voice conversion output |
| `top_db` 30 → 50 | `clone_voice_word()`, both `librosa.effects.trim()` calls | Natural reverb tail preserved; replacement no longer sounds dry |
| Duck zone -24 dB instead of silence | `build_clean_audio()` mute zone construction | Room tone and reverb continuity maintained; acoustic "hole" eliminated |
| Equal-power S-curve crossfades, 30 ms | `_apply_crossfade()`, `_apply_eq_power_fade()` throughout `build_clean_audio()` | No level-sweep artefact at edit boundaries; constant perceived loudness |
| Self-verification (9 metrics) | `_verify_replacement()` + `_print_verification()` called after each overlay in `build_clean_audio()` | Automatically detects whether the replacement blended; prints PASS / WARN / FAIL with per-metric detail |

---

## Self-Verification

After each word is replaced, the pipeline runs `_verify_replacement()` which takes 2 seconds of audio before and after the replaced section and compares 9 metrics against the replacement:

| # | Metric | Ideal | Threshold | What it catches |
|---|---|---|---|---|
| 1 | RMS energy ratio | ≈ 1.0 | 0.25 – 4.0 | Replacement too quiet or too loud vs surrounding audio |
| 2 | Spectral centroid drift | < 60% | 60% | Tonal brightness mismatch — replacement sounds thin or muffled |
| 3 | Spectral rolloff drift | < 50% | 50% | High-frequency brightness mismatch — sounds dull or tinny |
| 4 | Splice-in continuity | ≈ 1.0 | 0.1 – 10.0 | Abrupt level jump at the entry cut point |
| 5 | Splice-out continuity | ≈ 1.0 | 0.1 – 10.0 | Abrupt level jump at the exit cut point |
| 6 | ZCR ratio at splice | < 5.0× | 5.0× | Click or pop at the splice edge (waveform discontinuity) |
| 7 | MFCC cosine distance | < 0.15 | 0.15 | Voice/timbre identity — sounds like a different singer |
| 8 | F0 pitch difference | < 3.0 st | 3 semitones | Replacement is singing the wrong note / outside the song's key |
| 9 | Chroma cosine distance | < 0.20 | 0.20 | Replacement sits in a different harmonic/musical key space |

**Verdict levels:**
- `PASS` — 0 warnings: replacement is well-integrated
- `WARN` — 1–2 warnings: minor issues, may be acceptable on playback
- `FAIL` — 3+ warnings: replacement likely audible, consider re-running

---

## Before / After Comparison

| Aspect | Before | After |
|---|---|---|
| Pitch | TTS natural prosody (wrong key) | Matched to original word's F0 center |
| Timbre/EQ | Neutral synthesis | Matched to recording's spectral envelope |
| Edit boundaries | Exponential fade, 100 ms | Equal-power S-curve, 30 ms |
| Mute zone | Complete silence | -24 dB (room tone preserved) |
| Timestamp accuracy | Whisper ±200 ms | Snapped to energy onset |
| VC quality | 10 diffusion steps | 30 diffusion steps |
| Reverb tail | Trimmed at -30 dB | Preserved at -50 dB |
| Stretch passes | 2 (pre-VC + post-VC) | 1 (post-VC only) |
