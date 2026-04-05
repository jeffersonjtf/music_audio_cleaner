# Usage Guide

## Requirements

- Python 3.10+
- A CUDA GPU is recommended (CPU works but is very slow for the full pipeline)
- ffmpeg (auto-installed via `imageio-ffmpeg` if not present)

```bash
pip install -r requirements.txt
```

> **Note:** `transformers==4.44.2` is pinned. Do not upgrade it — XTTS-v2 requires `BeamSearchScorer` which was removed in transformers 5.x.

---

## Library layout

Organize your songs as `library/Artist/Album/Song/<song>.mp3`. Each song folder can have its own `targetwords.txt` and an optional `official_lyrics.txt`:

```
library/
  Nelson Freitas/
    Simple Girl/
      Nelson Freitas - Simple Girl.mp3
      targetwords.txt
      official_lyrics.txt        ← optional, improves re-sing text alignment
```

---

## Step 1 — Define target words

Create `targetwords.txt` in the song folder (or the project root as a fallback). It is a CSV with two columns:

```
Target,Replace
damn,
booty,fun
come,
sexy,
```

- **Target** — the word (or phrase) as sung; case-insensitive.
- **Replace** — optional suggested replacement. Leave empty for auto-selection.

**Auto-selection** picks the best child-friendly phonetic match from the CMU Pronouncing Dictionary — no network calls. It scores on three axes (syllable count, stress pattern, stressed vowel) and picks the highest-scoring word not in the adult content blocklist.

If you provide a suggestion, it is validated: if it scores below 2/3 phonetically or is in the blocklist, auto-selection runs instead.

Multi-word phrases work: `oh my god,oh my gosh`.

---

## Step 2 — Choose your approach

### Option A: Melody-clean (recommended)

Mutes the target word and remixes with the original instrumental. Fast, no AI synthesis,
sounds natural. The word before the gap fades out over 1 second; the word after fades in
over 1 second.

```bash
python lyrics_cleaner.py "library/Artist/Album/Song/song.mp3" --keep-melody-remove-target
```

Output: `song_melody_clean.mp3`

**Fast path:** if `_timestamps.json` and demucs outputs are already cached, this completes in ~5 seconds. Whisper is not re-run.

---

### Option B: Word replacement (full pipeline)

Synthesizes a replacement word using XTTS-v2 voice cloning + SEED-VC voice conversion.
More complex, results vary by song and replacement word.

```bash
python lyrics_cleaner.py "library/Artist/Album/Song/song.mp3"
```

Outputs: `song_clean.mp3` (voice-replaced) + `song_melody_clean.mp3` (always produced alongside).

The full pipeline also always produces a melody-clean version, so you get both outputs in one run.

---

### Option C: Re-sing (experimental — not recommended)

Synthesizes the entire song line-by-line in the singer's cloned voice.

> **Quality warning:** The re-sing pipeline produces results that, in honest assessment, are *"horrible"*. The cloned voice does not resemble the original artist. This mode is kept for future research but is not recommended for practical use.

```bash
python lyrics_cleaner.py "library/Artist/Album/Song/song.mp3" --resing
```

**Fast path:** if `_cleaned.txt`, `_timestamps.json`, demucs outputs, and `cloned_voice/voice_reference.wav` all exist, skips the full replacement pipeline and runs re-sing only.

---

## Other switches

```bash
# Re-run 9-metric verification against an existing clean file (no models loaded)
python lyrics_cleaner.py "song.mp3" --verify

# Specify a different targetwords.txt
python lyrics_cleaner.py "song.mp3" path/to/targetwords.txt

# Use a larger Whisper model for better timestamp accuracy
python lyrics_cleaner.py "song.mp3" targetwords.txt small
#                                                    ^
#                                        tiny | base | small | medium | large
```

---

## Output files

| File | Description |
|---|---|
| `*_melody_clean.mp3` | Melody-clean: word muted, 1s fades, full remix with instrumental |
| `*_clean.mp3` | Word-replacement version (full pipeline only) |
| `*_original.txt` | Full original lyrics transcript |
| `*_cleaned.txt` | Lyrics with target words substituted |
| `*_vocals_clean.wav` | Cleaned isolated vocals sidecar (used by `--verify`) |
| `*_timestamps.json` | Cached Whisper word timestamps |
| `demucs_output/*/vocals.wav` | Separated vocals (cached) |
| `demucs_output/*/no_vocals.wav` | Separated instrumental (cached) |
| `cloned_voice/voice_reference.wav` | 30 s voice clone reference (cached) |
| `resing/*_resing_mix.mp3` | Re-sing full mix (experimental) |
| `logs/terminal_*.log` | Timestamped session log |

---

## First-run downloads

| Model | Size |
|---|---|
| XTTS-v2 (Coqui TTS) | ~1.8 GB |
| SEED-VC + CampPlus | ~500 MB |
| Whisper `base` | ~140 MB |
| Demucs htdemucs | ~80 MB |

Cached in `checkpoints/` and the Hugging Face / Torch hub caches. Subsequent runs skip downloads.

---

## Phonetic report

When loading `targetwords.txt`, the tool prints a compatibility report for each word:

```
  Target: "come"
    Replacement: "hum"  [auto-selected]
    Phonetic compatibility (3/3):
      Syllables:      1 == 1  ✓
      Stress pattern: 1 == 1  ✓
      Stressed vowel: AH == AH  ✓
      → Excellent phonetic fit
```

Scores: 3/3 = excellent · 2/3 = acceptable · 1/3 or 0/3 = rejected, auto-selection runs.

---

## Verification report

After each word replacement the pipeline prints a 9-metric quality check:

```
    Verification: PASS (0 warnings)
      RMS ratio:          0.98  ✓
      Spectral centroid:  1.04  ✓
      Spectral rolloff:   0.97  ✓
      Splice-in:          1.01  ✓
      Splice-out:         0.99  ✓
      ZCR ratio:          1.2   ✓
      MFCC distance:      0.09  ✓
      F0 difference:      1.1 st ✓
      Chroma distance:    0.12  ✓
```

WARN triggers for 1–2 out-of-range metrics; FAIL for 3+. The auto-correction loop runs up
to 5 times and keeps the best result.

---

## Tuning tips

| Goal | Change |
|---|---|
| Better timestamps | Use `small` or `medium` Whisper model |
| Faster word replacement | Lower `diffusion_steps` in `clone_voice_word()` (min ~15, default 30) |
| Higher VC quality | Raise `diffusion_steps` (e.g. 50) — significantly slower |
| Longer fades in melody-clean | Adjust `FADE_OUT_MS` / `FADE_IN_MS` in `build_melody_clean()` (default 1000 ms) |
