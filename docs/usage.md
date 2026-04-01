# Usage Guide

## Requirements

- Python 3.10+
- A CUDA GPU is recommended (CPU works but is slow)
- ffmpeg (auto-installed via `imageio-ffmpeg` if not present)

All Python dependencies install automatically on first run. Or install manually:

```bash
pip install -r requirements.txt
```

> **Note:** `transformers==4.44.2` is pinned. Do not upgrade it — XTTS-v2 requires `BeamSearchScorer` which was removed in transformers 5.x.

---

## Step 1 — Define target words

Edit `targetwords.txt`. It is a CSV with two columns:

```
Target,Replace
damn,darn
booty,beauty
sexy,zesty
```

- **Target** — the word (or phrase) as it appears in the lyrics
- **Replace** — what to substitute in the audio

Multi-word phrases work: `"oh my god,oh my gosh"`.

---

## Step 2 — Run

```bash
python lyrics_cleaner.py "path/to/song.mp3"
```

Optional arguments:

```bash
python lyrics_cleaner.py "song.mp3" targetwords.txt base
#                                    ^               ^
#                                    words CSV       Whisper model size
```

Whisper model sizes (accuracy vs speed): `tiny` · `base` · `small` · `medium` · `large`  
Default is `base`. Use `small` or `medium` for better timestamp accuracy.

---

## Output files

All outputs are written next to the input MP3:

| File | Description |
|---|---|
| `song_clean.mp3` | Cleaned audio with replaced words |
| `song_original.txt` | Full original lyrics transcript |
| `song_cleaned.txt` | Lyrics with target words substituted |

---

## First run

On first run the tool downloads model checkpoints into `checkpoints/`:

| Model | Size |
|---|---|
| XTTS-v2 (Coqui TTS) | ~1.8 GB |
| SEED-VC + CampPlus | ~500 MB |

Subsequent runs reuse the cache. Vocal separation results are also cached in `demucs_output/` — delete that folder to force re-separation.

---

## Tuning tips

| Goal | Change |
|---|---|
| Better timestamps | Use `small` or `medium` Whisper model |
| Faster processing | Lower `diffusion_steps` in `clone_voice_word()` (min ~15) |
| Higher VC quality | Raise `diffusion_steps` (e.g. 50) |
| Replacement too short/long | The word duration is inferred from Whisper; use a larger Whisper model for more accurate timing |
