# Music Audio Cleaner

Automatically removes or replaces specific words in songs while keeping the original music intact.

Two strategies are available — pick the one that sounds best for your track:

| Strategy | Flag | How it works | Quality |
|---|---|---|---|
| **Melody-clean** | `--keep-melody-remove-target` | Mutes the target word in the vocal track, fades the surrounding vocals smoothly, and remixes with the full instrumental | Clean, instant, music sounds natural |
| **Word replacement** | *(default full pipeline)* | Synthesizes a replacement word in the singer's voice using XTTS-v2 + SEED-VC and splices it in | Variable — see note below |

> **Voice cloning note:** The full voice-clone + re-sing pipeline (`--resing`) was built and works technically, but the output quality is, to put it plainly, *"it's horrible"*. The synthesized voice bears little resemblance to the original singer. The melody-clean approach (mute + remix) is currently the recommended path for the best-sounding result.

---

## How it works

### Melody-clean (recommended)
1. **Transcribe** — Whisper ASR finds the target word and its exact timestamps.
2. **Separate** — Demucs splits the track into `vocals.wav` and `no_vocals.wav`.
3. **Mute** — The target word's region is silenced in the vocal track. The word before fades out over 1 second heading into the gap; the word after fades in over 1 second coming out of it, so the edit blends musically.
4. **Remix** — Modified vocals are overlaid onto the original instrumental. The gap is filled by the accompaniment playing through naturally.

### Word replacement (full pipeline)
1. **Transcribe** — Whisper ASR with word-level timestamps.
2. **Separate** — Demucs htdemucs → `vocals.wav` + `no_vocals.wav`.
3. **Voice sample** — 30 s of clean singer audio extracted and cached in `cloned_voice/`.
4. **Phonetic selection** — The replacement word is scored on syllable count, stress pattern, and stressed vowel (0–3). Auto-selects the best child-friendly phonetic match if no suggestion provided.
5. **Synthesize** — XTTS-v2 generates the replacement word in a cloned voice.
6. **Convert** — SEED-VC (30 diffusion steps) converts the synthesis to match the singer's timbre.
7. **Blend** — F0 pitch correction, spectral/EQ matching, onset-snapped splice, harmonic bed blending, equal-power crossfades, up to 5 auto-correction rounds.

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) — install separately or let `imageio-ffmpeg` handle it
- A CUDA-capable GPU is strongly recommended (CPU works but is very slow for the full pipeline)

### Install

```bash
git clone <repo-url>
cd music_audio_cleaner
pip install -r requirements.txt
```

`requirements.txt` includes: `numpy`, `librosa`, `scipy`, `soundfile`, `pydub`,
`imageio-ffmpeg`, `openai-whisper`, `demucs`, `pronouncing`, `transformers==4.44.2`,
`TTS` (Coqui), `seed-vc`.

> `transformers==4.44.2` is pinned — do not upgrade. XTTS-v2 requires `BeamSearchScorer` which was removed in transformers 5.x.

---

## Library layout

Organize songs as `library/Artist/Album/Song/<song>.mp3`. Each song folder can hold its own `targetwords.txt` and an optional `official_lyrics.txt` reference:

```
library/
  Nelson Freitas/
    Simple Girl/
      Nelson Freitas - Simple Girl.mp3
      targetwords.txt
      official_lyrics.txt
```

---

## targetwords.txt format

```csv
Target,Replace
damn,
booty,fun
come,
```

- **Target** — word or phrase as sung (case-insensitive).
- **Replace** — optional suggestion. Leave empty for auto-selection.

Auto-selection picks the best child-friendly phonetic match from the CMU Pronouncing Dictionary (no network calls). If a suggestion is provided it is validated; if it scores below 2/3 phonetically, auto-selection runs instead.

---

## Usage

```bash
# Show full manual page (all switches explained)
python lyrics_cleaner.py --manual

# Melody-clean (fast, recommended)
python lyrics_cleaner.py "library/Artist/Album/Song/song.mp3" --keep-melody-remove-target

# Full pipeline (word replacement with voice cloning)
python lyrics_cleaner.py "library/Artist/Album/Song/song.mp3"

# Re-verify an existing clean file (no models loaded)
python lyrics_cleaner.py "song.mp3" --verify

# Re-sing entire song in cloned voice (experimental — quality varies widely)
python lyrics_cleaner.py "song.mp3" --resing

# Use a more accurate Whisper model
python lyrics_cleaner.py "song.mp3" targetwords.txt small
```

---

## Output files

All outputs are written into the song folder:

| File | Description |
|---|---|
| `*_melody_clean.mp3` | Melody-clean: target word muted, vocals faded, full remix |
| `*_clean.mp3` | Voice-replacement clean version |
| `*_original.txt` | Full original lyrics transcript |
| `*_cleaned.txt` | Lyrics with target words substituted |
| `*_vocals_clean.wav` | Cleaned isolated vocals sidecar (used by `--verify`) |
| `demucs_output/*/vocals.wav` | Separated vocals (cached) |
| `demucs_output/*/no_vocals.wav` | Separated instrumental (cached) |
| `cloned_voice/voice_reference.wav` | 30 s voice clone reference (cached) |

---

## Caching / fast paths

| Switch | Needs | Skips |
|---|---|---|
| `--keep-melody-remove-target` | `_timestamps.json` + demucs cache | Whisper re-transcription |
| `--resing` | `_cleaned.txt` + `_timestamps.json` + demucs + voice clone | Full replacement pipeline |
| `--verify` | `*_clean.mp3` + `*_timestamps.json` | Everything — pure audio math |

---

## First run downloads

| Model | Size | Location |
|---|---|---|
| XTTS-v2 (Coqui TTS) | ~1.8 GB | `checkpoints/` |
| SEED-VC + CampPlus | ~500 MB | `checkpoints/` |
| Whisper `base` | ~140 MB | Hugging Face cache |
| Demucs htdemucs | ~80 MB | Torch hub cache |

---

## License

Copyright (C) 2024 Music Audio Cleaner Contributors

This program is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see <https://www.gnu.org/licenses/>.
