# Music Audio Cleaner

Automatically censors or replaces specific words in songs while preserving the original singer's voice.

## How it works

1. **Transcribe** — Uses [OpenAI Whisper](https://github.com/openai/whisper) to transcribe the song's lyrics with timestamps.
2. **Separate** — Uses [Demucs](https://github.com/facebookresearch/demucs) to split the track into vocals and instrumentals.
3. **Voice sample** — Extracts a clean voice sample of the singer from silence gaps in the vocals track.
4. **Synthesize** — Generates replacement words using [Coqui XTTS-v2](https://github.com/coqui-ai/TTS) voice cloning.
5. **Convert** — Runs the synthesized audio through [SEED-VC](https://github.com/Plachta/Seed-VC) to match the singer's voice characteristics.
6. **Blend** — Replaces the target word segments in the original audio with crossfading and volume matching.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) (installed automatically via `imageio-ffmpeg` if not found)
- A CUDA-capable GPU is recommended for faster processing

Python dependencies are installed automatically on first run, or manually:

```bash
pip install -r requirements.txt
```

## Usage

1. Define the words you want to replace in `targetwords.txt` (CSV format):

```csv
Target,Replace
damn,darn
booty,beauty
```

2. Run the cleaner with your MP3 file:

```bash
python lyrics_cleaner.py "path/to/song.mp3"
```

The cleaned audio is saved alongside the original as `song_clean.mp3`.

## Notes

- On first run, model checkpoints (~2 GB) are downloaded automatically to the `checkpoints/` directory.
- `transformers==4.44.2` is pinned for compatibility between XTTS-v2 and SEED-VC.
