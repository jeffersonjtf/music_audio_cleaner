import sys
import subprocess
import csv
import re
import tempfile
from pathlib import Path


def ensure_dependencies():
    """Install all required packages if missing. Run once on a fresh machine."""
    deps = [
        # (import_name, pip_name)
        ("numpy", "numpy"),
        ("soundfile", "soundfile"),
        ("pydub", "pydub"),
        ("librosa", "librosa"),
        ("whisper", "openai-whisper"),
        ("demucs", "demucs"),
        ("scipy", "scipy"),
    ]

    missing = []
    for import_name, pip_name in deps:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

    # TTS (Coqui) needs special handling — heavy deps, pin transformers
    try:
        __import__("TTS")
    except ImportError:
        print("Installing TTS (Coqui XTTS-v2)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "TTS"])
        # TTS requires transformers <5 (BeamSearchScorer removed in 5.x)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.44.2"])
        # TTS tokenizer needs spacy
        try:
            __import__("spacy")
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])

    # seed-vc needs special handling — it pulls transformers 5.x, so pin after
    try:
        from seed_vc.seed_vc_wrapper import SeedVCWrapper  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        print("Installing seed-vc (SEED-VC voice conversion)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seed-vc"])
        # Re-pin transformers for TTS compatibility (seed-vc works fine with 4.44.2)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.44.2"])

    # ffmpeg is needed by pydub/whisper/demucs for audio I/O
    import shutil
    if shutil.which("ffmpeg") is None:
        print("Installing ffmpeg via pip (imageio-ffmpeg)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
        # imageio-ffmpeg bundles a static ffmpeg binary — find it and add to PATH
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = str(Path(ffmpeg_path).parent)
        import os
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"  ffmpeg available at: {ffmpeg_path}")


ensure_dependencies()

import numpy as np
import soundfile as sf
from pydub import AudioSegment


# Lazy-loaded TTS model (loaded once, reused for all replacements)
_tts_model = None

# Lazy-loaded SEED-VC v2 wrapper (loaded once, reused for all conversions)
_seedvc_wrapper = None


def get_tts_model():
    """Load XTTS-v2 model once and cache it."""
    global _tts_model
    if _tts_model is None:
        import torch
        from TTS.api import TTS

        # torch 2.9+ defaults weights_only=True which breaks TTS model loading
        # (TTS pickles many config classes). Temporarily allow unsafe loading.
        _original_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_load(*args, **kwargs)
        torch.load = _patched_load

        print("Loading XTTS-v2 voice cloning model (first run downloads ~1.8GB)...")
        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
        print("  Model loaded.")

        # Restore original torch.load
        torch.load = _original_load
    return _tts_model


def get_seedvc_wrapper():
    """Load SEED-VC wrapper once and cache it."""
    global _seedvc_wrapper
    if _seedvc_wrapper is None:
        import torch
        from seed_vc.seed_vc_wrapper import SeedVCWrapper

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading SEED-VC voice conversion model (first run downloads models)...")
        _seedvc_wrapper = SeedVCWrapper(device=device)
        print("  SEED-VC model loaded.")
    return _seedvc_wrapper


def load_target_words(filepath):
    """Load target/replace word pairs from CSV file."""
    pairs = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row["Target"].strip().lower()
            replace = row["Replace"].strip()
            if replace:  # skip empty replacements
                pairs[target] = replace
    return pairs


def transcribe_with_timestamps(mp3_path, model_size="base"):
    """Transcribe MP3 with word-level timestamps using Whisper."""
    import whisper
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    print(f"Transcribing: {mp3_path}")
    result = model.transcribe(str(mp3_path), word_timestamps=True)

    words = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"],
            })
    return result["text"], words


def separate_vocals(mp3_path):
    """Use demucs Python API to separate vocals from accompaniment.
    Saves output with soundfile to avoid torchaudio.save() / torchcodec issues."""
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import AudioFile

    print("Separating vocals from instruments with demucs...")
    mp3_path = Path(mp3_path)
    out_dir = mp3_path.parent / "demucs_output" / mp3_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already separated (skip if so)
    vocals_path = out_dir / "vocals.wav"
    no_vocals_path = out_dir / "no_vocals.wav"
    if vocals_path.exists() and no_vocals_path.exists():
        print("  Using cached separation output.")
        return vocals_path, no_vocals_path

    model = get_model("htdemucs")
    model.eval()

    print("  Loading audio...")
    wav = AudioFile(str(mp3_path)).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()

    print("  Running separation (this takes a minute)...")
    with torch.no_grad():
        sources = apply_model(model, wav[None], progress=True)[0]

    sources = sources * ref.std() + ref.mean()

    source_names = model.sources
    vocals_idx = source_names.index("vocals")

    vocals_tensor = sources[vocals_idx].cpu().numpy().T
    no_vocals_tensor = sum(
        sources[i] for i in range(len(source_names)) if i != vocals_idx
    ).cpu().numpy().T

    sf.write(str(vocals_path), vocals_tensor, model.samplerate)
    sf.write(str(no_vocals_path), no_vocals_tensor, model.samplerate)

    print(f"  Vocals:        {vocals_path}")
    print(f"  Accompaniment: {no_vocals_path}")
    return vocals_path, no_vocals_path


def extract_voice_sample(vocals_path, words, replacements, target_duration=30.0):
    """Extract multiple clean vocal segments (avoiding target words) for voice cloning.
    Combines 3-4 segments from different parts of the song, totaling ~30s.
    More variety = better clone (singer at different pitches, intensities)."""

    # Build list of time ranges to avoid (target word positions + padding)
    avoid_ranges = []
    for r in replacements:
        avoid_ranges.append((r["start"] - 0.5, r["end"] + 0.5))

    # Read full vocals to get duration and sample rate
    info = sf.info(str(vocals_path))
    sr = info.samplerate
    total_duration = info.duration

    def overlaps_avoid(start, end):
        for a_start, a_end in avoid_ranges:
            if start < a_end and end > a_start:
                return True
        return False

    # Build list of all clean stretches (gaps between avoid ranges)
    # First, merge overlapping avoid ranges
    sorted_avoids = sorted(avoid_ranges, key=lambda x: x[0])
    merged = []
    for s, e in sorted_avoids:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Find clean windows between avoid ranges
    clean_windows = []
    prev_end = 0.0
    for s, e in merged:
        if s > prev_end + 2.0:  # at least 2s of clean audio
            clean_windows.append((prev_end, s))
        prev_end = e
    if total_duration > prev_end + 2.0:
        clean_windows.append((prev_end, total_duration))

    if not clean_windows:
        # Fallback: use first 10s of vocals
        clean_windows = [(0.0, min(10.0, total_duration))]

    # Distribute target_duration across segments from different parts of the song
    # Pick up to 4 windows, spread across the song for variety
    num_segments = min(4, len(clean_windows))
    # Space them evenly across available windows
    if len(clean_windows) > num_segments:
        step = len(clean_windows) / num_segments
        selected = [clean_windows[int(i * step)] for i in range(num_segments)]
    else:
        selected = clean_windows

    # Calculate how much to take from each segment
    per_segment = target_duration / len(selected)

    segments_audio = []
    collected = 0.0

    print(f"  Collecting ~{target_duration:.0f}s of voice reference from {len(selected)} segments:")

    for win_start, win_end in selected:
        available = win_end - win_start
        take = min(per_segment, available)
        # Center the take within the window
        seg_start = win_start + (available - take) / 2
        seg_end = seg_start + take

        start_frame = int(seg_start * sr)
        stop_frame = int(seg_end * sr)
        y, _ = sf.read(str(vocals_path), start=start_frame, stop=stop_frame)
        segments_audio.append(y)
        collected += take
        print(f"    {seg_start:.1f}s - {seg_end:.1f}s ({take:.1f}s)")

        if collected >= target_duration:
            break

    # Concatenate all segments with tiny silence gaps between them
    silence_gap = np.zeros((int(0.1 * sr), segments_audio[0].shape[1] if segments_audio[0].ndim > 1 else 1))
    if segments_audio[0].ndim == 1:
        silence_gap = silence_gap.flatten()

    combined = []
    for i, seg in enumerate(segments_audio):
        combined.append(seg)
        if i < len(segments_audio) - 1:
            combined.append(silence_gap)

    combined_audio = np.concatenate(combined, axis=0)

    sample_path = Path(vocals_path).parent / "voice_sample.wav"
    sf.write(str(sample_path), combined_audio, sr)
    print(f"  Total voice sample: {collected:.1f}s -> {sample_path}")
    return sample_path




def clone_voice_word(text, voice_sample_path, vocals_path, start_s, end_s, duration_ms):
    """Generate a replacement word that sounds like the original singer.
    Two-stage pipeline:
    1. XTTS-v2 generates the replacement word as speech (correct phonemes)
    2. SEED-VC v2 converts that speech to match the singer's voice (preserves melody)
    3. Time-stretched to exact duration"""
    import librosa

    tts_model = get_tts_model()

    # Step 1: Generate TTS of the replacement word (voice timbre doesn't matter much
    # since SEED-VC will convert it, but using singer's sample helps with prosody)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tts_raw_path = tmp.name

    tts_model.tts_to_file(
        text=text,
        speaker_wav=str(voice_sample_path),
        language="en",
        file_path=tts_raw_path,
    )

    # Step 2: Time-stretch TTS output to match original word duration
    y, sr = librosa.load(tts_raw_path, sr=None)
    Path(tts_raw_path).unlink(missing_ok=True)

    y, _ = librosa.effects.trim(y, top_db=30)

    if len(y) == 0:
        return AudioSegment.silent(duration=duration_ms)

    target_samples = int(duration_ms * sr / 1000)
    if target_samples > 0 and len(y) > 0:
        stretch_factor = len(y) / target_samples
        stretch_factor = max(0.5, min(3.0, stretch_factor))
        if abs(stretch_factor - 1.0) > 0.05:
            y = librosa.effects.time_stretch(y, rate=stretch_factor)

    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    # Save time-stretched TTS for SEED-VC input
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tts_stretched_path = tmp.name
    sf.write(tts_stretched_path, y, sr)

    # Step 3: Voice conversion with SEED-VC
    # Converts the TTS speech to sound like the original singer
    # f0_condition=True preserves melody/pitch from source
    print(f"    Converting to singer's voice with SEED-VC...")
    wrapper = get_seedvc_wrapper()

    # convert_voice is a generator — collect the last yielded full audio
    vc_sr = None
    vc_audio = None
    for mp3_bytes, full_audio in wrapper.convert_voice(
        source=tts_stretched_path,
        target=str(voice_sample_path),
        diffusion_steps=10,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        f0_condition=True,  # preserve melody from source
        auto_f0_adjust=True,
        pitch_shift=0,
        stream_output=True,
    ):
        if full_audio is not None:
            vc_sr, vc_audio = full_audio

    Path(tts_stretched_path).unlink(missing_ok=True)

    if vc_audio is None:
        return AudioSegment.silent(duration=duration_ms)

    # Trim silence from VC output
    vc_audio = vc_audio.astype(np.float32)
    vc_trimmed, _ = librosa.effects.trim(vc_audio, top_db=30)
    if len(vc_trimmed) == 0:
        vc_trimmed = vc_audio

    # Final time-stretch to exact target duration
    target_samples_vc = int(duration_ms * vc_sr / 1000)
    if target_samples_vc > 0 and len(vc_trimmed) > 0:
        stretch_factor = len(vc_trimmed) / target_samples_vc
        stretch_factor = max(0.5, min(3.0, stretch_factor))
        if abs(stretch_factor - 1.0) > 0.05:
            vc_trimmed = librosa.effects.time_stretch(vc_trimmed, rate=stretch_factor)

    if len(vc_trimmed) < target_samples_vc:
        vc_trimmed = np.pad(vc_trimmed, (0, target_samples_vc - len(vc_trimmed)))
    else:
        vc_trimmed = vc_trimmed[:target_samples_vc]

    # Convert to AudioSegment
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, vc_trimmed, vc_sr)
    result = AudioSegment.from_wav(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    return result


def _apply_crossfade(audio, fade_ms=30):
    """Apply smooth fade in/out to avoid clicks."""
    if len(audio) < fade_ms * 2:
        fade_ms = max(5, len(audio) // 4)
    return audio.fade_in(fade_ms).fade_out(fade_ms)


def find_target_words(words, target_pairs):
    """Find words in transcript that match target words."""
    replacements = []
    max_phrase_len = max((len(t.split()) for t in target_pairs), default=1)

    i = 0
    while i < len(words):
        matched = False
        for phrase_len in range(min(max_phrase_len, len(words) - i), 0, -1):
            phrase_words = words[i:i + phrase_len]
            phrase_text = " ".join(w["word"] for w in phrase_words).lower()
            phrase_clean = re.sub(r"[^\w\s]", "", phrase_text).strip()

            if phrase_clean in target_pairs:
                replacements.append({
                    "original": phrase_text,
                    "replacement": target_pairs[phrase_clean],
                    "start": phrase_words[0]["start"],
                    "end": phrase_words[-1]["end"],
                })
                i += phrase_len
                matched = True
                break

        if not matched:
            word_clean = re.sub(r"[^\w]", "", words[i]["word"]).lower()
            if word_clean in target_pairs:
                replacements.append({
                    "original": words[i]["word"],
                    "replacement": target_pairs[word_clean],
                    "start": words[i]["start"],
                    "end": words[i]["end"],
                })
            i += 1

    return replacements


def build_clean_audio(replacements, vocals_path, no_vocals_path, voice_sample_path):
    """Build cleaned MP3 using TTS + SEED-VC voice conversion + seamless blending."""
    print(f"\nLoading separated tracks...")
    vocals = AudioSegment.from_wav(str(vocals_path))
    original_vocals = AudioSegment.from_wav(str(vocals_path))  # pristine copy
    accompaniment = AudioSegment.from_wav(str(no_vocals_path))

    if not replacements:
        print("No target words found — nothing to replace.")
        min_len = min(len(vocals), len(accompaniment))
        return accompaniment[:min_len].overlay(vocals[:min_len])

    print(f"Replacing {len(replacements)} word(s) with voice-cloned audio...\n")

    fade_ms = 100  # longer crossfade for smoother transitions

    for r in replacements:
        start_ms = int(r["start"] * 1000)
        end_ms = int(r["end"] * 1000)

        # Pad for crossfade zones
        pad = 120  # ms
        mute_start = max(0, start_ms - pad)
        mute_end = min(len(vocals), end_ms + pad)
        mute_duration = mute_end - mute_start

        print(f"  [{r['start']:.2f}s - {r['end']:.2f}s] "
              f"\"{r['original']}\" -> \"{r['replacement']}\"")

        # Clone the singer's voice + apply their melody + match dynamics
        tts = clone_voice_word(
            r["replacement"], voice_sample_path, vocals_path,
            r["start"], r["end"], mute_duration
        )

        # --- Volume matching ---
        ref_segment = original_vocals[start_ms:end_ms]
        original_loudness = ref_segment.dBFS
        if tts.dBFS > -50 and original_loudness > -50:
            tts = tts.apply_gain(original_loudness - tts.dBFS + 2)

        # --- Smooth crossfade on replacement ---
        tts = _apply_crossfade(tts, fade_ms=fade_ms)

        # --- Crossfaded suppression of original word ---
        before = vocals[:mute_start]

        fade_out_end = min(mute_start + fade_ms, mute_end)
        fade_out_zone = vocals[mute_start:fade_out_end].fade_out(fade_ms)

        duck_start = fade_out_end
        duck_end = max(mute_end - fade_ms, duck_start)
        ducked_zone = AudioSegment.silent(duration=duck_end - duck_start,
                                          frame_rate=vocals.frame_rate)

        fade_in_start = duck_end
        fade_in_zone = vocals[fade_in_start:mute_end].fade_in(fade_ms)

        after = vocals[mute_end:]

        vocals = before + fade_out_zone + ducked_zone + fade_in_zone + after
        vocals = vocals.overlay(tts, position=mute_start)
        print(f"    Done.")

    # Remix vocals + accompaniment
    print("\nRemixing vocals with accompaniment...")
    min_len = min(len(vocals), len(accompaniment))
    cleaned = accompaniment[:min_len].overlay(vocals[:min_len])
    return cleaned


def replace_words_text(text, pairs):
    """Replace target words in text, preserving case."""
    cleaned = text
    sorted_pairs = sorted(pairs.items(), key=lambda p: len(p[0]), reverse=True)
    for target, replace in sorted_pairs:
        pattern = re.compile(re.escape(target), re.IGNORECASE)
        def match_case(match, _replace=replace):
            original = match.group()
            if original.isupper():
                return _replace.upper()
            if original[0].isupper():
                return _replace[0].upper() + _replace[1:]
            return _replace
        cleaned = pattern.sub(match_case, cleaned)
    return cleaned


def main():
    if len(sys.argv) < 2:
        print("Usage: python lyrics_cleaner.py <song.mp3> [targetwords.txt] [model_size]")
        print()
        print("  song.mp3         Path to the MP3 file")
        print("  targetwords.txt  Path to target words CSV (default: targetwords.txt)")
        print("  model_size       Whisper model: tiny, base, small, medium, large (default: base)")
        sys.exit(1)

    mp3_path = Path(sys.argv[1])
    if not mp3_path.exists():
        print(f"Error: File not found: {mp3_path}")
        sys.exit(1)

    words_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("targetwords.txt")
    if not words_path.exists():
        print(f"Error: Target words file not found: {words_path}")
        sys.exit(1)

    model_size = sys.argv[3] if len(sys.argv) > 3 else "base"

    # Load target words
    pairs = load_target_words(words_path)
    print(f"Loaded {len(pairs)} target word(s) from {words_path}")

    # Step 1: Transcribe with word timestamps
    lyrics, words = transcribe_with_timestamps(mp3_path, model_size)

    print()
    print("=" * 60)
    print("ORIGINAL LYRICS")
    print("=" * 60)
    print(lyrics)

    # Step 2: Find target words
    replacements = find_target_words(words, pairs)

    cleaned_text = replace_words_text(lyrics, pairs)
    print()
    print("=" * 60)
    print("CLEANED LYRICS")
    print("=" * 60)
    print(cleaned_text)

    if not replacements:
        print("\nNo target words found in the audio. Nothing to do.")
        return

    # Step 3: Separate vocals from instruments
    vocals_path, no_vocals_path = separate_vocals(mp3_path)

    # Step 4: Extract a clean voice sample for cloning
    print("\nExtracting voice sample for cloning...")
    voice_sample_path = extract_voice_sample(vocals_path, words, replacements)

    # Step 5: Build cleaned audio with voice-cloned replacements
    cleaned_audio = build_clean_audio(replacements, vocals_path, no_vocals_path, voice_sample_path)

    # Save outputs
    stem = mp3_path.stem
    output_dir = mp3_path.parent

    original_txt = output_dir / f"{stem}_original.txt"
    cleaned_txt_file = output_dir / f"{stem}_cleaned.txt"
    cleaned_mp3 = output_dir / f"{stem}_clean.mp3"

    original_txt.write_text(lyrics, encoding="utf-8")
    cleaned_txt_file.write_text(cleaned_text, encoding="utf-8")

    print(f"\nExporting cleaned MP3...")
    cleaned_audio.export(str(cleaned_mp3), format="mp3", bitrate="192k")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Original lyrics: {original_txt}")
    print(f"Cleaned lyrics:  {cleaned_txt_file}")
    print(f"Cleaned MP3:     {cleaned_mp3}")


if __name__ == "__main__":
    main()
