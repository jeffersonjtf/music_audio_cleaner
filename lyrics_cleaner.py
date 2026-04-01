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
    Pipeline:
    1. XTTS-v2 generates speech (phonemes/timing) — no pre-VC stretch
    2. SEED-VC converts it to the singer's voice (30 diffusion steps for quality)
    3. F0 correction aligns pitch center to original sung word
    4. Spectral matching aligns timbre/EQ to original recording
    5. Single time-stretch to exact target duration"""
    import librosa

    tts_model = get_tts_model()

    # Step 1: Generate TTS
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tts_raw_path = tmp.name

    tts_model.tts_to_file(
        text=text,
        speaker_wav=str(voice_sample_path),
        language="en",
        file_path=tts_raw_path,
    )

    # Step 2: Trim TTS — relaxed threshold preserves natural reverb tail
    y, sr = librosa.load(tts_raw_path, sr=None)
    Path(tts_raw_path).unlink(missing_ok=True)

    y, _ = librosa.effects.trim(y, top_db=50)  # was 30 — looser = keeps reverb tail

    if len(y) == 0:
        return AudioSegment.silent(duration=duration_ms)

    # Save raw TTS directly for SEED-VC (no pre-VC stretch — avoids double-stretch artifacts)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tts_for_vc_path = tmp.name
    sf.write(tts_for_vc_path, y, sr)

    # Step 3: Voice conversion with SEED-VC
    print(f"    Converting to singer's voice with SEED-VC...")
    wrapper = get_seedvc_wrapper()

    vc_sr = None
    vc_audio = None
    for mp3_bytes, full_audio in wrapper.convert_voice(
        source=tts_for_vc_path,
        target=str(voice_sample_path),
        diffusion_steps=30,         # was 10 — higher = much better voice quality
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        f0_condition=True,
        auto_f0_adjust=True,
        pitch_shift=0,
        stream_output=True,
    ):
        if full_audio is not None:
            vc_sr, vc_audio = full_audio

    Path(tts_for_vc_path).unlink(missing_ok=True)

    if vc_audio is None:
        return AudioSegment.silent(duration=duration_ms)

    vc_audio = vc_audio.astype(np.float32)

    # Trim VC output — same relaxed threshold
    vc_trimmed, _ = librosa.effects.trim(vc_audio, top_db=50)
    if len(vc_trimmed) == 0:
        vc_trimmed = vc_audio

    # Step 4: F0 correction — shift pitch center to match original sung word
    y_orig, _ = librosa.load(str(vocals_path), sr=vc_sr, offset=start_s,
                              duration=max(end_s - start_s, 0.1))
    vc_trimmed = _correct_f0(vc_trimmed, vc_sr, y_orig, vc_sr)

    # Step 5: Spectral matching — match EQ/timbre to original recording context
    vc_trimmed = _spectral_match(vc_trimmed, y_orig, vc_sr)

    # Step 6: Single time-stretch to exact target duration (only one pass total)
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


def _eq_power_curve(n, fade_in=True):
    """Equal-power (sin/cos) S-curve for crossfading. Returns float64 array [0,1] or [1,0]."""
    t = np.linspace(0, np.pi / 2, n)
    return np.sin(t) if fade_in else np.cos(t)


def _apply_eq_power_fade(segment, fade_ms, fade_in=True):
    """Apply an equal-power fade-in or fade-out to an AudioSegment using numpy."""
    if len(segment) == 0:
        return segment
    fade_ms = min(fade_ms, len(segment) // 2)
    if fade_ms <= 0:
        return segment

    n_fade = int(fade_ms * segment.frame_rate / 1000)
    channels = segment.channels
    sample_width = segment.sample_width

    arr = np.array(segment.get_array_of_samples(), dtype=np.float64)
    curve = _eq_power_curve(n_fade, fade_in=fade_in)

    for c in range(channels):
        if fade_in:
            arr[c::channels][:n_fade] *= curve
        else:
            arr[c::channels][-n_fade:] *= curve

    max_val = (2 ** (sample_width * 8 - 1)) - 1
    arr = np.clip(arr, -max_val, max_val)
    int_type = np.int16 if sample_width == 2 else np.int32
    return segment._spawn(arr.astype(int_type).tobytes())


def _apply_crossfade(audio, fade_ms=30):
    """Apply equal-power S-curve fade in + fade out."""
    if len(audio) == 0:
        return audio
    fade_ms = min(fade_ms, len(audio) // 2)
    audio = _apply_eq_power_fade(audio, fade_ms, fade_in=True)
    audio = _apply_eq_power_fade(audio, fade_ms, fade_in=False)
    return audio


def _correct_f0(y_vc, sr_vc, y_orig, sr_orig):
    """Pitch-shift VC output so its mean F0 matches the original sung word.
    Corrects the key mismatch between flat TTS prosody and the song's melody."""
    import librosa

    def mean_f0(y, sr):
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        voiced_f0 = f0[voiced & ~np.isnan(f0)] if f0 is not None else np.array([])
        return float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    mean_orig = mean_f0(y_orig, sr_orig)
    mean_vc   = mean_f0(y_vc,   sr_vc)

    if mean_orig <= 0 or mean_vc <= 0:
        return y_vc

    semitones = 12.0 * np.log2(mean_orig / mean_vc)
    semitones = float(np.clip(semitones, -12, 12))

    if abs(semitones) < 0.5:
        return y_vc

    print(f"    F0 correction: {mean_vc:.1f} Hz → {mean_orig:.1f} Hz ({semitones:+.1f} st)")
    return librosa.effects.pitch_shift(y_vc, sr=sr_vc, n_steps=semitones)


def _spectral_match(y_replacement, y_original, sr):
    """Match the per-frequency magnitude envelope of the replacement to the original.
    Ensures the replacement word has the same tonal character as the recording."""
    import librosa
    from scipy.ndimage import uniform_filter1d

    if len(y_original) < 512 or len(y_replacement) < 512:
        return y_replacement

    n_fft = 2048
    D_orig = librosa.stft(y_original,    n_fft=n_fft)
    D_repl = librosa.stft(y_replacement, n_fft=n_fft)

    mag_orig = np.mean(np.abs(D_orig), axis=1) + 1e-8
    mag_repl = np.mean(np.abs(D_repl), axis=1) + 1e-8

    gains = mag_orig / mag_repl
    gains = uniform_filter1d(gains, size=20)        # smooth over frequency bins
    gains = np.clip(gains, 0.1, 10.0)               # ±20 dB max correction

    D_matched = D_repl * gains[:, np.newaxis]
    y_matched = librosa.istft(D_matched, length=len(y_replacement))
    return y_matched.astype(np.float32)


def _snap_to_onset(vocals_np, sr, timestamp_s, window_s=0.25):
    """Snap a Whisper word timestamp to the nearest energy onset in the vocals.
    Corrects Whisper's ±50-200 ms drift so mute/overlay windows align precisely."""
    import librosa

    center = int(timestamp_s * sr)
    half   = int(window_s * sr)
    seg_start = max(0, center - half)
    seg_end   = min(len(vocals_np), center + half)

    if seg_end <= seg_start:
        return timestamp_s

    segment = vocals_np[seg_start:seg_end]

    onsets = librosa.onset.onset_detect(
        y=segment, sr=sr, units="samples", hop_length=128, backtrack=True
    )

    if len(onsets) == 0:
        return timestamp_s

    center_in_seg = center - seg_start
    closest = onsets[int(np.argmin(np.abs(onsets - center_in_seg)))]

    # Clamp snap to ±200 ms of the original Whisper timestamp
    snapped_s = (seg_start + int(closest)) / sr
    snapped_s = float(np.clip(snapped_s, timestamp_s - 0.2, timestamp_s + 0.2))
    return snapped_s


def _verify_replacement(vocals_after, mute_start_ms, mute_end_ms, sr,
                        context_s=2.0, orig_word_np=None):
    """Compare the replaced section against 2s of surrounding audio on 9 axes.

    orig_word_np: the original sung word as a numpy array (same sample rate as
    vocals_after). When provided, F0 is verified against this specific note
    rather than the 2s context window, which contains other melody notes and
    would give a false F0 warning.

    Checks:
      1.  RMS energy ratio          — loudness match
      2.  Spectral centroid drift   — tonal brightness match
      3.  Splice-in  RMS continuity — level jump at entry cut
      4.  Splice-out RMS continuity — level jump at exit cut
      5.  ZCR discontinuity        — click/pop at splice edges
      6.  MFCC cosine distance      — voice/timbre identity match
      7.  F0 pitch match            — is replacement on the same note as original word?
      8.  Spectral rolloff match    — high-frequency brightness match
      9.  Chroma cosine similarity  — harmonic/musical key match
    """
    import librosa
    from scipy.spatial.distance import cosine as cosine_dist

    # ------------------------------------------------------------------ setup
    raw = np.array(vocals_after.get_array_of_samples(), dtype=np.float32)
    if vocals_after.channels == 2:
        raw = raw.reshape(-1, 2).mean(axis=1)
    raw /= (2 ** (vocals_after.sample_width * 8 - 1))

    frame_rate   = vocals_after.frame_rate
    total_samples = len(raw)
    ctx_samples  = int(context_s * frame_rate)
    start_s      = int(mute_start_ms * frame_rate / 1000)
    end_s        = int(mute_end_ms   * frame_rate / 1000)

    seg_before   = raw[max(0, start_s - ctx_samples):start_s]
    seg_replaced = raw[start_s:end_s]
    seg_after    = raw[end_s:min(total_samples, end_s + ctx_samples)]
    seg_ctx      = np.concatenate([seg_before, seg_after])  # combined context

    results = {}

    def rms(x):
        return float(np.sqrt(np.mean(x ** 2))) if len(x) > 0 else 0.0

    def safe_feature(fn, x, fallback=None):
        return fn(x) if len(x) >= 512 else fallback

    # ------------------------------------------------------------------ 1. RMS ratio
    rms_ctx = (rms(seg_before) + rms(seg_after)) / 2 + 1e-9
    rms_ratio = rms(seg_replaced) / rms_ctx
    results["rms_ratio"] = round(rms_ratio, 3)
    results["rms_ok"]    = 0.25 <= rms_ratio <= 4.0

    # ------------------------------------------------------------------ 2. Spectral centroid drift
    def mean_centroid(x):
        return float(np.mean(librosa.feature.spectral_centroid(y=x, sr=frame_rate)))

    c_ctx = safe_feature(mean_centroid, seg_ctx)
    c_rep = safe_feature(mean_centroid, seg_replaced)
    if c_ctx and c_rep:
        drift = abs(c_rep - c_ctx) / (c_ctx + 1e-9)
        results["centroid_drift_pct"] = round(drift * 100, 1)
        results["centroid_ok"]        = drift < 0.60
    else:
        results["centroid_drift_pct"] = None
        results["centroid_ok"]        = True

    # ------------------------------------------------------------------ 3 & 4. Splice-point RMS continuity
    snap = int(0.05 * frame_rate)  # 50 ms window

    pre_in   = raw[max(0, start_s - snap):start_s]
    post_in  = raw[start_s:start_s + snap]
    pre_out  = raw[max(0, end_s - snap):end_s]
    post_out = raw[end_s:end_s + snap]

    splice_in  = rms(post_in)  / (rms(pre_in)  + 1e-9)
    splice_out = rms(post_out) / (rms(pre_out) + 1e-9)
    results["splice_in_ratio"]  = round(splice_in,  3)
    results["splice_out_ratio"] = round(splice_out, 3)
    results["splice_ok"] = (0.1 <= splice_in <= 10.0 and 0.1 <= splice_out <= 10.0)

    # ------------------------------------------------------------------ 5. Zero-crossing rate at splice edges
    def zcr_density(x):
        if len(x) < 2:
            return 0.0
        return float(np.mean(librosa.feature.zero_crossing_rate(x)))

    zcr_ctx_in  = (zcr_density(pre_in)  + zcr_density(post_in))  / 2 + 1e-9
    zcr_ctx_out = (zcr_density(pre_out) + zcr_density(post_out)) / 2 + 1e-9
    zcr_rep     = zcr_density(seg_replaced)
    zcr_ratio   = zcr_rep / ((zcr_ctx_in + zcr_ctx_out) / 2 + 1e-9)
    results["zcr_ratio"] = round(zcr_ratio, 3)
    results["zcr_ok"]    = zcr_ratio < 5.0  # >5× spike = likely click

    # ------------------------------------------------------------------ 6. MFCC cosine distance (voice identity)
    def mean_mfcc(x):
        return np.mean(librosa.feature.mfcc(y=x, sr=frame_rate, n_mfcc=13), axis=1)

    mfcc_ctx = safe_feature(mean_mfcc, seg_ctx)
    mfcc_rep = safe_feature(mean_mfcc, seg_replaced)
    if mfcc_ctx is not None and mfcc_rep is not None:
        mfcc_dist = float(cosine_dist(mfcc_ctx, mfcc_rep))
        results["mfcc_distance"] = round(mfcc_dist, 4)
        results["mfcc_ok"]       = mfcc_dist < 0.15  # >0.15 = noticeably different voice
    else:
        results["mfcc_distance"] = None
        results["mfcc_ok"]       = True

    # ------------------------------------------------------------------ 7. F0 pitch match
    # Reference: the original word's isolated audio (same note as the replacement
    # should be singing). The 2s context window spans multiple melody notes and
    # would give a misleading average — a word can be correct while still being
    # semitones away from the surrounding context's mean pitch.
    def mean_f0(x):
        if len(x) < 2048:
            return None
        f0, voiced, _ = librosa.pyin(
            x, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=frame_rate
        )
        vf = f0[voiced & ~np.isnan(f0)] if f0 is not None else np.array([])
        return float(np.mean(vf)) if len(vf) > 0 else None

    f0_ref = mean_f0(orig_word_np) if orig_word_np is not None and len(orig_word_np) >= 2048 else mean_f0(seg_ctx)
    f0_rep = mean_f0(seg_replaced)
    if f0_ref and f0_rep:
        semitones = abs(12.0 * np.log2(f0_rep / f0_ref))
        results["f0_semitone_diff"] = round(float(semitones), 2)
        results["f0_ok"]            = semitones < 3.0  # >3 semitones = wrong note
    else:
        results["f0_semitone_diff"] = None
        results["f0_ok"]            = True

    # ------------------------------------------------------------------ 8. Spectral rolloff match
    def mean_rolloff(x):
        return float(np.mean(librosa.feature.spectral_rolloff(y=x, sr=frame_rate)))

    ro_ctx = safe_feature(mean_rolloff, seg_ctx)
    ro_rep = safe_feature(mean_rolloff, seg_replaced)
    if ro_ctx and ro_rep:
        rolloff_drift = abs(ro_rep - ro_ctx) / (ro_ctx + 1e-9)
        results["rolloff_drift_pct"] = round(rolloff_drift * 100, 1)
        results["rolloff_ok"]        = rolloff_drift < 0.50  # >50% = brightness mismatch
    else:
        results["rolloff_drift_pct"] = None
        results["rolloff_ok"]        = True

    # ------------------------------------------------------------------ 9. Chroma cosine similarity (musical key)
    def mean_chroma(x):
        return np.mean(librosa.feature.chroma_stft(y=x, sr=frame_rate), axis=1)

    ch_ctx = safe_feature(mean_chroma, seg_ctx)
    ch_rep = safe_feature(mean_chroma, seg_replaced)
    if ch_ctx is not None and ch_rep is not None:
        chroma_dist = float(cosine_dist(ch_ctx + 1e-9, ch_rep + 1e-9))
        results["chroma_distance"] = round(chroma_dist, 4)
        results["chroma_ok"]       = chroma_dist < 0.20  # >0.20 = wrong musical key
    else:
        results["chroma_distance"] = None
        results["chroma_ok"]       = True

    # ------------------------------------------------------------------ Verdict
    ok_flags = [
        results["rms_ok"], results["centroid_ok"], results["splice_ok"],
        results["zcr_ok"], results["mfcc_ok"], results["f0_ok"],
        results["rolloff_ok"], results["chroma_ok"],
    ]
    warn_count = ok_flags.count(False)
    results["warn_count"] = warn_count
    results["verdict"]    = "PASS" if warn_count == 0 else ("WARN" if warn_count <= 2 else "FAIL")
    return results


def _print_verification(metrics, word):
    verdict = metrics["verdict"]
    icons = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}
    tag = icons.get(verdict, "?")

    def flag(ok, msg):
        return "" if ok else f"  ← {msg}"

    print(f"    [{tag}] Verification for \"{word}\"  ({verdict}, {metrics['warn_count']} warning(s)):")
    print(f"        RMS energy ratio:          {metrics['rms_ratio']:.3f}   (ideal ≈ 1.0)"
          + flag(metrics["rms_ok"], "loudness mismatch"))
    if metrics["centroid_drift_pct"] is not None:
        print(f"        Spectral centroid drift:   {metrics['centroid_drift_pct']:.1f}%   (ideal < 60%)"
              + flag(metrics["centroid_ok"], "timbral brightness mismatch"))
    if metrics["rolloff_drift_pct"] is not None:
        print(f"        Spectral rolloff drift:    {metrics['rolloff_drift_pct']:.1f}%   (ideal < 50%)"
              + flag(metrics["rolloff_ok"], "high-freq brightness mismatch"))
    print(f"        Splice-in  continuity:     {metrics['splice_in_ratio']:.3f}   (ideal ≈ 1.0)"
          + flag(metrics["splice_ok"], "abrupt level jump at entry"))
    print(f"        Splice-out continuity:     {metrics['splice_out_ratio']:.3f}   (ideal ≈ 1.0)"
          + flag(metrics["splice_ok"], "abrupt level jump at exit"))
    print(f"        ZCR ratio at splice:       {metrics['zcr_ratio']:.3f}   (ideal < 5.0)"
          + flag(metrics["zcr_ok"], "click/pop detected at splice edge"))
    if metrics["mfcc_distance"] is not None:
        print(f"        MFCC voice distance:       {metrics['mfcc_distance']:.4f}  (ideal < 0.15)"
              + flag(metrics["mfcc_ok"], "voice/timbre sounds like different singer"))
    if metrics["f0_semitone_diff"] is not None:
        print(f"        F0 pitch difference:       {metrics['f0_semitone_diff']:.2f} st  (ideal < 3.0 st)"
              + flag(metrics["f0_ok"], "replacement is in the wrong musical key"))
    if metrics["chroma_distance"] is not None:
        print(f"        Chroma key similarity:     {metrics['chroma_distance']:.4f}  (ideal < 0.20)"
              + flag(metrics["chroma_ok"], "replacement sits in wrong harmonic space"))
    conclusion = {
        "PASS": "→ Replacement sounds integrated.",
        "WARN": "→ Minor issues detected — may be acceptable.",
        "FAIL": "→ Replacement likely audible — consider re-running.",
    }
    print(f"        {conclusion[verdict]}")


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
    import librosa

    print(f"\nLoading separated tracks...")
    vocals = AudioSegment.from_wav(str(vocals_path))
    original_vocals = AudioSegment.from_wav(str(vocals_path))  # pristine copy
    accompaniment = AudioSegment.from_wav(str(no_vocals_path))

    if not replacements:
        print("No target words found — nothing to replace.")
        min_len = min(len(vocals), len(accompaniment))
        return accompaniment[:min_len].overlay(vocals[:min_len])

    # Load vocals as mono numpy array once for onset snapping
    print("  Loading vocals for onset detection...")
    vocals_np, vocals_sr = librosa.load(str(vocals_path), sr=None, mono=True)

    print(f"Replacing {len(replacements)} word(s) with voice-cloned audio...\n")

    fade_ms = 30  # shorter = snappier, less level-sweep artefact at boundaries

    for r in replacements:
        # Snap Whisper timestamps to true energy onsets (corrects ±50-200ms drift)
        snapped_start = _snap_to_onset(vocals_np, vocals_sr, r["start"])
        snapped_end   = _snap_to_onset(vocals_np, vocals_sr, r["end"])

        start_ms = int(snapped_start * 1000)
        end_ms   = int(snapped_end   * 1000)

        # Tighter pad now that timestamps are accurately snapped
        pad = 80  # ms
        mute_start = max(0, start_ms - pad)
        mute_end   = min(len(vocals), end_ms + pad)
        mute_duration = mute_end - mute_start

        print(f"  [{r['start']:.2f}s → {snapped_start:.2f}s] "
              f"\"{r['original']}\" -> \"{r['replacement']}\"")

        tts = clone_voice_word(
            r["replacement"], voice_sample_path, vocals_path,
            snapped_start, snapped_end, mute_duration
        )

        # Volume matching against the snapped segment
        ref_segment = original_vocals[start_ms:end_ms]
        original_loudness = ref_segment.dBFS
        if tts.dBFS > -50 and original_loudness > -50:
            tts = tts.apply_gain(original_loudness - tts.dBFS + 2)

        # Equal-power crossfade on the replacement clip
        tts = _apply_crossfade(tts, fade_ms=fade_ms)

        # --- Crossfaded suppression of original word ---
        # Duck to -24 dB instead of silence: preserves room tone and reverb tail
        before = vocals[:mute_start]

        fade_out_end = min(mute_start + fade_ms, mute_end)
        fade_out_zone = _apply_eq_power_fade(
            vocals[mute_start:fade_out_end], fade_ms, fade_in=False
        )

        duck_start = fade_out_end
        duck_end   = max(mute_end - fade_ms, duck_start)
        # -24 dB keeps reverb/room tone; the replacement overlaid on top dominates
        ducked_zone = original_vocals[duck_start:duck_end].apply_gain(-24)

        fade_in_start = duck_end
        fade_in_zone = _apply_eq_power_fade(
            vocals[fade_in_start:mute_end], fade_ms, fade_in=True
        )

        after = vocals[mute_end:]

        vocals = before + fade_out_zone + ducked_zone + fade_in_zone + after
        vocals = vocals.overlay(tts, position=mute_start)

        # Self-verify: compare replaced section against 2s of surrounding context.
        # Pass the original word audio so F0 is checked against that specific note,
        # not the context window average (which spans many different melody notes).
        orig_word_np = vocals_np[int(snapped_start * vocals_sr):int(snapped_end * vocals_sr)]
        metrics = _verify_replacement(vocals, mute_start, mute_end, vocals_sr,
                                      orig_word_np=orig_word_np)
        _print_verification(metrics, r["replacement"])

    # Remix vocals + accompaniment
    print("\nRemixing vocals with accompaniment...")
    min_len = min(len(vocals), len(accompaniment))
    cleaned = accompaniment[:min_len].overlay(vocals[:min_len])
    # Return the modified vocals-only track alongside the mix so callers can
    # save it as a sidecar for --verify (F0 check needs isolated vocals, not the mix)
    return cleaned, vocals


def save_timestamps(words, path):
    """Cache word timestamps to JSON so --verify skips re-transcription."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(words, f)


def load_timestamps(path):
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_only(mp3_path, words_path):
    """Run the 9-metric verification against an already-cleaned MP3.
    Requires the *_timestamps.json sidecar written during the full pipeline run.
    No models are loaded — just audio math against the existing clean file."""
    stem      = mp3_path.stem
    out_dir   = mp3_path.parent
    clean_mp3 = out_dir / f"{stem}_clean.mp3"
    ts_file   = out_dir / f"{stem}_timestamps.json"

    if not clean_mp3.exists():
        print(f"Error: clean file not found: {clean_mp3}")
        print("Run the full pipeline first to produce the clean MP3.")
        sys.exit(1)
    if not ts_file.exists():
        print(f"Error: timestamps file not found: {ts_file}")
        print("Run the full pipeline once (without --verify) to cache timestamps.")
        sys.exit(1)

    pairs = load_target_words(words_path)
    words = load_timestamps(ts_file)
    replacements = find_target_words(words, pairs)

    if not replacements:
        print("No target words found in cached timestamps.")
        return

    # Use isolated vocals sidecar for verification — the full mix contains
    # instruments that corrupt pitch (F0) and timbre (MFCC) measurements.
    vocals_clean_wav = out_dir / f"{stem}_vocals_clean.wav"
    if vocals_clean_wav.exists():
        print(f"Loading clean vocals track: {vocals_clean_wav}")
        verify_audio = AudioSegment.from_wav(str(vocals_clean_wav))
    else:
        print(f"Loading clean audio (vocals sidecar not found, falling back to mix): {clean_mp3}")
        print("  NOTE: F0 and MFCC checks may be inaccurate due to instrument bleed.")
        print("  Re-run the full pipeline to generate the vocals sidecar.")
        verify_audio = AudioSegment.from_mp3(str(clean_mp3))

    # Load original vocals (cached by demucs) for word-accurate F0 reference
    import librosa
    orig_vocals_path = out_dir / "demucs_output" / mp3_path.stem / "vocals.wav"
    vocals_np = None
    vocals_sr = None
    if orig_vocals_path.exists():
        print(f"  Loading original vocals for F0 reference...")
        vocals_np, vocals_sr = librosa.load(str(orig_vocals_path), sr=None, mono=True)
    else:
        print("  (Original vocals not found — F0 check will use context window as fallback)")

    pad = 80  # ms — same as full pipeline
    print(f"\nVerifying {len(replacements)} replacement(s) against clean file...\n")

    all_pass = True
    for r in replacements:
        start_ms   = int(r["start"] * 1000)
        end_ms     = int(r["end"]   * 1000)
        mute_start = max(0, start_ms - pad)
        mute_end   = min(len(verify_audio), end_ms + pad)

        # Extract the original word audio for accurate F0 reference
        orig_word_np = None
        if vocals_np is not None:
            orig_word_np = vocals_np[int(r["start"] * vocals_sr):int(r["end"] * vocals_sr)]

        print(f"  [{r['start']:.2f}s - {r['end']:.2f}s]  "
              f"\"{r['original']}\" -> \"{r['replacement']}\"")
        metrics = _verify_replacement(verify_audio, mute_start, mute_end,
                                      verify_audio.frame_rate, orig_word_np=orig_word_np)
        _print_verification(metrics, r["replacement"])
        if metrics["verdict"] != "PASS":
            all_pass = False

    print()
    print("=" * 60)
    if all_pass:
        print("ALL REPLACEMENTS PASSED verification.")
    else:
        print("SOME REPLACEMENTS need review (see WARNs/FAILs above).")
    print("=" * 60)


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
    args = sys.argv[1:]
    verify_mode = "--verify" in args
    args = [a for a in args if a != "--verify"]

    if not args:
        print("Usage: python lyrics_cleaner.py <song.mp3> [targetwords.txt] [model_size] [--verify]")
        print()
        print("  song.mp3         Path to the original MP3 file")
        print("  targetwords.txt  Path to target words CSV (default: targetwords.txt)")
        print("  model_size       Whisper model: tiny, base, small, medium, large (default: base)")
        print("  --verify         Re-run only the 9-metric verification against an existing")
        print("                   *_clean.mp3 (uses cached *_timestamps.json — no models loaded)")
        sys.exit(1)

    mp3_path = Path(args[0])
    if not mp3_path.exists():
        print(f"Error: File not found: {mp3_path}")
        sys.exit(1)

    words_path = Path(args[1]) if len(args) > 1 else Path("targetwords.txt")
    if not words_path.exists():
        print(f"Error: Target words file not found: {words_path}")
        sys.exit(1)

    model_size = args[2] if len(args) > 2 else "base"

    # --verify: skip all model loading, just verify the existing clean file
    if verify_mode:
        verify_only(mp3_path, words_path)
        return

    # Load target words
    pairs = load_target_words(words_path)
    print(f"Loaded {len(pairs)} target word(s) from {words_path}")

    # Step 1: Transcribe with word timestamps
    lyrics, words = transcribe_with_timestamps(mp3_path, model_size)

    # Cache timestamps so --verify can run later without re-transcribing
    stem = mp3_path.stem
    output_dir = mp3_path.parent
    timestamps_file = output_dir / f"{stem}_timestamps.json"
    save_timestamps(words, timestamps_file)

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
    cleaned_audio, cleaned_vocals = build_clean_audio(
        replacements, vocals_path, no_vocals_path, voice_sample_path
    )

    # Save outputs
    original_txt       = output_dir / f"{stem}_original.txt"
    cleaned_txt_file   = output_dir / f"{stem}_cleaned.txt"
    cleaned_mp3        = output_dir / f"{stem}_clean.mp3"
    cleaned_vocals_wav = output_dir / f"{stem}_vocals_clean.wav"

    original_txt.write_text(lyrics, encoding="utf-8")
    cleaned_txt_file.write_text(cleaned_text, encoding="utf-8")

    print(f"\nExporting cleaned MP3...")
    cleaned_audio.export(str(cleaned_mp3), format="mp3", bitrate="192k")

    # Save isolated vocals sidecar so --verify can do accurate F0 checks
    # (full mix contains instruments which corrupt pitch detection)
    cleaned_vocals.export(str(cleaned_vocals_wav), format="wav")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Original lyrics: {original_txt}")
    print(f"Cleaned lyrics:  {cleaned_txt_file}")
    print(f"Cleaned MP3:     {cleaned_mp3}")


if __name__ == "__main__":
    main()
