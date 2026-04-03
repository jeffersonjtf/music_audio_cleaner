import sys
import subprocess
import csv
import re
import tempfile
import os
from pathlib import Path
from datetime import datetime


class _Tee:
    """Mirror stdout to terminal (verbatim) and log file (with per-line timestamps)."""

    def __init__(self, terminal, log_fh):
        self._terminal = terminal
        self._log_fh   = log_fh
        self._buf      = ""  # partial-line buffer for timestamp stamping

    def write(self, data):
        self._terminal.write(data)
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            ts = datetime.now().strftime("%H:%M:%S.%f")[:12]  # HH:MM:SS.mmm
            self._log_fh.write(f"[{ts}] {line}\n")

    def flush(self):
        self._terminal.flush()
        self._log_fh.flush()


def _setup_logging():
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    now  = datetime.now()
    path = os.path.join(logs_dir, f"terminal_{now.strftime('%Y-%m-%d_%H%M%S')}.log")
    log_fh = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    return path, log_fh


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
        ("pronouncing", "pronouncing"),
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


# ---------------------------------------------------------------------------
# Phonetic child-friendly replacement selection
# ---------------------------------------------------------------------------

# Blocklist of words not appropriate for children. Conservative — covers
# profanity, sexual terms, and violent language commonly found in adult music.
_ADULT_WORDS = {
    "ass", "arse", "asshole", "bastard", "bitch", "bitches", "booty", "bugger",
    "bullshit", "cock", "cocks", "crap", "cum", "cunt", "cunts", "damn", "damned",
    "dick", "dicks", "dildo", "dumbass", "dyke", "fag", "faggot", "fags", "fart",
    "fuck", "fucked", "fucker", "fuckers", "fucking", "fucks", "goddamn", "goddamned",
    "hell", "homo", "horny", "jackass", "jerk", "jizz", "kill", "kills", "killing",
    "kys", "motherfucker", "motherfuckers", "motherfucking", "negro", "nigga",
    "nigger", "niggers", "penis", "piss", "pissed", "prick", "pricks", "pussy",
    "pussies", "rape", "raped", "rapist", "retard", "retarded", "sex", "sexy",
    "shit", "shits", "shitty", "slut", "sluts", "spunk", "suck", "sucking",
    "tit", "tits", "twat", "twats", "vagina", "wank", "wanker", "whore", "whores",
    # softer terms still flagged
    "come", "boobs", "boner", "butt", "crap", "dammit", "freaking", "friggin",
    "frigging", "hoe", "hoes", "ho", "humping", "jerk-off", "kinky", "lust",
    "lusty", "naked", "nude", "orgasm", "pervert", "pimp", "pimping", "smut",
    "stripper", "thot", "twerk", "twerking", "whoring",
}


def _is_child_friendly(word: str) -> bool:
    """Return True if word is not in the adult-content blocklist."""
    return word.strip().lower() not in _ADULT_WORDS


def _get_stressed_vowel(phones_str: str) -> str | None:
    """Extract the ARPAbet symbol of the primary-stressed vowel from a phones string."""
    vowels = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
              "IH", "IY", "OW", "OY", "UH", "UW"}
    for ph in phones_str.split():
        if ph.endswith("1") and ph[:-1] in vowels:
            return ph[:-1]
    return None


def _phonetic_score(target: str, replacement: str) -> dict:
    """Score phonetic compatibility of replacement vs target on three axes.

    Returns a dict:
      score          0–3 (higher = more compatible)
      syllables_match, stress_match, vowel_match  — bool each
      target_syllables, repl_syllables            — int
      target_stress, repl_stress                  — str  (e.g. "1", "10", "010")
      target_vowel,  repl_vowel                   — str|None  (ARPAbet, e.g. "AH")
      cmu_found                                   — bool (False if either word not in CMU dict)
    """
    import pronouncing

    t_phones = pronouncing.phones_for_word(target.lower())
    r_phones = pronouncing.phones_for_word(replacement.lower())

    if not t_phones or not r_phones:
        return {"score": 0, "syllables_match": False, "stress_match": False,
                "vowel_match": False, "cmu_found": False,
                "target_syllables": None, "repl_syllables": None,
                "target_stress": None, "repl_stress": None,
                "target_vowel": None, "repl_vowel": None}

    tp = t_phones[0]
    rp = r_phones[0]

    t_syll  = pronouncing.syllable_count(tp)
    r_syll  = pronouncing.syllable_count(rp)
    t_stress = pronouncing.stresses(tp)
    r_stress = pronouncing.stresses(rp)
    t_vowel  = _get_stressed_vowel(tp)
    r_vowel  = _get_stressed_vowel(rp)

    syll_ok   = t_syll == r_syll
    stress_ok = t_stress == r_stress
    vowel_ok  = (t_vowel is not None and t_vowel == r_vowel)
    score     = int(syll_ok) + int(stress_ok) + int(vowel_ok)

    return {"score": score, "syllables_match": syll_ok, "stress_match": stress_ok,
            "vowel_match": vowel_ok, "cmu_found": True,
            "target_syllables": t_syll, "repl_syllables": r_syll,
            "target_stress": t_stress, "repl_stress": r_stress,
            "target_vowel": t_vowel, "repl_vowel": r_vowel}


def _find_best_replacement(target: str) -> tuple[str, dict]:
    """Auto-select the best child-friendly, phonetically-matched replacement
    from the CMU Pronouncing Dictionary.

    Priority:
      1. Same syllable count + same stressed vowel + child-friendly
      2. Same syllable count + child-friendly  (vowel relaxed)
      3. First child-friendly 1-syllable word found  (last resort)
    """
    import pronouncing

    t_phones = pronouncing.phones_for_word(target.lower())
    if not t_phones:
        return ("beep", {"score": 0, "cmu_found": False})

    tp       = t_phones[0]
    t_stress = pronouncing.stresses(tp)
    t_syll   = pronouncing.syllable_count(tp)
    t_vowel  = _get_stressed_vowel(tp)

    # All CMU words with same stress pattern
    candidates = pronouncing.search_stresses(f"^{re.escape(t_stress)}$")

    # Score and filter
    scored = []
    for w in candidates:
        if w.lower() == target.lower():
            continue
        if not _is_child_friendly(w):
            continue
        sc = _phonetic_score(target, w)
        if sc["syllables_match"]:
            scored.append((sc["score"], w, sc))

    if scored:
        # Sort by score desc, then alphabetically for stability
        scored.sort(key=lambda x: (-x[0], x[1]))
        _, best_word, best_sc = scored[0]
        return (best_word, best_sc)

    # Fallback: same syllable count, child-friendly, any vowel
    all_words = [w for w in pronouncing.search_stresses(r"\d")
                 if w.lower() != target.lower() and _is_child_friendly(w)]
    same_syll = []
    for w in all_words:
        wp = pronouncing.phones_for_word(w.lower())
        if wp and pronouncing.syllable_count(wp[0]) == t_syll:
            same_syll.append(w)
    if same_syll:
        best = same_syll[0]
        return (best, _phonetic_score(target, best))

    return ("beep", {"score": 0, "cmu_found": False})


def _validate_and_resolve_replacement(target: str, suggested: str | None) -> tuple[str, dict]:
    """Validate a user-suggested replacement and fall back to auto-selection if needed.

    Returns (final_word, report) where report contains:
      source          — "user_suggestion" | "auto_selected"
      rejected_reason — str | None  (why user suggestion was rejected)
      phonetic        — result from _phonetic_score
    """
    if suggested:
        child_ok = _is_child_friendly(suggested)
        ps = _phonetic_score(target, suggested)

        if not child_ok:
            auto_word, auto_ps = _find_best_replacement(target)
            return (auto_word, {"source": "auto_selected",
                                "rejected_reason": f"'{suggested}' is not child-friendly",
                                "phonetic": auto_ps, "rejected_word": suggested})

        if ps["score"] < 2:
            reason = []
            if not ps["syllables_match"]:
                reason.append(f"syllable mismatch ({ps['target_syllables']} vs {ps['repl_syllables']})")
            if not ps["vowel_match"]:
                reason.append(f"vowel mismatch ({ps['target_vowel']} vs {ps['repl_vowel']})")
            auto_word, auto_ps = _find_best_replacement(target)
            return (auto_word, {"source": "auto_selected",
                                "rejected_reason": f"'{suggested}' score {ps['score']}/3: " + "; ".join(reason),
                                "phonetic": auto_ps, "rejected_word": suggested})

        # Suggestion passes both checks
        return (suggested, {"source": "user_suggestion", "rejected_reason": None, "phonetic": ps})

    # No suggestion — auto-select
    auto_word, auto_ps = _find_best_replacement(target)
    return (auto_word, {"source": "auto_selected", "rejected_reason": None, "phonetic": auto_ps})


def _print_phonetic_report(target: str, report: dict) -> None:
    """Print the phonetic selection report for one target word."""
    ps     = report["phonetic"]
    source = report["source"]
    final  = report.get("final_word", "?")

    if report["rejected_reason"]:
        print(f"    ✗ Rejected '{report['rejected_word']}': {report['rejected_reason']}")

    src_tag = "user suggestion" if source == "user_suggestion" else "auto-selected"
    print(f"    Replacement: \"{final}\"  [{src_tag}]")

    if not ps.get("cmu_found", True):
        print(f"    (word not in CMU dict — phonetic score unavailable)")
        return

    def tick(ok): return "✓" if ok else "✗"
    print(f"    Phonetic compatibility ({ps['score']}/3):")
    print(f"      Syllables:      {ps['target_syllables']} {'==' if ps['syllables_match'] else '!='} {ps['repl_syllables']}  {tick(ps['syllables_match'])}")
    print(f"      Stress pattern: {ps['target_stress']} {'==' if ps['stress_match'] else '!='} {ps['repl_stress']}  {tick(ps['stress_match'])}")
    tv = ps['target_vowel'] or '?'
    rv = ps['repl_vowel']   or '?'
    print(f"      Stressed vowel: {tv} {'==' if ps['vowel_match'] else '!='} {rv}  {tick(ps['vowel_match'])}")
    if ps["score"] == 3:
        print(f"      → Excellent phonetic fit")
    elif ps["score"] == 2:
        print(f"      → Acceptable fit")
    else:
        print(f"      → Poor fit (audio processing will need to compensate)")


def load_target_words(filepath):
    """Load and resolve target→replace pairs from CSV.

    Replace column is optional. Each replacement is:
      1. Validated for child-friendliness
      2. Scored for phonetic compatibility (syllables, stress, vowel)
      3. Replaced by auto-selection if it fails either check or is missing

    Returns dict: {target_lower: resolved_replacement_word}
    Phonetic reports are printed during loading.
    """
    pairs = {}
    print()
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target  = row["Target"].strip().lower()
            if not target:
                continue
            suggest = row.get("Replace", "").strip() or None

            final_word, report = _validate_and_resolve_replacement(target, suggest)
            report["final_word"] = final_word

            print(f'  Target: "{target}"')
            _print_phonetic_report(target, report)
            print()

            pairs[target] = final_word
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

    # Step 4: F0 contour transfer — warp pitch to follow original word's melodic arc
    y_orig, _ = librosa.load(str(vocals_path), sr=vc_sr, offset=start_s,
                              duration=max(end_s - start_s, 0.1))
    vc_trimmed = _transfer_f0_contour(vc_trimmed, vc_sr, y_orig, vc_sr)

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


_MAX_CORRECTION_ROUNDS = 5


def _seg_to_np(seg):
    """Convert pydub AudioSegment to float32 mono numpy array. Returns (array, sample_rate)."""
    raw = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if seg.channels == 2:
        raw = raw.reshape(-1, 2).mean(axis=1)
    raw /= (2 ** (seg.sample_width * 8 - 1))
    return raw, seg.frame_rate


def _np_to_seg(y, sr):
    """Convert float32 mono numpy array to a mono 16-bit pydub AudioSegment."""
    y_int = np.clip(y * 32767, -32768, 32767).astype(np.int16)
    return AudioSegment(y_int.tobytes(), frame_rate=int(sr), sample_width=2, channels=1)


def _apply_splice(vocals_base, repl_seg, mute_start, mute_end, original_vocals, fade_ms):
    """Splice repl_seg into vocals_base at [mute_start:mute_end] ms with equal-power crossfades.
    original_vocals is the pristine untouched vocal track used only for the -24 dB duck zone."""
    before = vocals_base[:mute_start]

    fade_out_end = min(mute_start + fade_ms, mute_end)
    fade_out_zone = _apply_eq_power_fade(
        vocals_base[mute_start:fade_out_end], fade_ms, fade_in=False
    )

    duck_start  = fade_out_end
    duck_end    = max(mute_end - fade_ms, duck_start)
    ducked_zone = original_vocals[duck_start:duck_end].apply_gain(-24)

    fade_in_start = duck_end
    fade_in_zone  = _apply_eq_power_fade(
        vocals_base[fade_in_start:mute_end], fade_ms, fade_in=True
    )

    after   = vocals_base[mute_end:]
    spliced = before + fade_out_zone + ducked_zone + fade_in_zone + after
    spliced = spliced.overlay(repl_seg, position=mute_start)
    return spliced


def _auto_correct(repl_np, sr, metrics, vocals_ctx, mute_start_ms, mute_end_ms, fade_ms):
    """Apply targeted per-metric corrections to repl_np based on failed _verify_replacement results.

    Corrections applied (in order):
      1. Directed F0 pitch shift  — uses stored f0_ref/f0_rep for signed correction
      2. Spectral re-match        — re-runs _spectral_match against verification context window
      3. Splice-in level ramp     — boosts/attenuates the replacement entry window
      4. Splice-out level ramp    — boosts/attenuates the replacement exit window
      5. RMS normalization        — scales overall loudness to match context

    Each correction is a partial step (0.6×) toward the ideal to avoid oscillation.
    """
    import librosa

    y = repl_np.copy()

    # Extract surrounding context audio from the vocals track at this splice point
    ctx_raw, ctx_sr = _seg_to_np(vocals_ctx)
    ctx_samp   = int(2.0 * ctx_sr)
    ctx_start  = int(mute_start_ms * ctx_sr / 1000)
    ctx_end    = int(mute_end_ms   * ctx_sr / 1000)
    ctx_before = ctx_raw[max(0, ctx_start - ctx_samp):ctx_start]
    ctx_after  = ctx_raw[ctx_end:min(len(ctx_raw), ctx_end + ctx_samp)]
    context_np = np.concatenate([ctx_before, ctx_after])

    # Resample context to match repl_np SR if needed (for spectral/RMS comparisons)
    if ctx_sr != sr and len(context_np) > 0:
        context_np = librosa.resample(context_np, orig_sr=ctx_sr, target_sr=sr)

    fade_samp = int(fade_ms * sr / 1000)
    snap_samp = int(0.05 * sr)  # 50ms — matches _verify_replacement measurement window

    # --- 1. Directed F0 pitch correction ---
    f0_ref = metrics.get("f0_ref")
    f0_rep = metrics.get("f0_rep")
    if not metrics["f0_ok"] and f0_ref and f0_rep and f0_ref > 0 and f0_rep > 0:
        n_steps = float(np.clip(12.0 * np.log2(f0_ref / f0_rep), -6.0, 6.0))
        if abs(n_steps) > 0.3:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(np.float32)

    # --- 2. Spectral re-match against context ---
    if not metrics.get("centroid_ok", True) or not metrics.get("rolloff_ok", True):
        if len(context_np) >= 512:
            y = _spectral_match(y, context_np, sr)

    def _level_ramp(arr, win_start, win_end, gain):
        """Apply a smoothly tapered gain envelope to arr[win_start:win_end]."""
        if win_end <= win_start or win_start >= len(arr):
            return arr
        win_end = min(win_end, len(arr))
        n = win_end - win_start
        taper = max(1, n // 6)
        env = np.full(n, gain, dtype=np.float32)
        env[:taper]  = np.linspace(1.0, gain, taper)
        env[-taper:] = np.linspace(gain, 1.0, taper)
        arr = arr.copy()
        arr[win_start:win_end] *= env
        return arr

    # --- 3. Splice-in boundary correction ---
    # splice_in = rms(post_in) / rms(pre_in)
    # post_in is 50ms of replacement after the fade-in window.
    # Multiply that window by (1/splice_in) to bring ratio toward 1.0.
    splice_in = metrics["splice_in_ratio"]
    if not (0.5 <= splice_in <= 2.0):
        raw_gain = float(np.clip(1.0 / max(splice_in, 1e-3), 0.25, 4.0))
        gain = 1.0 + (raw_gain - 1.0) * 0.6
        y = _level_ramp(y, fade_samp, fade_samp + snap_samp, gain)

    # --- 4. Splice-out boundary correction ---
    # splice_out = rms(post_out) / rms(pre_out)
    # pre_out is 50ms of replacement just before the fade-out window.
    # Multiply that window by splice_out to bring ratio toward 1.0.
    splice_out = metrics["splice_out_ratio"]
    if not (0.5 <= splice_out <= 2.0):
        raw_gain = float(np.clip(splice_out, 0.25, 4.0))
        gain = 1.0 + (raw_gain - 1.0) * 0.6
        fade_out_start = max(0, len(y) - fade_samp)
        y = _level_ramp(y, fade_out_start - snap_samp, fade_out_start, gain)

    # --- 5. RMS normalization ---
    if not metrics["rms_ok"]:
        rms_ctx = float(np.sqrt(np.mean(context_np ** 2))) if len(context_np) > 0 else 1.0
        rms_y   = float(np.sqrt(np.mean(y ** 2))) + 1e-9
        gain    = float(np.clip(rms_ctx / rms_y, 0.5, 2.0))
        y = y * gain

    return y.astype(np.float32)


def _mean_f0_correct_fallback(y_vc, sr_vc, y_orig, sr_orig):
    """Single mean-pitch shift. Used when the clip is too short for contour transfer."""
    import librosa

    def mean_f0(y, sr):
        if len(y) < 2048:
            return 0.0
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        vf = f0[voiced & ~np.isnan(f0)] if f0 is not None else np.array([])
        return float(np.mean(vf)) if len(vf) > 0 else 0.0

    mean_orig = mean_f0(y_orig, sr_orig)
    mean_vc   = mean_f0(y_vc,   sr_vc)

    if mean_orig <= 0 or mean_vc <= 0:
        return y_vc

    semitones = float(np.clip(12.0 * np.log2(mean_orig / mean_vc), -12, 12))
    if abs(semitones) < 0.5:
        return y_vc

    print(f"    F0 correction (fallback mean): {mean_vc:.1f} Hz → {mean_orig:.1f} Hz ({semitones:+.1f} st)")
    return librosa.effects.pitch_shift(y_vc, sr=sr_vc, n_steps=semitones).astype(np.float32)


def _transfer_f0_contour(y_vc, sr_vc, y_orig, sr_orig):
    """Spline-based F0 contour transfer: replaces single mean-pitch shift with
    per-segment shifts derived from a time-normalized F0 curve extracted from
    the original sung word. The replacement follows the melodic arc of the
    original rather than just landing on the average note.

    Splits audio into N segments (min 4096 samples each, max 6).
    For each segment uses the median of the smoothed semitone-offset curve
    over that time slice, then applies librosa.pitch_shift independently.

    Falls back to mean-pitch correction when either signal has fewer than
    4 reliably voiced pyin frames (too short for contour estimation).
    """
    import librosa
    from scipy.ndimage import gaussian_filter1d

    HOP      = 256    # pyin hop length (frames)
    MIN_SEG  = 4096   # min samples per segment (~93 ms @ 44100 Hz)
    MAX_SEGS = 6      # cap to limit phase-vocoder passes
    SIGMA    = 4      # Gaussian smoothing width (frames) on semitone curve

    def extract_f0_full(y, sr):
        if len(y) < 2048:
            return None, None
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, hop_length=HOP,
        )
        return f0, voiced

    f0_orig, v_orig = extract_f0_full(y_orig, sr_orig)
    f0_vc,   v_vc   = extract_f0_full(y_vc,   sr_vc)

    if f0_orig is None or f0_vc is None:
        return _mean_f0_correct_fallback(y_vc, sr_vc, y_orig, sr_orig)

    valid_orig = v_orig & ~np.isnan(f0_orig) & (f0_orig > 0)
    valid_vc   = v_vc   & ~np.isnan(f0_vc)   & (f0_vc   > 0)

    if valid_orig.sum() < 4 or valid_vc.sum() < 4:
        return _mean_f0_correct_fallback(y_vc, sr_vc, y_orig, sr_orig)

    # Time-normalize: interpolate f0_orig onto vc's frame axis so different
    # word durations map onto the same 0–1 time axis before comparison.
    t_orig = np.linspace(0, 1, len(f0_orig))
    t_vc   = np.linspace(0, 1, len(f0_vc))
    f0_orig_resampled = np.interp(t_vc, t_orig[valid_orig], f0_orig[valid_orig])

    # Per-frame semitone offset where vc is voiced
    semitone_curve = np.zeros(len(f0_vc))
    semitone_curve[valid_vc] = 12.0 * np.log2(
        f0_orig_resampled[valid_vc] / (f0_vc[valid_vc] + 1e-9)
    )
    # Fill unvoiced frames with voiced mean so smoothing doesn't pull toward 0
    mean_shift = float(np.mean(semitone_curve[valid_vc]))
    semitone_curve[~valid_vc] = mean_shift
    semitone_curve = gaussian_filter1d(semitone_curve, sigma=SIGMA)
    semitone_curve = np.clip(semitone_curve, -12, 12)

    n_seg = max(1, min(MAX_SEGS, len(y_vc) // MIN_SEG))

    if n_seg == 1:
        # Too short for multi-segment — apply single median shift from the curve
        shift = float(np.median(semitone_curve[valid_vc]))
        if abs(shift) < 0.5:
            return y_vc
        print(f"    F0 contour (1 seg): {shift:+.1f} st")
        return librosa.effects.pitch_shift(y_vc, sr=sr_vc, n_steps=shift).astype(np.float32)

    # Apply per-segment shifts derived from the smoothed contour curve
    boundaries = np.linspace(0, len(y_vc), n_seg + 1, dtype=int)
    result_segs = []
    shifts_applied = []

    for i in range(n_seg):
        seg = y_vc[boundaries[i]:boundaries[i + 1]]

        f_start = int(i       / n_seg * len(semitone_curve))
        f_end   = int((i + 1) / n_seg * len(semitone_curve))
        shift = float(np.median(semitone_curve[f_start:f_end]))
        shift = float(np.clip(shift, -12, 12))
        shifts_applied.append(shift)

        if abs(shift) >= 0.3:
            seg = librosa.effects.pitch_shift(seg, sr=sr_vc, n_steps=shift)

        result_segs.append(seg.astype(np.float32))

    print(f"    F0 contour ({n_seg} segs): " +
          " / ".join(f"{s:+.1f}" for s in shifts_applied) + " st")

    return np.concatenate(result_segs).astype(np.float32)


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
                        context_s=2.0, orig_word_np=None, fade_ms=30, mix_word_np=None):
    """Compare the replaced section against 2s of surrounding audio on 10 axes.

    orig_word_np: the original sung word as a numpy array (same sample rate as
    vocals_after). When provided, F0 and contour metrics compare against this
    specific note rather than the 2s context window.

    fade_ms: the crossfade window used during blending (default 30). Used to
    offset the splice-continuity measurement windows past the transition zone
    so they reflect steady-state replacement level rather than the ramp itself.

    Checks:
      1.  RMS energy ratio           — loudness match
      2.  Spectral centroid drift    — tonal brightness match
      3.  Splice-in  RMS continuity  — level at steady-state entry vs pre-edit
      4.  Splice-out RMS continuity  — level at pre-exit vs post-edit
      5.  ZCR discontinuity          — click/pop at splice edges
      6.  MFCC cosine distance       — voice/timbre identity match
      7.  F0 mean pitch match        — replacement on the same note as original?
      8.  Spectral rolloff match     — high-frequency brightness match
      9.  Chroma cosine similarity   — harmonic/musical key match
      10. F0 contour correlation     — melodic arc shape match (contour transfer quality)
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
    # Windows are offset by fade_ms past the splice boundary so they measure
    # steady-state levels (after the crossfade completes) rather than the ramp
    # itself — this gives values that are meaningfully close to 1.0 when the
    # blend is working correctly.
    snap      = int(0.05  * frame_rate)  # 50 ms measurement window
    fade_samp = int(fade_ms * frame_rate / 1000)

    # Splice-in:  50ms before splice (original) vs 50ms starting after the fade-in
    pre_in  = raw[max(0, start_s - snap):start_s]
    post_in = raw[start_s + fade_samp : start_s + fade_samp + snap]

    # Splice-out: 50ms ending before the fade-out starts vs 50ms after splice
    pre_out  = raw[max(0, end_s - fade_samp - snap) : end_s - fade_samp]
    post_out = raw[end_s:end_s + snap]

    splice_in  = rms(post_in)  / (rms(pre_in)  + 1e-9)
    splice_out = rms(post_out) / (rms(pre_out) + 1e-9)
    results["splice_in_ratio"]  = round(splice_in,  3)
    results["splice_out_ratio"] = round(splice_out, 3)
    results["splice_ok"] = (0.5 <= splice_in <= 2.0 and 0.5 <= splice_out <= 2.0)

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
        results["f0_ref"]           = round(float(f0_ref), 2)  # stored for directed auto-correction
        results["f0_rep"]           = round(float(f0_rep), 2)
    else:
        results["f0_semitone_diff"] = None
        results["f0_ok"]            = True
        results["f0_ref"]           = None
        results["f0_rep"]           = None

    # ------------------------------------------------------------------ 7b. F0 vs full song mix (informational)
    # Extracts F0 from the original unseparated mix at this word's timestamp.
    # Instruments add noise but this shows how the replacement sits in the full musical context.
    # Not counted as a warning — informational display only.
    if mix_word_np is not None and len(mix_word_np) >= 2048:
        f0_mix = mean_f0(mix_word_np)
        f0_cur = results.get("f0_rep") or (mean_f0(seg_replaced) if len(seg_replaced) >= 2048 else None)
        if f0_mix and f0_cur:
            results["f0_mix_diff"] = round(abs(12.0 * np.log2(f0_cur / f0_mix)), 2)
            results["f0_mix_ref"]  = round(float(f0_mix), 2)
        else:
            results["f0_mix_diff"] = None
            results["f0_mix_ref"]  = None
    else:
        results["f0_mix_diff"] = None
        results["f0_mix_ref"]  = None

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

    # ------------------------------------------------------------------ 10. F0 contour correlation (contour transfer quality)
    # Compares the melodic shape of the replacement against the original word.
    # Uses Pearson r on time-normalized voiced F0 frames.
    # Only meaningful when orig_word_np is provided and both clips are long enough.
    def voiced_f0_curve(x, sr):
        if len(x) < 2048:
            return None
        f0, voiced, _ = librosa.pyin(
            x, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, hop_length=256,
        )
        valid = voiced & ~np.isnan(f0) & (f0 > 0) if f0 is not None else np.zeros(0, bool)
        return (f0, valid) if valid.sum() >= 4 else None

    contour_orig = voiced_f0_curve(orig_word_np, frame_rate) if orig_word_np is not None else None
    contour_rep  = voiced_f0_curve(seg_replaced, frame_rate)

    if contour_orig is not None and contour_rep is not None:
        f0_o, v_o = contour_orig
        f0_r, v_r = contour_rep
        # Time-normalize both to 32 points for a stable correlation estimate
        N = 32
        t_o = np.linspace(0, 1, len(f0_o))
        t_r = np.linspace(0, 1, len(f0_r))
        t_n = np.linspace(0, 1, N)
        curve_o = np.interp(t_n, t_o[v_o], f0_o[v_o])
        curve_r = np.interp(t_n, t_r[v_r], f0_r[v_r])
        # Pearson correlation on log-Hz so semitone differences are linear
        log_o = np.log2(curve_o + 1e-9)
        log_r = np.log2(curve_r + 1e-9)
        corr = float(np.corrcoef(log_o, log_r)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        results["f0_contour_corr"] = round(corr, 3)
        results["f0_contour_ok"]   = corr > 0.50  # <0.5 = contour shape doesn't match
    else:
        results["f0_contour_corr"] = None
        results["f0_contour_ok"]   = True  # not enough data → don't penalise

    # ------------------------------------------------------------------ Verdict
    ok_flags = [
        results["rms_ok"], results["centroid_ok"], results["splice_ok"],
        results["zcr_ok"], results["mfcc_ok"], results["f0_ok"],
        results["rolloff_ok"], results["chroma_ok"], results["f0_contour_ok"],
    ]
    warn_count = ok_flags.count(False)
    results["warn_count"] = warn_count
    results["verdict"]    = "PASS" if warn_count == 0 else ("WARN" if warn_count <= 2 else "FAIL")
    return results


def _print_verification(metrics, word, phonetic_score=None, round_label=""):
    verdict = metrics["verdict"]
    icons = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}
    tag = icons.get(verdict, "?")

    def flag(ok, msg):
        return "" if ok else f"  ← {msg}"

    label_suffix = f"  [{round_label}]" if round_label else ""
    print(f"    [{tag}] Verification for \"{word}\"  ({verdict}, {metrics['warn_count']} warning(s)){label_suffix}:")
    if phonetic_score is not None:
        ps = phonetic_score
        if ps.get("cmu_found", False):
            tv = ps.get("target_vowel") or "?"
            rv = ps.get("repl_vowel")   or "?"
            vowel_note = f"  ({tv}={'==' if ps['vowel_match'] else '!='}{rv})"
            print(f"        Phonetic fit:              {ps['score']}/3"
                  + f"  (syll {'✓' if ps['syllables_match'] else '✗'}"
                  + f"  stress {'✓' if ps['stress_match'] else '✗'}"
                  + f"  vowel {'✓' if ps['vowel_match'] else '✗'})"
                  + vowel_note)
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
        print(f"        F0 vs isolated vocals:     {metrics['f0_semitone_diff']:.2f} st  (ideal < 3.0 st)"
              + flag(metrics["f0_ok"], "replacement is in the wrong musical key"))
    if metrics.get("f0_mix_diff") is not None:
        print(f"        F0 vs original mix:        {metrics['f0_mix_diff']:.2f} st  (ref {metrics['f0_mix_ref']:.1f} Hz, informational)")
    if metrics["chroma_distance"] is not None:
        print(f"        Chroma key similarity:     {metrics['chroma_distance']:.4f}  (ideal < 0.20)"
              + flag(metrics["chroma_ok"], "replacement sits in wrong harmonic space"))
    if metrics.get("f0_contour_corr") is not None:
        print(f"        F0 contour correlation:    {metrics['f0_contour_corr']:.3f}   (ideal > 0.50)"
              + flag(metrics["f0_contour_ok"], "melodic arc shape doesn't match original"))
    conclusion = {
        "PASS": "→ Replacement sounds integrated.",
        "WARN": "→ Minor issues detected — may be acceptable.",
        "FAIL": "→ Replacement likely audible — consider re-running.",
    }
    print(f"        {conclusion[verdict]}")


def find_target_words(words, target_pairs):
    """Find words in transcript that match target words.
    Each replacement dict includes phonetic_score for use in verification output."""
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
                repl = target_pairs[phrase_clean]
                replacements.append({
                    "original": phrase_text,
                    "replacement": repl,
                    "start": phrase_words[0]["start"],
                    "end": phrase_words[-1]["end"],
                    "phonetic_score": _phonetic_score(phrase_clean, repl),
                })
                i += phrase_len
                matched = True
                break

        if not matched:
            word_clean = re.sub(r"[^\w]", "", words[i]["word"]).lower()
            if word_clean in target_pairs:
                repl = target_pairs[word_clean]
                replacements.append({
                    "original": words[i]["word"],
                    "replacement": repl,
                    "start": words[i]["start"],
                    "end": words[i]["end"],
                    "phonetic_score": _phonetic_score(word_clean, repl),
                })
            i += 1

    return replacements


def build_clean_audio(replacements, vocals_path, no_vocals_path, voice_sample_path,
                      mix_path=None):
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

    # Load original mix for secondary F0 reference (full song, pre-separation)
    mix_np = None
    mix_sr = None
    if mix_path is not None:
        print("  Loading original mix for F0 reference comparison...")
        mix_np, mix_sr = librosa.load(str(mix_path), sr=None, mono=True)

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

        # Convert to numpy so corrections can be applied iteratively
        repl_np, sr_vc = _seg_to_np(tts)

        # Snapshot of vocals before this word's splice — used for re-trying after corrections
        vocals_before_word = vocals

        # Original word audio for F0 reference (extracted from unmodified isolated vocals)
        orig_word_np = vocals_np[int(snapped_start * vocals_sr):int(snapped_end * vocals_sr)]

        # Full-mix reference segment for secondary F0 comparison (informational)
        mix_word_np = None
        if mix_np is not None and mix_sr is not None:
            mix_word_np = mix_np[int(snapped_start * mix_sr):int(snapped_end * mix_sr)]

        for attempt in range(_MAX_CORRECTION_ROUNDS + 1):
            # Convert corrected numpy back to AudioSegment; resample to vocals rate if needed
            repl_seg = _np_to_seg(repl_np, sr_vc)
            if repl_seg.frame_rate != vocals_before_word.frame_rate:
                repl_seg = repl_seg.set_frame_rate(vocals_before_word.frame_rate)

            # Splice into a candidate vocals track (non-destructive — base is unchanged)
            vocals_candidate = _apply_splice(
                vocals_before_word, repl_seg, mute_start, mute_end,
                original_vocals, fade_ms
            )

            # Verify: compare replaced section against surrounding context on 10 axes
            metrics = _verify_replacement(
                vocals_candidate, mute_start, mute_end, vocals_sr,
                orig_word_np=orig_word_np, fade_ms=fade_ms,
                mix_word_np=mix_word_np
            )
            round_label = "initial" if attempt == 0 else f"round {attempt}"
            _print_verification(metrics, r["replacement"],
                                phonetic_score=r.get("phonetic_score"),
                                round_label=round_label)

            if metrics["verdict"] == "PASS" or attempt == _MAX_CORRECTION_ROUNDS:
                vocals = vocals_candidate
                break

            # Apply targeted corrections to the numpy replacement for next round
            print(f"    Auto-correcting ({round_label} → round {attempt + 1})...")
            repl_np = _auto_correct(
                repl_np, sr_vc, metrics, vocals_before_word,
                mute_start, mute_end, fade_ms
            )

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
                                      verify_audio.frame_rate, orig_word_np=orig_word_np,
                                      fade_ms=30)
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
    log_path, _log_fh = _setup_logging()
    print(f"Logging to: {log_path}")

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
        replacements, vocals_path, no_vocals_path, voice_sample_path,
        mix_path=mp3_path
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
