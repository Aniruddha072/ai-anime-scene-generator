"""
scripts/03_generate_voice.py

Generates anime-style AI voice narration using Bark (local TTS).
  - Runs fully on CPU (slow but free)
  - Supports multiple speaker voices
  - Handles long text by splitting into sentence chunks
  - Exports a clean .wav file

Install:
    pip install git+https://github.com/suno-ai/bark.git
    pip install scipy

Usage:
    python scripts/03_generate_voice.py
    python scripts/03_generate_voice.py --text "A lone samurai stands at dawn."
    python scripts/03_generate_voice.py --text "Your narration." --speaker v2/en_speaker_6
    python scripts/03_generate_voice.py --text "Your narration." --output assets/audio/custom.wav
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np


# ── Speaker presets (Bark built-in) ─────────────────────────
# Voice reference: https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683
SPEAKER_PRESETS = {
    "narrator_m":  "v2/en_speaker_6",   # deep, clear male — best for anime narration
    "narrator_f":  "v2/en_speaker_9",   # calm, clear female
    "dramatic_m":  "v2/en_speaker_3",   # slightly dramatic male
    "soft_f":      "v2/en_speaker_0",   # soft, gentle female
}

DEFAULT_SPEAKER  = "narrator_m"
MAX_CHUNK_CHARS  = 200   # Bark works best with chunks under ~200 chars
SILENCE_DURATION = 0.3   # seconds of silence inserted between chunks


# ── Bark model loader ────────────────────────────────────────
def load_bark():
    """
    Imports and preloads Bark models into RAM.
    Called once — subsequent calls use cached models.
    """
    print("   Loading Bark models (first run downloads ~1.6GB)...")
    try:
        from bark import preload_models
        preload_models()
        print("   ✅ Bark models ready.")
    except ImportError:
        print("\n❌ Bark is not installed.")
        print("   Run: pip install git+https://github.com/suno-ai/bark.git")
        sys.exit(1)


# ── Text chunker ─────────────────────────────────────────────
def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Splits narration text into sentence-level chunks safe for Bark.

    Strategy:
      1. Split on sentence boundaries (. ! ?)
      2. Merge short sentences to avoid unnecessary pauses
      3. Hard-split any chunk still over max_chars

    Args:
        text     : Full narration string.
        max_chars: Target max characters per chunk.

    Returns:
        List of clean text chunks.
    """
    # Split on sentence-ending punctuation, keeping the punctuation
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current = ""

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence keeps us under limit, merge
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            # Flush current chunk if non-empty
            if current:
                chunks.append(current)
            # Hard-split sentence if it's too long on its own
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i : i + max_chars])
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


# ── Audio generation ─────────────────────────────────────────
def generate_chunk_audio(text_chunk: str, speaker_preset: str) -> np.ndarray:
    """
    Generates audio for a single text chunk using Bark.

    Args:
        text_chunk     : Short string (≤ MAX_CHUNK_CHARS).
        speaker_preset : Bark voice preset string.

    Returns:
        1-D NumPy float32 array of audio samples.
    """
    from bark import SAMPLE_RATE, generate_audio

    print(f'   🎙️  Generating: "{text_chunk[:60]}{"..." if len(text_chunk) > 60 else ""}"')

    audio = generate_audio(text_chunk, history_prompt=speaker_preset)
    return audio.astype(np.float32)


def make_silence(sample_rate: int, duration: float) -> np.ndarray:
    """Returns a silence buffer of the given duration."""
    return np.zeros(int(sample_rate * duration), dtype=np.float32)


# ── Save WAV ─────────────────────────────────────────────────
def save_wav(audio: np.ndarray, sample_rate: int, path: str) -> None:
    """
    Saves audio as a 16-bit PCM WAV file using scipy.

    Args:
        audio      : NumPy float32 array in range [-1, 1].
        sample_rate: Bark's native sample rate.
        path       : Destination file path.
    """
    try:
        from scipy.io.wavfile import write as wav_write
    except ImportError:
        print("\n❌ scipy is not installed.")
        print("   Run: pip install scipy")
        sys.exit(1)

    # Clip to [-1, 1] to avoid clipping artifacts, convert to int16
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)

    wav_write(path, sample_rate, audio_int16)


# ── Main function ────────────────────────────────────────────
def generate_voice(
    text: str,
    speaker: str = DEFAULT_SPEAKER,
    output_path: str | None = None,
) -> str:
    """
    Full pipeline: text → chunked Bark synthesis → WAV file.

    Args:
        text        : Narration text (any length).
        speaker     : Key from SPEAKER_PRESETS or a raw Bark preset string.
        output_path : Optional explicit output path. Auto-generated if None.

    Returns:
        Path to the saved .wav file.
    """
    from bark import SAMPLE_RATE

    # ── Resolve speaker preset ───────────────────────────────
    speaker_preset = SPEAKER_PRESETS.get(speaker, speaker)
    print(f"\n🎤 Voice generation starting...")
    print(f"   Speaker : {speaker} → {speaker_preset}")
    print(f"   Text    : {text[:80]}{'...' if len(text) > 80 else ''}")

    # ── Split text into safe chunks ──────────────────────────
    chunks = split_into_chunks(text)
    print(f"   Chunks  : {len(chunks)}")

    # ── Generate audio per chunk ─────────────────────────────
    audio_segments = []

    for i, chunk in enumerate(chunks):
        print(f"\n   [{i+1}/{len(chunks)}]")
        chunk_audio = generate_chunk_audio(chunk, speaker_preset)
        audio_segments.append(chunk_audio)

        # Add short silence between chunks (not after the last one)
        if i < len(chunks) - 1:
            audio_segments.append(make_silence(SAMPLE_RATE, SILENCE_DURATION))

    # ── Concatenate all segments ─────────────────────────────
    full_audio = np.concatenate(audio_segments)

    duration_sec = len(full_audio) / SAMPLE_RATE
    print(f"\n   Audio duration: {duration_sec:.2f}s")

    # ── Build output path ────────────────────────────────────
    if output_path is None:
        output_dir = Path("assets/audio")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_speaker = speaker.replace("/", "_")
        output_path = str(output_dir / f"narration_{safe_speaker}_{timestamp}.wav")

    # ── Save WAV ─────────────────────────────────────────────
    save_wav(full_audio, SAMPLE_RATE, output_path)
    print(f"   ✅ Saved → {output_path}")

    return output_path


# ── Default cinematic story lines ───────────────────────────
# Each item is a short line of narration.
# Natural pauses are inserted between them by format_story().
DEFAULT_STORY_LINES = [
    "At the edge of silence...",
    "a lone samurai stands.",
    "The wind carries stories of forgotten battles...",
    "And today,",
    "a new chapter begins.",
]


def format_story(story_lines: list[str]) -> str:
    """
    Join story lines with ellipsis pauses to create natural cinematic pacing.

    Each line is separated by " ... " which Bark interprets as a breath pause.
    Trailing ellipses on lines are preserved so they don't double-up awkwardly.

    Example output:
        'At the edge of silence... ... a lone samurai stands. ... And today, ...'

    Args:
        story_lines: List of short narration strings.

    Returns:
        Single formatted string ready for synthesis.
    """
    joined = " ... ".join(line.strip() for line in story_lines if line.strip())
    return joined


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate anime-style AI voice narration using Bark TTS."
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help=(
            "Narration text to synthesize. "
            "If omitted, the built-in cinematic story lines are used."
        ),
    )
    parser.add_argument(
        "--speaker", "-s",
        type=str,
        default=DEFAULT_SPEAKER,
        help=(
            f"Speaker preset. Options: {', '.join(SPEAKER_PRESETS.keys())} "
            f"or any raw Bark preset (e.g. v2/en_speaker_6). "
            f"Default: {DEFAULT_SPEAKER}"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional explicit output .wav path",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="Print available speaker presets and exit",
    )

    args = parser.parse_args()

    # ── List speakers ────────────────────────────────────────
    if args.list_speakers:
        print("\n\u2019\u2014 Available speaker presets:")
        for name, preset in SPEAKER_PRESETS.items():
            marker = " \u2190 default" if name == DEFAULT_SPEAKER else ""
            print(f"   {name:<14} \u2192 {preset}{marker}")
        sys.exit(0)

    # ── Resolve narration text ───────────────────────────────
    if args.text:
        # User supplied text explicitly via --text
        narration_text = args.text
        print("\u2139\ufe0f  Using provided --text argument.")
    else:
        # Fall back to the cinematic story lines with formatted pauses
        narration_text = format_story(DEFAULT_STORY_LINES)
        print("\u2139\ufe0f  No --text given. Using default cinematic story:")
        for line in DEFAULT_STORY_LINES:
            print(f"   \u2022 {line}")
        print(f"   Formatted: \"{narration_text[:80]}...\"")

    # ── Load Bark (once) ─────────────────────────────────────
    load_bark()

    # ── Generate ─────────────────────────────────────────────
    generate_voice(
        text=narration_text,
        speaker=args.speaker,
        output_path=args.output,
    )
