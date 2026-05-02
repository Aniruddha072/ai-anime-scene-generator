"""
scripts/04_assemble_video.py

Combines an animation clip (MP4) with a voice audio (WAV) into a final video.
- Auto-picks the latest files from animation_clips/ and audio/ if no args given.
- Loops the video if audio is longer than the clip.
- Exports to assets/final_videos/ as final_scene_<timestamp>.mp4
"""

import argparse
from pathlib import Path
from datetime import datetime

from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips


# ── Cinematic settings ──────────────────────────────────
FADE_DURATION  = 1.0   # seconds for fade-in and fade-out
AUDIO_DELAY    = 0.5   # seconds of silence before narration begins


# ── Utility ─────────────────────────────────────────────────

def find_latest_file(folder: str, extension: str) -> Path:
    """
    Return the most recently modified file with the given extension
    inside `folder`. Raises FileNotFoundError if none found.

    Args:
        folder    : path to the directory to search (str or Path)
        extension : file extension to look for, e.g. ".mp4" or ".wav"

    Returns:
        Path object pointing to the latest matching file.
    """
    folder = Path(folder)
    matches = sorted(folder.glob(f"*{extension}"), key=lambda f: f.stat().st_mtime)

    if not matches:
        raise FileNotFoundError(
            f"No '{extension}' files found in '{folder}'. "
            "Run the earlier pipeline steps first."
        )

    return matches[-1]  # most recently modified


# ── Core assembly logic ──────────────────────────────────────

def assemble_video(video_path: str, audio_path: str) -> str:
    """
    Merge animation clip with voice audio and export a cinematic final video.

    Pipeline:
      1. Load video + audio.
      2. Loop video if audio is longer, then trim to exact audio duration.
      3. Delay audio start by AUDIO_DELAY seconds (cinematic pause before narration).
      4. Attach audio to video.
      5. Apply fade-in and fade-out of FADE_DURATION seconds.
      6. Export as MP4 (libx264 / aac / 24 fps).

    Args:
        video_path : path to the .mp4 animation file
        audio_path : path to the .wav audio file

    Returns:
        Path string of the exported final video.
    """

    # ── Load clips ───────────────────────────────────────────
    print(f"\n🎬 Loading video  → {video_path}")
    video = VideoFileClip(str(video_path))
    print(f"   Video duration : {video.duration:.2f}s")

    print(f"\n🎙️  Loading audio  → {audio_path}")
    audio = AudioFileClip(str(audio_path))
    print(f"   Audio duration : {audio.duration:.2f}s")

    # ── Loop video if audio is longer ────────────────────────
    if audio.duration > video.duration:
        print(
            f"\n🔁 Audio ({audio.duration:.2f}s) is longer than video "
            f"({video.duration:.2f}s) → looping video..."
        )
        loops_needed = int(audio.duration / video.duration) + 1
        video = concatenate_videoclips([video] * loops_needed)
        print(f"   Looped video duration : {video.duration:.2f}s")

    # ── Trim video to exact audio duration ───────────────────
    print(f"\n✂️  Trimming video to {audio.duration:.2f}s...")
    video = video.subclip(0, audio.duration)

    # ── Delay audio start (cinematic pause before narration) ─
    print(f"\n⏳ Adding {AUDIO_DELAY}s audio delay (opening silence)...")
    audio = audio.set_start(AUDIO_DELAY)

    # ── Attach audio to video ────────────────────────────────
    print("\n🔗 Attaching audio to video...")
    final_clip = video.set_audio(audio)

    # ── Apply cinematic fade-in and fade-out ─────────────────
    print(f"\n✨ Applying fade-in ({FADE_DURATION}s) and fade-out ({FADE_DURATION}s)...")
    final_clip = final_clip.fadein(FADE_DURATION).fadeout(FADE_DURATION)

    # ── Prepare output path ──────────────────────────────────
    output_dir = Path("assets/final_videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"final_scene_{timestamp}.mp4"

    # ── Export ───────────────────────────────────────────────
    print(f"\n📦 Exporting final video → {output_path}")
    final_clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=24,
        logger=None,        # suppress MoviePy's verbose ffmpeg output
    )

    # ── Clean up ─────────────────────────────────────────────
    video.close()
    audio.close()
    final_clip.close()

    print(f"\n✅ Done! Final video saved → {output_path}")
    return str(output_path)


# ── CLI ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Assemble animation + audio into a final video."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to animation .mp4 file. Auto-picks latest if omitted.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to voice .wav file. Auto-picks latest if omitted.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Resolve video path ───────────────────────────────────
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Provided video not found: {video_path}")
    else:
        print("📂 No --video given → auto-picking latest .mp4 from animation_clips/")
        video_path = find_latest_file("assets/animation_clips", ".mp4")
        print(f"   Found: {video_path}")

    # ── Resolve audio path ───────────────────────────────────
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Provided audio not found: {audio_path}")
    else:
        print("📂 No --audio given → auto-picking latest .wav from audio/")
        audio_path = find_latest_file("assets/audio", ".wav")
        print(f"   Found: {audio_path}")

    # ── Run assembly ─────────────────────────────────────────
    assemble_video(str(video_path), str(audio_path))
