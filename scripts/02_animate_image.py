"""
scripts/02_animate_image.py

Cinematic Ken Burns animation with depth and atmosphere:
  - Fake parallax: 3 layers (background / midground / foreground) at different speeds
  - Floating particles (petals / dust) with low opacity
  - Subtle drifting fog overlay
  - Ease-in-out easing — smooth, no jitter
  - Motion plays once, then holds final frame

Timeline:
  [0s → 9s]   slow eased motion with atmospheric effects
  [9s → 12s]  freeze on final frame

Usage:
    python scripts/02_animate_image.py
    python scripts/02_animate_image.py --image path/to/img.png
    python scripts/02_animate_image.py --image img.png --effect pan_right
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime


# ── Constants ────────────────────────────────────────────────
FPS             = 24
MOVE_DURATION   = 9     # seconds of actual motion
FREEZE_DURATION = 3     # seconds the final frame is held
TOTAL_DURATION  = MOVE_DURATION + FREEZE_DURATION  # 12s total

DEFAULT_EFFECT  = "zoom_in"
ZOOM_SCALE      = 1.20  # gentle 20% zoom

# Parallax offset multipliers (relative to midground = 1.0)
# Applied as pixel shifts AFTER the single global camera transform.
# Keep values very close to 1.0 for stability — subtle depth is the goal.
BG_SPEED  = 0.85  # background drifts slightly less than camera
MID_SPEED = 1.0   # midground follows camera exactly (reference layer)
FG_SPEED  = 1.15  # foreground drifts slightly more than camera

# Particle settings
NUM_PARTICLES   = 40
PARTICLE_ALPHA  = 0.35

# Fog settings
FOG_ALPHA = 0.06   # very subtle — must not dominate


# ── Easing ───────────────────────────────────────────────────
def ease_in_out(t: float) -> float:
    """Smoothstep: slow start → gradual → slow end. t in [0, 1]."""
    return t * t * (3.0 - 2.0 * t)


# ── Gradient masks for parallax layers ───────────────────────
def build_layer_masks(H: int, W: int):
    """
    Build 3 gradient masks (bg, mid, fg) that sum to exactly 1.0
    at every pixel. Blending is along the vertical axis:
      - background  : dominates at the top (sky, mountains)
      - foreground  : dominates at the bottom (trees, ground)
      - midground   : fills the middle band
    """
    y = np.linspace(0, 1, H, dtype=np.float32)

    bg_col  = np.clip(1.0 - 2.0 * y, 0.0, 1.0)   # 1→0 in top half
    fg_col  = np.clip(2.0 * y - 1.0, 0.0, 1.0)   # 0→1 in bottom half
    mid_col = 1.0 - bg_col - fg_col                # remainder in middle

    # Expand to (H, W, 1) for direct float32 compositing
    bg_mask  = np.tile(bg_col [:, None, None], (1, W, 1))
    mid_mask = np.tile(mid_col[:, None, None], (1, W, 1))
    fg_mask  = np.tile(fg_col [:, None, None], (1, W, 1))

    return bg_mask, mid_mask, fg_mask


# ── Crop & resize helper ─────────────────────────────────────
def crop_and_resize(image: np.ndarray, cx: float, cy: float, zoom: float) -> np.ndarray:
    """
    Crop image centered at (cx, cy) with given zoom, resize back to full dims.
    cx, cy in [0, 1]. zoom > 1 means more zoomed in.
    """
    H, W = image.shape[:2]
    crop_w = int(W / zoom)
    crop_h = int(H / zoom)

    x0 = int(cx * W - crop_w / 2)
    y0 = int(cy * H - crop_h / 2)
    x0 = max(0, min(x0, W - crop_w))
    y0 = max(0, min(y0, H - crop_h))

    cropped = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)


# ── Pixel-shift helper ───────────────────────────────────────
def _shift_horizontal(frame: np.ndarray, px: int) -> np.ndarray:
    """
    Shift `frame` left (px < 0) or right (px > 0) by |px| pixels.
    Edge columns are replicated to fill the gap — no black bars, no wrap-around.
    """
    if px == 0:
        return frame
    W = frame.shape[1]
    result = np.empty_like(frame)
    if px > 0:
        result[:, px:] = frame[:, : W - px]
        result[:, :px] = frame[:, :1]          # replicate left edge
    else:
        s = -px
        result[:, : W - s] = frame[:, s:]
        result[:, W - s :] = frame[:, -1:]     # replicate right edge
    return result


# ── Parallax composite ────────────────────────────────────────
def build_parallax_frame(
    image: np.ndarray,
    bg_mask: np.ndarray,
    mid_mask: np.ndarray,
    fg_mask: np.ndarray,
    cx: float,
    cy: float,
    zoom: float,
) -> np.ndarray:
    """
    Composite 3 depth layers to create a subtle parallax illusion.

    Architecture (single-transform, shift-only):
      1. ONE global camera crop — identical for every layer.
         This guarantees zero shake: all layers share the same base image.
      2. Apply SMALL horizontal pixel shifts to bg and fg AFTER the crop.
         bg shifts slightly less than camera → appears farther away.
         fg shifts slightly more than camera → appears closer.
      3. Blend layers using pre-computed gradient masks.

    For zoom_in / zoom_out: cx = cy = 0.5, so dx = 0 → all shifts are 0.
    Depth during zoom comes from the focus change, not lateral movement.
    """
    _, W = image.shape[:2]

    # ── Step 1: single global camera transform ───────────────
    base = crop_and_resize(image, cx, cy, zoom)

    # ── Step 2: compute parallax pixel offsets ───────────────
    # dx: how far the camera has moved from center, in normalized coords [−0.5, 0.5]
    dx = cx - 0.5

    # Convert to pixels. Scale by (speed − 1.0) so midground gets 0 shift.
    bg_px  = int(dx * W * (BG_SPEED  - MID_SPEED))  # negative → shifts left
    fg_px  = int(dx * W * (FG_SPEED  - MID_SPEED))  # positive → shifts right

    bg_frame  = _shift_horizontal(base, bg_px)
    mid_frame = base                                  # no shift — reference layer
    fg_frame  = _shift_horizontal(base, fg_px)

    # ── Step 3: blend using gradient masks ───────────────────
    composite = (
        bg_frame.astype(np.float32)  * bg_mask  +
        mid_frame.astype(np.float32) * mid_mask +
        fg_frame.astype(np.float32)  * fg_mask
    )
    return np.clip(composite, 0, 255).astype(np.uint8)


# ── Particle system (petals / dust) ──────────────────────────
class ParticleSystem:
    """
    Floating particles that drift downward with slight sideways variation.
    Drawn at low opacity so they feel atmospheric, not distracting.
    """

    def __init__(self, W: int, H: int, n: int = NUM_PARTICLES):
        rng = np.random.default_rng(seed=7)  # fixed seed = reproducible look
        self.W, self.H = W, H

        self.x  = rng.uniform(0, W, n).astype(np.float32)
        self.y  = rng.uniform(0, H, n).astype(np.float32)
        self.vy = rng.uniform(0.3, 0.9, n).astype(np.float32)   # downward speed
        self.vx = rng.uniform(-0.2, 0.3, n).astype(np.float32)  # gentle rightward drift
        self.r  = rng.integers(2, 5, n)                          # radius in px

        # Random per-particle phase for sinusoidal wind sway
        self._phase = rng.uniform(0, 2 * np.pi, n).astype(np.float32)

        # Petal pink in BGR
        raw = rng.uniform([180, 160, 200], [215, 200, 255], (n, 3))
        self.colors = [(int(c[0]), int(c[1]), int(c[2])) for c in raw]

    def step(self, frame_idx: int = 0):
        """
        Advance positions by one frame.
        Horizontal motion combines constant drift with a sinusoidal wind gust
        so particles sway naturally rather than moving in a straight line.
        """
        # Sinusoidal wind component — each particle has a slightly different phase
        wind = np.sin(frame_idx * 0.04 + self._phase) * 0.3
        self.x += self.vx + wind
        self.y += self.vy
        self.x = np.where(self.x > self.W, 0,      self.x)
        self.x = np.where(self.x < 0,     self.W,  self.x)
        self.y = np.where(self.y > self.H, 0,      self.y)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw all particles onto the frame at PARTICLE_ALPHA opacity."""
        overlay = frame.copy()
        for i in range(len(self.x)):
            cv2.circle(
                overlay,
                (int(self.x[i]), int(self.y[i])),
                int(self.r[i]),
                self.colors[i],
                -1,
                cv2.LINE_AA,
            )
        return cv2.addWeighted(frame, 1 - PARTICLE_ALPHA, overlay, PARTICLE_ALPHA, 0)


# ── Fog overlay ───────────────────────────────────────────────
def draw_fog(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Overlay a slowly drifting semi-transparent mist band.
    Fog moves consistently to the right — one direction only, no oscillation.
    `frame_idx` drives a linear horizontal scroll so motion is smooth and stable.
    Very subtle (FOG_ALPHA = 0.06) — should barely be noticed.
    """
    H, W = frame.shape[:2]

    # Vertical Gaussian band slightly above center
    y_coords     = np.linspace(0, 1, H, dtype=np.float32)
    vert_profile = np.exp(-0.5 * ((y_coords - 0.45) / 0.25) ** 2)

    # Linear horizontal scroll: fog completes ~0.4 of a full cycle over 9 seconds
    scroll_speed = 0.4 / (MOVE_DURATION * FPS)   # fraction of width per frame
    offset       = (frame_idx * scroll_speed) % 1.0

    # Build a wide fog strip and roll it left→right
    x_base        = np.linspace(0, 2 * np.pi, W, dtype=np.float32)
    horiz_profile = 0.6 + 0.4 * np.sin(x_base + offset * 2 * np.pi)

    # Combine into (H, W, 1) alpha map
    alpha_map = np.outer(vert_profile, horiz_profile)[:, :, np.newaxis] * FOG_ALPHA

    fog_color = np.full_like(frame, 225, dtype=np.float32)  # near-white
    result    = frame.astype(np.float32) * (1 - alpha_map) + fog_color * alpha_map
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Effect parameter builder ──────────────────────────────────
def get_effect_params(effect: str, move_frames: int) -> list:
    """
    Return (cx, cy, zoom) per frame for the motion phase.
    All values eased with smoothstep so motion is slow-start, slow-end.
    """
    params = []
    for i in range(move_frames):
        t = ease_in_out(i / (move_frames - 1))

        if effect == "zoom_in":
            zoom = 1.0 + (ZOOM_SCALE - 1.0) * t
            cx, cy = 0.5, 0.5

        elif effect == "zoom_out":
            zoom = ZOOM_SCALE - (ZOOM_SCALE - 1.0) * t
            cx, cy = 0.5, 0.5

        elif effect == "pan_right":
            zoom = 1.15
            cx, cy = 0.38 + 0.24 * t, 0.5

        elif effect == "pan_left":
            zoom = 1.15
            cx, cy = 0.62 - 0.24 * t, 0.5

        elif effect == "pan_up":
            zoom = 1.15
            cx, cy = 0.5, 0.62 - 0.24 * t

        elif effect == "pan_down":
            zoom = 1.15
            cx, cy = 0.5, 0.38 + 0.24 * t

        else:
            raise ValueError(
                f"Unknown effect '{effect}'. "
                "Choose: zoom_in | zoom_out | pan_right | pan_left | pan_up | pan_down"
            )

        params.append((cx, cy, zoom))
    return params


# ── Core animation function ───────────────────────────────────
def animate_image(
    image_path: str,
    effect: str = DEFAULT_EFFECT,
    output_path: str | None = None,
) -> str:
    """
    Generate a cinematic MP4 from a static image.

    Pipeline per frame:
      1. Parallax composite   (3 layers at different speeds)
      2. Particle system step + draw  (floating petals/dust)
      3. Fog overlay          (slow horizontal drift)

    Final 3 seconds: last frame held as a freeze.

    Args:
        image_path  : Input PNG/JPG path.
        effect      : zoom_in | zoom_out | pan_right | pan_left | pan_up | pan_down
        output_path : Output .mp4 path. Auto-generated if None.

    Returns:
        Path of the saved .mp4 file.
    """
    # ── Load image ───────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    H, W = image.shape[:2]
    move_frames   = int(FPS * MOVE_DURATION)    # 9s × 24 = 216 frames
    freeze_frames = int(FPS * FREEZE_DURATION)  # 3s × 24 = 72 frames
    total_frames  = move_frames + freeze_frames  # 288 frames

    print(f"\n🎬 Animating image (cinematic + depth + atmosphere)...")
    print(f"   Input      : {image_path}")
    print(f"   Effect     : {effect}")
    print(f"   Motion     : {MOVE_DURATION}s  → {move_frames} frames")
    print(f"   Freeze hold: {FREEZE_DURATION}s  → {freeze_frames} frames")
    print(f"   Total      : {TOTAL_DURATION}s  ({total_frames} frames @ {FPS}fps)")
    print(f"   Resolution : {W}x{H}")

    # ── Pre-build expensive objects once ─────────────────────
    print("   Building parallax masks...")
    bg_mask, mid_mask, fg_mask = build_layer_masks(H, W)

    print("   Initialising particle system...")
    particles = ParticleSystem(W, H)

    # ── Output path ──────────────────────────────────────────
    if output_path is None:
        output_dir = Path("assets/animation_clips")
        output_dir.mkdir(parents=True, exist_ok=True)
        stem      = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"{stem}_{effect}_{timestamp}.mp4")

    # ── VideoWriter ──────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter failed to open: {output_path}")

    # ── Phase 1: motion frames ───────────────────────────────
    print("\n   ▶  Rendering motion frames...")
    frame_params = get_effect_params(effect, move_frames)
    last_frame   = None

    for idx, (cx, cy, zoom) in enumerate(frame_params):
        # 1. Single-transform parallax composite (no shake possible)
        frame = build_parallax_frame(image, bg_mask, mid_mask, fg_mask, cx, cy, zoom)

        # 2. Particles — sinusoidal sideways drift
        particles.step(frame_idx=idx)
        frame = particles.draw(frame)

        # 3. Fog — consistent rightward drift
        frame = draw_fog(frame, frame_idx=idx)

        last_frame = frame
        writer.write(frame)

        if idx % (FPS // 2) == 0 or idx == move_frames - 1:
            pct = int((idx + 1) / move_frames * 100)
            print(f"      {pct}%", end="\r")

    print("      100% — motion complete")

    # ── Phase 2: freeze frames ───────────────────────────────
    print(f"   ⏸  Writing {freeze_frames} freeze frames (hold final position)...")
    for _ in range(freeze_frames):
        writer.write(last_frame)

    writer.release()
    print(f"\n   ✅ Saved → {output_path}")
    return output_path


# ── Auto-detect latest image ──────────────────────────────────
def get_latest_image() -> str:
    """Return path to most recently modified image in assets/generated_images/."""
    img_dir = Path("assets/generated_images")
    if not img_dir.exists():
        raise FileNotFoundError(
            "No 'assets/generated_images' directory found. "
            "Run 01_generate_image.py first."
        )
    images = sorted(
        list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not images:
        raise FileNotFoundError(
            "No images found in assets/generated_images/. "
            "Run 01_generate_image.py first."
        )
    return str(images[0])


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply cinematic Ken Burns + depth + atmosphere to an anime image."
    )
    parser.add_argument(
        "--image", "-i",
        type=str, default=None,
        help="Input image path (default: latest in assets/generated_images/)",
    )
    parser.add_argument(
        "--effect", "-e",
        type=str, default=DEFAULT_EFFECT,
        choices=["zoom_in", "zoom_out", "pan_right", "pan_left", "pan_up", "pan_down"],
        help=f"Animation effect (default: {DEFAULT_EFFECT})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Optional output .mp4 path",
    )

    args = parser.parse_args()

    if args.image is None:
        print("ℹ️  No --image provided. Auto-detecting latest generated image...")
        image_path = get_latest_image()
        print(f"   Using: {image_path}")
    else:
        image_path = args.image

    animate_image(
        image_path=image_path,
        effect=args.effect,
        output_path=args.output,
    )
