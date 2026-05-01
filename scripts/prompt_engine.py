"""
Stage 1 — Prompt Template Engine
Wraps raw user text in a structured cinematic anime prompt.
"""

# ─── Quality & style anchors ──────────────────────────────────────────────────
QUALITY_ANCHORS = "masterpiece, best quality, ultra-detailed, sharp focus, 8k"

STYLE_ANCHORS = (
    "anime style, by Makoto Shinkai, Studio Ghibli color palette, "
    "cinematic composition, dramatic lighting, volumetric fog, 4k"
)

NEGATIVE_BASE = (
    "lowres, bad anatomy, worst quality, blurry, deformed, ugly, "
    "watermark, text, 3d render, nsfw, extra limbs, missing fingers"
)

# ─── Preset scene templates ───────────────────────────────────────────────────
SCENE_PRESETS = {
    "dawn":      "golden hour dawn, cherry blossoms falling, misty mountain peak, epic cinematic",
    "rain":      "heavy rain, neon reflections on wet asphalt, atmospheric, god rays, cyberpunk",
    "forest":    "twilight, bioluminescent particles, mist, warm ethereal light, dreamlike",
    "ocean":     "vast ocean, towering waves, silver moonlight, lonely lighthouse, melancholy",
    "cityscape": "aerial view, neon-lit city, bustling crowd, vertical pan, hyperdetailed",
}


def build_prompt(raw_text: str, scene_preset: str = None, extra_tags: str = "") -> dict:
    """
    Build a complete image + narration prompt from raw user text.

    Args:
        raw_text:      The user's core scene description.
        scene_preset:  Optional preset key from SCENE_PRESETS.
        extra_tags:    Additional comma-separated style tags.

    Returns:
        dict with 'positive', 'negative', and 'narration' keys.
    """
    # Assemble scene flavor
    preset_flavor = SCENE_PRESETS.get(scene_preset, "") if scene_preset else ""

    # Build positive prompt
    parts = [QUALITY_ANCHORS, raw_text]
    if preset_flavor:
        parts.append(preset_flavor)
    if extra_tags:
        parts.append(extra_tags)
    parts.append(STYLE_ANCHORS)

    positive = ", ".join(filter(None, parts))
    negative = NEGATIVE_BASE

    # Narration script (clean — no comma-separated tags)
    narration = f"In this scene, {raw_text.lower().rstrip('.')}."

    return {
        "positive":  positive,
        "negative":  negative,
        "narration": narration,
    }


# ─── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = build_prompt(
        raw_text="lone samurai standing on a misty mountain peak",
        scene_preset="dawn",
    )
    print("=" * 60)
    print("✅ POSITIVE PROMPT:")
    print(result["positive"])
    print()
    print("🚫 NEGATIVE PROMPT:")
    print(result["negative"])
    print()
    print("🎙️  NARRATION SCRIPT:")
    print(result["narration"])
    print("=" * 60)
