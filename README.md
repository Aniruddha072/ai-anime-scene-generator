# 🎌 AI Anime Scene Generator

> Text → Image → Animation → Voice → Cinematic Video

![Demo](assets/samples/demo.gif)

---

## 🚀 What this does

This project converts a simple text prompt into a narrated cinematic anime-style video clip — fully locally.

```bash
python scripts/run_pipeline.py --prompt "A lone samurai at dawn"
```

---

## 🎬 Pipeline

Prompt
→ Image (Stable Diffusion)
→ Animation (OpenCV engine)
→ Voice (Bark TTS)
→ Final Video (MoviePy)

---

## 🧠 Key Features

* Custom animation engine (parallax, fog, particles)
* No paid APIs — runs locally
* Modular pipeline (can skip stages)
* Cinematic output with narration

---

## 🎥 Example Scenes

### 🌄 Scene 1 — Samurai at Dawn

![scene1](assets/samples/scene1.png)

### 🌃 Scene 2 — Cyberpunk Night

![scene2](assets/samples/scene2.png)

### 🌅 Scene 3 — Peaceful Village

![scene3](assets/samples/scene3.png)

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --prompt "your prompt"
```

---

## 🧩 Architecture

The system is designed as a modular pipeline:

* Image generation (diffusers)
* Animation engine (OpenCV, no ML)
* Voice synthesis (Bark)
* Video assembly (MoviePy)

---

## 🎯 Why this project is interesting

Most projects stop at image generation.

This project builds a full pipeline that produces a narrated cinematic video, including a custom animation system implemented without ML models.
