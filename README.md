# 🎬 AI Anime Scene Generator

An AI-powered pipeline that generates anime-style scenes from text prompts using Stable Diffusion and prompt engineering.

---

## 🚀 Features
- Prompt engineering system for cinematic anime scenes
- Modular pipeline for image generation
- Config-driven workflow
- Ready for animation + voice integration

---

## 🧠 Tech Stack
- Python
- Stable Diffusion (Local - AUTOMATIC1111 WebUI)
- Prompt Engineering
- YAML Configuration

---

## ⚙️ Project Structure
ai-anime-scene-generator/
│
├── scripts/
│ ├── 01_generate_image.py
│ ├── prompt_engine.py
│
├── prompts/
│ └── raw_prompts.txt
│
├── config.yaml
├── requirements.txt


---

## 🎯 Example Use Case

Input: lone samurai standing on a misty mountain peak at golden hour
Output:
- AI-generated anime-style scene (via Stable Diffusion)

---

## 🛠️ How It Works

1. User provides scene prompt  
2. Prompt is enhanced using prompt_engine  
3. Final prompt is passed to image generation model  
4. Output image is generated  

---

## 📌 Future Improvements
- Add animation pipeline
- Add voice narration
- Build UI (Gradio)
- Multi-scene video generation

---

## 💡 Note
Currently uses local Stable Diffusion setup instead of APIs due to cost and flexibility advantages.

---

## 👨‍💻 Author
Aniruddha
