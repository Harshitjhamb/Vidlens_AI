# 🎬 Video Understanding Local

![Video Understanding](images/main_image.png)

AI-powered video analysis that combines audio transcription and visual understanding - running entirely on your machine!

Perfect for organizing raw footage, preparing video montages, or extracting insights from your video content - all while keeping your data completely private on your own machine.

## ✨ Key Features

- **👁️ Visual Scene Analysis**: Frame-by-frame understanding with SmolVLM2
- **🎙️ Audio Undrestanding**: Speech-to-text using Whisper model
- **🔒 Fully Offline**: All models run locally - under your entire control

## 🚀 Installation

Simply install from PyPI:

```bash
pip install video_understanding
```

## 💡 Quick Start

```python
from video_understanding.video_understanding import analyze_video

# Analyze any video file
summary = analyze_video("path/to/video.mp4")
print(summary)
```

That's it! The AI will watch your video and tell you what's happening.

## 📸 Example: Auto-Organizing Raw Footage

Here's a practical example that can save you hours - automatically renaming raw video files based on their content:

```python
import os
from video_understanding.video_understanding import analyze_video

# Define what you want the AI to focus on
system_prompt = """
Analyze this video and generate a concise filename that describes 
the main action and subject. Use lowercase with underscores.
Focus on key visual elements and dialogue.
"""

# Process videos in your folder
video_folder = "raw_footage"

for video_file in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_file)
    print(f"🎬 Processing: {video_file}")
    
    # Get AI-generated descriptive name
    new_name = analyze_video(video_path, system_prompt)
    ext = video_file.split(".")[-1]
    new_filename = f"{new_name}.{ext}"
    
    # Rename file
    os.rename(video_path, os.path.join(video_folder, new_filename))
    print(f"✅ Renamed to: {new_filename}\n")
```

This transforms generic filenames like "VIDEO_001.mp4" into descriptive ones like "team_assembling_robot_arm.mp4", making your footage instantly searchable and organized! 🎉

## 📝 Description

This package analyzes videos using AI models to understand both audio and visual content. It intelligently splits long videos into manageable chunks, transcribes speech with Whisper, analyzes visual scenes with SmolVLM2, and generates comprehensive summaries using Qwen2.5. 

## 💻 Requirements

### Hardware
- **GPU**: CUDA-compatible GPU strongly recommended (NVIDIA)
- **VRAM**: Minimum 8GB for smooth operation
- **Disk Space**: ~25GB for model storage

### Software
- Python 3.10+
- CUDA toolkit
- FFmpeg (if on Windows, install it with `winget install ffmpeg --version 7.1.1`)

### Models (Downloaded Automatically)
On first run, these models will be downloaded:
- 🎙️ Whisper Base (~140MB) - audio transcription
- 👁️ SmolVLM2-2.2B-Instruct (~9GB) - visual understanding
- 🧠 Qwen2.5-7B-Instruct (~14GB) - final summary generation

## More examples
```python
# General analysis
summary = analyze_video("video.mp4")

# Custom analysis with specific instructions
custom_prompt = "List all technical topics discussed in this video."
topics = analyze_video("video.mp4", system_prompt=custom_prompt)

# Extract specific information
prompt = "Describe what tools and materials are being used in this tutorial."
tools_list = analyze_video("tutorial.mp4", system_prompt=prompt)
```

# VidLens_AI
