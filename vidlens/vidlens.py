import os
import sys
import subprocess
import torch
import gc
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
import whisper
import shutil
CHUNK_BASE_DURATION = 60   # seconds
OVERLAP = 5          # seconds
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def _check_ffmpeg_installed():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print(
            "ERROR: ffmpeg and ffprobe must be installed and available in PATH.\n"
            "Please install FFmpeg:\n"
            "- Windows: winget install ffmpeg --version 7.1.1\n"
            "- macOS: brew install ffmpeg\n"
            "- Linux: sudo apt install ffmpeg\n",
            file=sys.stderr
        )
        raise RuntimeError("FFmpeg not found. Please install it to use video processing.")
    
def split_video_into_chunks(video_path):
    """
    Splits video into overlapping chunks using ffmpeg.
    Returns list of chunk file paths and their durations.
    """
    # Get total duration in seconds via ffprobe
    _check_ffmpeg_installed()
    total_duration = subprocess.run(
        ["ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    total_duration = float(total_duration.stdout.strip())
    print(f"Total video duration: {total_duration:.2f} seconds")

    chunks = []
    start = 0.0
    chunk_index = 0

    while start < total_duration:
        end = min(start + CHUNK_BASE_DURATION + OVERLAP, total_duration)
        os.makedirs("chunks", exist_ok=True)
        chunk_path = f"chunks/chunk_{chunk_index}.mp4"
        duration = end - start

        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
            "-t", str(duration), "-c", "copy",
            chunk_path
        ]
        print(f"Creating chunk {chunk_index}: {start:.1f}s–{end:.1f}s → {chunk_path}")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        chunks.append(chunk_path)
        if end >= total_duration:
            break
        start += CHUNK_BASE_DURATION
        chunk_index += 1

    return chunks

def understand_video_chunks(chunk_paths):
    print("\n[STAGE 2] Analyzing video chunks...")
    # use_fast=false to prevent laczos warning
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        dtype=torch.bfloat16,
        device_map=device
    )

    results = []
    for i, chunk_path in enumerate(chunk_paths):
        print(f"  Analyzing {chunk_path} ({i+1}/{len(chunk_paths)})")

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "path": chunk_path},
                {"type": "text", "text": f"Describe what is happening in this video segment in detail."}
            ]
        }]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        desc = processor.batch_decode(output, skip_special_tokens=True)[0]
        desc = desc.split("\nAssistant: ", 1)[1]
        results.append(desc)

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    return results

def transcribe_audio(chunk_paths):
    print("\n[STAGE 1] Transcribing audio...")
    asr_model = whisper.load_model("base").to(device)
    chunk_transcripts = []
    try:
        for chunk_path in chunk_paths:
            chunk_transcript = asr_model.transcribe(
                chunk_path,
                fp16=(device == "cuda"),
                language="en"
            )
            chunk_transcripts.append(chunk_transcript["text"].strip())
    # if track has no audio
    except RuntimeError:
        return [None]*len(chunk_paths)
    # cleanup gpu memory
    del asr_model
    torch.cuda.empty_cache()
    gc.collect()

    return chunk_transcripts

def generate_final_summary(video_results, transcript_chunks, system_prompt=None):
    print("\n[STAGE 3] Generating final narrative...")

    if not system_prompt:
        system_prompt = (
"""
Analyze provided video explained in chunks of video and audio elements and write what happens in that video.
"""
    )
    #Write a detailed paragraph describing what happens in the entire video.
    prompt = "Below are pairs of video chunk descriptions (image only) and corresponding audio transcripts. Chunks have small overlap.\n\n<video chunks descriptions>\n\n"
    for video_description, transcript in zip(video_results, transcript_chunks):
        prompt += "---next chunk---\n\n"
        prompt += f"Visual part description of chunk:\n{video_description}\n\n"
        prompt += f"Chunk audio:\n{transcript}\n\n"

    prompt += "</video chunks descriptions>"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        dtype=torch.bfloat16,
        device_map=device
    )

    messages = [{
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": prompt
    }]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    summary = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    # free gpu memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return summary

def cleanup_chunks(chunk_paths):
    for path in chunk_paths:
        if os.path.exists(path):
            os.remove(path)

def analyze_video(video_path, system_prompt=None):
    chunk_paths = split_video_into_chunks(video_path)
    print(f"Split into {len(chunk_paths)} video chunks.")

    try:
        # Step 1: Transcribe audio
        #audio_transcript = transcribe_audio_full()
        audio_transcripts = transcribe_audio(chunk_paths)
        # Step 2: Analyze each chunk as video
        video_descrpts = understand_video_chunks(chunk_paths)

        # Step 3: Generate final summary
        final_summary = generate_final_summary(video_descrpts, audio_transcripts, system_prompt)
        final_summary = final_summary.rsplit("\nassistant\n", 1)[-1]


    finally:
        # Always clean up temporary chunks
        cleanup_chunks(chunk_paths)

    return final_summary
