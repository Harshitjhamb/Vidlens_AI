import streamlit as st
import os
import tempfile
import time
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Understanding AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Top header bar */
.top-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
}
.top-header h1 { color: #fff; margin: 0; font-size: 2rem; font-weight: 700; }
.top-header p  { color: #a0aec0; margin: 0; font-size: 0.97rem; }

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin: 0 0 1.5rem; }
.metric-card {
    flex: 1;
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .val { font-size: 1.5rem; font-weight: 700; color: #2d3748; }
.metric-card .lbl { font-size: 0.78rem; color: #718096; margin-top: 2px; }

/* Upload zone */
.upload-zone {
    border: 2px dashed #cbd5e0;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    background: #f7fafc;
    transition: border-color .2s;
}
.upload-zone:hover { border-color: #667eea; }

/* Mode cards */
.mode-card {
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 10px;
    background: #fff;
    cursor: pointer;
    transition: border-color .2s, box-shadow .2s;
}
.mode-card.active { border-color: #667eea; box-shadow: 0 0 0 3px rgba(102,126,234,.15); }
.mode-card h4 { margin: 0 0 4px; font-size: 0.95rem; color: #2d3748; }
.mode-card p  { margin: 0; font-size: 0.8rem; color: #718096; }

/* Pipeline steps */
.pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    background: #f7fafc;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    flex-wrap: wrap;
    gap: 6px;
}
.pip-step {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.8rem;
    font-weight: 500;
    color: #4a5568;
    white-space: nowrap;
}
.pip-arrow { color: #cbd5e0; font-size: 1.1rem; padding: 0 2px; }

/* Result box */
.result-box {
    background: #f0fff4;
    border: 1.5px solid #9ae6b4;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.result-box h3 { color: #276749; margin: 0 0 .75rem; font-size: 1rem; }
.result-box pre {
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 0.93rem;
    color: #2d3748;
    margin: 0;
    background: transparent;
    border: none;
    padding: 0;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-green  { background: #c6f6d5; color: #22543d; }
.badge-yellow { background: #fefcbf; color: #744210; }
.badge-blue   { background: #bee3f8; color: #2a4365; }

/* Model pill */
.model-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #edf2ff;
    color: #3c366b;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px 2px;
}

/* Sidebar info card */
.info-card {
    background: #ebf8ff;
    border-left: 4px solid #4299e1;
    border-radius: 0 8px 8px 0;
    padding: .75rem 1rem;
    font-size: 0.82rem;
    color: #2a4365;
    margin-bottom: .75rem;
}

/* Warn card */
.warn-card {
    background: #fffbeb;
    border-left: 4px solid #f6ad55;
    border-radius: 0 8px 8px 0;
    padding: .75rem 1rem;
    font-size: 0.82rem;
    color: #7b341e;
    margin-bottom: .75rem;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def try_import_analyzer():
    """Try to import the real analyze_video; return None if not installed."""
    try:
        from video_understanding.video_understanding import analyze_video
        return analyze_video
    except ImportError:
        return None


def fake_analyze(video_path: str, system_prompt: str | None, progress_cb) -> str:
    """
    Demo mode: simulate the pipeline with sleep steps and return a
    plausible mock result so the UI can be tested without the heavy models.
    """
    steps = [
        ("🔪 Chunking video into segments…",        0.15),
        ("🎙️ Whisper: transcribing audio…",         0.35),
        ("👁️ SmolVLM2: analysing visual frames…",   0.65),
        ("🧠 Qwen2.5: generating final summary…",   0.90),
        ("✅ Done!",                                  1.00),
    ]
    for msg, pct in steps:
        progress_cb(pct, msg)
        time.sleep(1.2)

    fname = Path(video_path).name
    if system_prompt and "filename" in system_prompt.lower():
        return "robot_arm_assembly_tutorial"
    if system_prompt and "topic" in system_prompt.lower():
        return (
            "Topics covered:\n"
            "• Industrial robotics and servo control\n"
            "• Assembly jig alignment\n"
            "• Torque calibration steps\n"
            "• Safety checks before power-on"
        )
    return (
        f"**Video:** {fname}\n\n"
        "**Summary (demo mode):**\n"
        "The video shows a technical tutorial on assembling a robot arm. "
        "The presenter walks through attaching the shoulder servo, aligning the elbow joint, "
        "and calibrating the gripper. Audio includes step-by-step verbal instructions. "
        "On-screen text labels each part as it is installed.\n\n"
        "*(Install `video_understanding` to run real AI analysis.)*"
    )


def run_analysis(video_path: str, system_prompt: str | None,
                 use_demo: bool, analyze_fn) -> tuple[str, float]:
    """Run analysis and return (result_text, elapsed_seconds)."""
    progress_bar  = st.progress(0.0)
    status_text   = st.empty()

    def update_progress(pct: float, msg: str):
        progress_bar.progress(pct)
        status_text.markdown(f"**{msg}**")

    t0 = time.time()
    if use_demo or analyze_fn is None:
        result = fake_analyze(video_path, system_prompt, update_progress)
    else:
        update_progress(0.05, "🔪 Chunking video…")
        result = analyze_fn(
            video_path,
            system_prompt=system_prompt if system_prompt else None,
        )
        update_progress(1.0, "✅ Done!")

    elapsed = time.time() - t0
    progress_bar.progress(1.0)
    status_text.empty()
    return result, elapsed


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/video.png",
        width=64,
    )
    st.markdown("## ⚙️ Settings")
    st.divider()

    # Demo / real mode toggle
    FORCE_DEMO = True  # Set to False when deploying on a machine with GPU + library installed

    if FORCE_DEMO:
        analyze_fn = None
        demo_mode = True
        st.markdown('<div class="badge badge-blue">Live demo mode</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="info-card">This hosted demo simulates the pipeline so anyone can try the UI '
            'without a GPU. Install <code>video_understanding</code> locally with a CUDA GPU to run real AI analysis.</div>',
            unsafe_allow_html=True,
        )
    else:
        analyze_fn = try_import_analyzer()
        lib_installed = analyze_fn is not None

        if lib_installed:
            st.markdown('<div class="badge badge-green">✔ video_understanding installed</div>',
                        unsafe_allow_html=True)
            demo_mode = st.toggle("Use demo mode (no GPU needed)", value=False)
        else:
            st.markdown('<div class="badge badge-yellow">⚠ Library not installed</div>',
                        unsafe_allow_html=True)
            demo_mode = True
            st.markdown('<div class="warn-card">Run <code>pip install video_understanding</code> then restart the app to enable real AI analysis.</div>',
                        unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🤖 Models used")
    for pill in ["Whisper Base · 140 MB", "SmolVLM2-2.2B · 9 GB", "Qwen2.5-7B · 14 GB"]:
        st.markdown(f'<span class="model-pill">📦 {pill}</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 💻 Hardware requirements")
    st.markdown("""
<div class="info-card">
<b>GPU:</b> CUDA-compatible NVIDIA<br>
<b>VRAM:</b> 8 GB minimum<br>
<b>Disk:</b> ~25 GB (models)<br>
<b>Python:</b> 3.10 +<br>
<b>FFmpeg:</b> must be installed
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔒 Privacy")
    st.markdown(
        "All processing happens **100% locally**. "
        "Your video never leaves your machine.",
    )
    st.caption("GitHub · Grigorij-Dudnik/video-understanding-local")


# ── Main Page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
  <div style="font-size:3rem">🎬</div>
  <div>
    <h1>Video Understanding AI</h1>
    <p>Analyse any video — audio + visuals — entirely on your machine. Fully private, fully offline.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Pipeline overview strip
st.markdown("""
<div class="pipeline">
  <span class="pip-step">📂 Upload video</span>
  <span class="pip-arrow">→</span>
  <span class="pip-step">🔪 Chunking</span>
  <span class="pip-arrow">→</span>
  <span class="pip-step">🎙️ Whisper ASR</span>
  <span class="pip-arrow">+</span>
  <span class="pip-step">👁️ SmolVLM2 vision</span>
  <span class="pip-arrow">→</span>
  <span class="pip-step">🧠 Qwen2.5 LLM</span>
  <span class="pip-arrow">→</span>
  <span class="pip-step">📝 Output</span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_analyse, tab_batch, tab_about = st.tabs(
    ["🔍 Analyse Video", "📁 Batch Rename", "📖 About & Docs"]
)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 – Single Video Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_analyse:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.subheader("📤 Upload your video")
        uploaded = st.file_uploader(
            "Drop a video file here",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            label_visibility="collapsed",
        )

        if uploaded:
            st.video(uploaded)
            size_mb = len(uploaded.getvalue()) / (1024 * 1024)
            st.caption(
                f"📄 **{uploaded.name}** · "
                f"{size_mb:.1f} MB · "
                f"{uploaded.type}"
            )

    with col_right:
        st.subheader("🎯 Analysis mode")

        # Preset mode picker
        mode = st.radio(
            "Choose what you want from the video",
            options=[
                "General summary",
                "Auto-generate filename",
                "List all topics covered",
                "Extract tools & materials",
                "Custom instruction",
            ],
            label_visibility="collapsed",
        )

        PRESET_PROMPTS = {
            "General summary": None,
            "Auto-generate filename": (
                "Analyse this video and generate a concise filename that describes "
                "the main action and subject. Use lowercase with underscores only. "
                "Return only the filename, no extension, no explanation."
            ),
            "List all topics covered": (
                "List all technical and conceptual topics discussed or demonstrated "
                "in this video. Use a numbered list."
            ),
            "Extract tools & materials": (
                "Describe all tools, materials, equipment, and software shown or "
                "mentioned in this video. Be specific."
            ),
            "Custom instruction": "__custom__",
        }

        system_prompt = PRESET_PROMPTS[mode]

        if mode == "Custom instruction":
            system_prompt = st.text_area(
                "Your custom instruction for the AI",
                placeholder=(
                    "e.g. Transcribe all dialogue spoken in this video.\n"
                    "e.g. Identify every person visible and what they are doing.\n"
                    "e.g. Summarise this lecture for a beginner."
                ),
                height=120,
            )
            if not system_prompt.strip():
                system_prompt = None
        elif system_prompt:
            with st.expander("View prompt sent to AI", expanded=False):
                st.code(system_prompt, language="text")

        st.divider()

        if demo_mode:
            st.markdown('<div class="badge badge-yellow">🟡 Demo mode – simulated output</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge badge-green">🟢 Real AI mode</div>',
                        unsafe_allow_html=True)

        run_btn = st.button(
            "▶️  Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )

    # ── Analysis execution ────────────────────────────────────────────────────
    if run_btn and uploaded:
        st.divider()
        st.subheader("⚙️ Processing…")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded.name).suffix
        ) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            result, elapsed = run_analysis(
                tmp_path, system_prompt, demo_mode, analyze_fn
            )
        finally:
            os.unlink(tmp_path)

        # ── Results ───────────────────────────────────────────────────────────
        st.success(f"Analysis complete in **{elapsed:.1f}s**")

        st.markdown(f"""
<div class="result-box">
  <h3>📝 Result</h3>
  <pre>{result}</pre>
</div>
""", unsafe_allow_html=True)

        # Copy / download helpers
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "⬇️ Download result (.txt)",
                data=result,
                file_name=f"analysis_{Path(uploaded.name).stem}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_b:
            st.code(result, language="text")

        # Stats strip
        st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="val">{elapsed:.1f}s</div>
    <div class="lbl">Processing time</div>
  </div>
  <div class="metric-card">
    <div class="val">{len(result.split())}</div>
    <div class="lbl">Words in output</div>
  </div>
  <div class="metric-card">
    <div class="val">{len(result)}</div>
    <div class="lbl">Characters</div>
  </div>
  <div class="metric-card">
    <div class="val">{'Demo' if demo_mode else 'Real AI'}</div>
    <div class="lbl">Mode used</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Save to session history
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "file": uploaded.name,
            "mode": mode,
            "result": result,
            "time": elapsed,
        })


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 – Batch Rename
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("📁 Batch Rename Raw Footage")
    st.markdown(
        "Upload multiple videos. The AI will generate a descriptive filename for each "
        "one based on its content. You can review suggestions before applying."
    )

    batch_files = st.file_uploader(
        "Upload multiple video files",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    batch_prompt = (
        "Analyse this video and generate a concise filename that describes "
        "the main action and subject. Use lowercase with underscores only. "
        "Return only the filename without extension and without any explanation."
    )

    if batch_files:
        st.info(f"{len(batch_files)} file(s) ready. Click **Run Batch** to process all.")
        run_batch = st.button("▶️  Run Batch Analysis", type="primary")

        if run_batch:
            results_table = []

            for i, vfile in enumerate(batch_files):
                st.markdown(f"**Processing {i+1}/{len(batch_files)}: {vfile.name}**")
                prog = st.progress(0.0)
                stat = st.empty()

                def cb(pct, msg, _prog=prog, _stat=stat):
                    _prog.progress(pct)
                    _stat.markdown(msg)

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(vfile.name).suffix
                ) as tmp:
                    tmp.write(vfile.getvalue())
                    tmp_path = tmp.name

                try:
                    if demo_mode or analyze_fn is None:
                        new_name = fake_analyze(tmp_path, batch_prompt, cb)
                    else:
                        cb(0.1, "Chunking…")
                        new_name = analyze_fn(tmp_path, system_prompt=batch_prompt)
                        cb(1.0, "Done")
                finally:
                    os.unlink(tmp_path)

                new_name = new_name.strip().replace(" ", "_").lower()
                ext = Path(vfile.name).suffix
                results_table.append({
                    "Original": vfile.name,
                    "Suggested name": f"{new_name}{ext}",
                    "Status": "✅ Ready",
                })
                prog.empty()
                stat.empty()

            st.success("Batch analysis complete!")
            st.subheader("📋 Rename suggestions")
            st.dataframe(results_table, use_container_width=True)

            import csv, io
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["Original", "Suggested name", "Status"])
            writer.writeheader()
            writer.writerows(results_table)
            st.download_button(
                "⬇️ Download rename table (.csv)",
                data=buf.getvalue(),
                file_name="rename_suggestions.csv",
                mime="text/csv",
            )
    else:
        st.markdown("""
<div class="upload-zone">
  <div style="font-size:2.5rem">📂</div>
  <p style="color:#718096;margin:.5rem 0 0">Drag & drop multiple video files here</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 – About
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.subheader("📖 About this app")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🔧 How it works")
        st.markdown("""
1. **Chunking** — Long video split into short time segments so models fit in GPU memory.
2. **Whisper ASR** — Each chunk's audio is transcribed to text (speech → words).
3. **SmolVLM2 Vision** — Frames are sampled and described visually per chunk.
4. **Context merge** — Audio transcript + visual description combined per chunk.
5. **Qwen2.5 LLM** — All chunk contexts fed to an LLM that produces the final output.
        """)

        st.markdown("### 📦 Models")
        st.markdown("""
| Model | Size | Role |
|---|---|---|
| Whisper Base | 140 MB | Speech-to-text |
| SmolVLM2-2.2B | ~9 GB | Visual understanding |
| Qwen2.5-7B | ~14 GB | Final summarisation |
        """)

    with c2:
        st.markdown("### 🚀 Quick start")
        st.code("""
# 1. Install the library
pip install video_understanding

# 2. Install FFmpeg (Windows)
winget install ffmpeg --version 7.1.1

# 3. Run this Streamlit app
streamlit run app.py
        """, language="bash")

        st.markdown("### 💡 Usage in Python")
        st.code("""
from video_understanding.video_understanding import analyze_video

# General summary
summary = analyze_video("video.mp4")

# Custom prompt
topics = analyze_video(
    "video.mp4",
    system_prompt="List all technical topics covered."
)
        """, language="python")

        st.markdown("### ⚙️ Hardware requirements")
        st.markdown("""
- **GPU:** CUDA-compatible NVIDIA
- **VRAM:** 8 GB minimum
- **Disk:** ~25 GB (for 3 models)
- **Python:** 3.10 +
- **FFmpeg:** required for video processing
        """)

    st.divider()
    st.markdown("""
<div style="text-align:center;color:#718096;font-size:.85rem;padding:1rem 0">
  Built on <b>video-understanding-local</b> by Grigorij Dudnik ·
  <a href="https://github.com/Grigorij-Dudnik/video-understanding-local" target="_blank">GitHub</a>
  · Whisper · SmolVLM2 · Qwen2.5 · All models run fully offline
</div>
""", unsafe_allow_html=True)

# ── Session history (sidebar footer) ─────────────────────────────────────────
if st.session_state.get("history"):
    with st.sidebar:
        st.divider()
        st.markdown("### 🕘 Session history")
        for item in reversed(st.session_state.history[-5:]):
            with st.expander(f"📄 {item['file'][:24]}…"):
                st.caption(f"Mode: {item['mode']} · {item['time']:.1f}s")
                st.text(item["result"][:300] + ("…" if len(item["result"]) > 300 else ""))