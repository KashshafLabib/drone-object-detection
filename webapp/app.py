"""
Drone Human Detection and Counting System — Web Application

A Streamlit-based interface for detecting humans and cars in drone/aerial
imagery, counting objects, and tracking across video frames.

Usage:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

from config import (
    DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMGSZ, IMGSZ_OPTIONS,
    TRACKER_OPTIONS, CLASS_COLORS_HEX, find_model_path,
)
from detector import load_model, detect_image, detect_image_sahi, track_video, is_sahi_available
from styles import get_css
from components import (
    metric_card, detection_table, sidebar_section_title,
    section_divider, placeholder_message,
)


# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="Drone Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown(get_css(), unsafe_allow_html=True)


# ── Model Loading (cached) ──────────────────────────────────
@st.cache_resource
def get_model(path):
    return load_model(path)


# ── Sidebar ──────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar and return configuration values."""
    with st.sidebar:
        sidebar_section_title("MODEL")

        model_path = find_model_path()
        custom_path = st.text_input("Model weights path", value=model_path or "best.pt")

        if custom_path and os.path.exists(custom_path):
            model = get_model(custom_path)
            st.success(f"Loaded: {Path(custom_path).name}")
        else:
            st.error("Model not found. Place best.pt in the project root or specify the path.")
            st.stop()

        section_divider()
        sidebar_section_title("DETECTION SETTINGS")

        conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05)
        iou = st.slider("IoU threshold (NMS)", 0.1, 0.95, DEFAULT_IOU, 0.05)
        imgsz = st.select_slider("Input resolution", options=IMGSZ_OPTIONS, value=DEFAULT_IMGSZ)

        section_divider()
        sidebar_section_title("SAHI — SLICED INFERENCE")

        sahi_ok = is_sahi_available()
        if not sahi_ok:
            st.caption(
                "⚠️ `sahi` not installed. "
                "Run `pip install sahi` to enable sliced inference."
            )
        use_sahi = st.toggle(
            "Enable SAHI (better small-object recall)",
            value=False,
            disabled=not sahi_ok,
            help=(
                "Divides the image into overlapping tiles and runs detection "
                "on each tile at native resolution. Significantly improves recall "
                "for tiny objects (pedestrians < 32 px) at the cost of speed."
            ),
        )
        sahi_slice = st.select_slider(
            "Tile size (px)",
            options=[320, 480, 512, 640, 800],
            value=640,
            disabled=not (sahi_ok and use_sahi),
            help="Width and height of each tile. Smaller tiles = more tiles = slower but finer detail.",
        )
        sahi_overlap = st.slider(
            "Tile overlap ratio",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            disabled=not (sahi_ok and use_sahi),
            help="Fractional overlap between adjacent tiles (0.2 = 20%). Higher overlap reduces missed detections at tile boundaries.",
        )

        section_divider()
        sidebar_section_title("TRACKING")

        tracker = st.selectbox(
            "Tracker", TRACKER_OPTIONS,
            format_func=lambda x: x.replace(".yaml", "").upper(),
        )

        section_divider()
        sidebar_section_title("ABOUT")
        st.markdown("""
        **Architecture**: YOLOv11m  
        **Classes**: Human, Car  
        **Dataset**: VisDrone 2019 DET  
        **Training**: 1280px, AdamW, 86 epochs
        """)

    return model, custom_path, conf, iou, imgsz, tracker, use_sahi, sahi_slice, sahi_overlap


# ── Image Detection Page ───────────────────────────────────────────
def render_image_tab(model, model_path, conf, iou, imgsz,
                    use_sahi, sahi_slice, sahi_overlap):
    """Render the image detection tab."""
    uploaded = st.file_uploader(
        "Upload a drone/aerial image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload",
    )

    if uploaded is None:
        placeholder_message(
            "Upload a drone or aerial image to begin detection."
        )
        return

    # Read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = image_bgr.shape[:2]

    # Run detection
    if use_sahi:
        mode_label = f"SAHI ({sahi_slice}px tiles, {int(sahi_overlap * 100)}% overlap)"
        spinner_msg = f"Running sliced inference ({mode_label})..."
    else:
        mode_label = "Standard"
        spinner_msg = "Running detection..."

    with st.spinner(spinner_msg):
        if use_sahi:
            annotated_rgb, stats = detect_image_sahi(
                model_path, image_bgr, conf, iou, imgsz,
                slice_size=sahi_slice,
                overlap_ratio=sahi_overlap,
            )
        else:
            annotated_rgb, stats = detect_image(model, image_bgr, conf, iou, imgsz)

    # Mode badge
    badge_color = "#4A90D9" if use_sahi else "#8892A0"
    st.markdown(
        f'<span style="background:{badge_color};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:600;">{mode_label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(stats["human_count"], "Humans", "human")
    with c2:
        metric_card(stats["car_count"], "Cars", "car")
    with c3:
        metric_card(stats["total"], "Total Objects", "total")
    with c4:
        metric_card(f"{w}x{h}", "Resolution")

    st.markdown("")

    # Side-by-side images
    col_orig, col_det = st.columns(2)
    with col_orig:
        st.markdown("**Original**")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_det:
        st.markdown(f"**Detected** ({mode_label})")
        st.image(annotated_rgb, use_container_width=True)

    # Detection details
    if stats["detections"]:
        with st.expander(f"Detection Details ({len(stats['detections'])} objects)"):
            detection_table(stats["detections"])


# ── Video Tracking Page ──────────────────────────────────────
def render_video_tab(model_path, conf, iou, imgsz, tracker):
    """Render the video tracking tab."""
    uploaded = st.file_uploader(
        "Upload a drone/aerial video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded is None:
        placeholder_message(
            "Upload a drone or aerial video to begin tracking. "
            "The system will assign unique IDs to each detected object across frames."
        )
        return

    # Save uploaded video to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded.read())
    temp_path = temp_file.name
    temp_file.close()

    # Read video metadata
    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_duration = total_frames / vid_fps
    cap.release()

    # Video info metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(total_frames, "Frames")
    with c2:
        metric_card(f"{vid_duration:.1f}s", "Duration")
    with c3:
        metric_card(f"{vid_w}x{vid_h}", "Resolution")
    with c4:
        metric_card(f"{vid_fps:.0f}", "FPS")

    st.markdown("")

    # Run button and time estimate
    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("Run Tracking", type="primary", use_container_width=True)
    with col_info:
        est_time = total_frames * 0.8
        st.markdown(
            f'<div style="color:#8892A0; padding:0.5rem;">'
            f'Estimated: ~{est_time:.0f}s on CPU ({total_frames} frames). '
            f'Use 640px for faster processing.</div>',
            unsafe_allow_html=True,
        )

    if run_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Fresh model instance for clean tracker state
        from detector import load_model as _load
        tracking_model = _load(model_path)

        def on_progress(frame, total, elapsed, eta):
            progress_bar.progress(frame / total)
            status_text.text(
                f"Processing frame {frame}/{total} "
                f"| Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
            )

        output_path, stats = track_video(
            tracking_model, temp_path, conf, iou, imgsz, tracker,
            on_progress=on_progress,
        )

        progress_bar.progress(1.0)
        status_text.text(
            f"Complete: {stats['total_frames']} frames in {stats['processing_time']:.1f}s"
        )

        # Results metrics
        section_divider()
        st.markdown("**Tracking Results**")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card(stats["unique_humans"], "Unique Humans", "human")
        with c2:
            metric_card(stats["unique_cars"], "Unique Cars", "car")
        with c3:
            fps_actual = stats["total_frames"] / stats["processing_time"]
            metric_card(f"{fps_actual:.1f}", "Processing FPS")
        with c4:
            metric_card(f"{stats['processing_time']:.1f}s", "Total Time")

        st.markdown("")

        # Display tracked video
        if os.path.exists(output_path):
            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Tracked Video",
                    data=f,
                    file_name=f"tracked_{tracker.replace('.yaml', '')}.mp4",
                    mime="video/mp4",
                )

        # Per-frame chart
        with st.expander("Per-Frame Detection Counts"):
            import pandas as pd
            df = pd.DataFrame(stats["frame_counts"])
            df.index.name = "Frame"
            st.line_chart(df, color=[CLASS_COLORS_HEX[0], CLASS_COLORS_HEX[1]])

    # Clean up
    try:
        os.unlink(temp_path)
    except Exception:
        pass


# ── Main ────────────────────────────────────────────────────
def main():
    model, model_path, conf, iou, imgsz, tracker, use_sahi, sahi_slice, sahi_overlap = render_sidebar()

    # Header
    st.markdown(
        '<div class="main-header">Drone Detection and Counting System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">'
        'Detect humans and cars in aerial drone imagery. Upload an image for '
        'instant detection and counting, or upload a video for multi-object '
        'tracking with unique ID assignment.'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_image, tab_video = st.tabs(["IMAGE DETECTION", "VIDEO TRACKING"])

    with tab_image:
        render_image_tab(model, model_path, conf, iou, imgsz,
                         use_sahi, sahi_slice, sahi_overlap)

    with tab_video:
        render_video_tab(model_path, conf, iou, imgsz, tracker)


if __name__ == "__main__":
    main()
