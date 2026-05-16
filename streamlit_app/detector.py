"""
Detection and tracking logic for the Drone Detection and Counting System.
Handles YOLO model inference, image detection, and video tracking.
Supports both standard whole-image inference and SAHI (Slicing Aided Hyper
Inference) for improved small-object recall in aerial imagery.
"""

import cv2
import time
import tempfile
import os
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

from config import CLASS_NAMES, CLASS_COLORS_BGR


# ── SAHI availability guard ───────────────────────────────────
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    _SAHI_AVAILABLE = True
except ImportError:
    _SAHI_AVAILABLE = False


def is_sahi_available():
    """Return True if the sahi package is installed."""
    return _SAHI_AVAILABLE


def load_model(model_path):
    """Load a YOLO model from the given path."""
    return YOLO(model_path)


def detect_image(model, image_bgr, conf, iou, imgsz):
    """
    Run object detection on a single image.

    Args:
        model: Loaded YOLO model.
        image_bgr: Input image in BGR format (numpy array).
        conf: Confidence threshold.
        iou: IoU threshold for NMS.
        imgsz: Input resolution for the model.

    Returns:
        annotated_rgb: Annotated image in RGB format.
        stats: Dict with human_count, car_count, total, detections list.
    """
    results = model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    result = results[0]

    human_count = 0
    car_count = 0
    detections = []
    annotated = result.orig_img.copy()

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))

        if cls_id == 0:
            human_count += 1
        elif cls_id == 1:
            car_count += 1

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background and text
        label = f"{cls_name} {conf_val:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 8),
                      (x1 + label_size[0] + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        detections.append({
            "class": cls_name,
            "confidence": conf_val,
            "bbox": [x1, y1, x2, y2],
        })

    # Draw count overlay on top-left corner
    _draw_count_overlay(annotated, human_count, car_count)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated_rgb, {
        "human_count": human_count,
        "car_count": car_count,
        "total": human_count + car_count,
        "detections": detections,
    }


def track_video(model, video_path, conf, iou, imgsz, tracker_type,
                on_progress=None):
    """
    Run multi-object tracking on a video file.

    Args:
        model: Loaded YOLO model (fresh instance for clean tracker state).
        video_path: Path to the input video file.
        conf: Confidence threshold.
        iou: IoU threshold for NMS.
        imgsz: Input resolution for the model.
        tracker_type: Tracker config name ('bytetrack.yaml' or 'botsort.yaml').
        on_progress: Optional callback(frame_idx, total_frames, elapsed, eta)
                     for progress reporting.

    Returns:
        output_path: Path to the annotated output video.
        stats: Dict with tracking statistics.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Prepare output video writer
    output_path = os.path.join(
        tempfile.gettempdir(),
        f"tracked_{tracker_type.replace('.yaml', '')}.mp4"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Run tracking
    unique_ids = defaultdict(set)
    frame_counts = []

    results = model.track(
        source=video_path,
        tracker=tracker_type,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        persist=True,
        stream=True,
        verbose=False,
    )

    start_time = time.time()

    for frame_idx, result in enumerate(results):
        boxes = result.boxes
        annotated = result.orig_img.copy()
        frame_humans = 0
        frame_cars = 0

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                track_id = int(box.id[0]) if box.id is not None else -1
                cls_name = CLASS_NAMES.get(cls_id, "Unknown")
                color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))

                if cls_id == 0:
                    frame_humans += 1
                    if track_id >= 0:
                        unique_ids[0].add(track_id)
                elif cls_id == 1:
                    frame_cars += 1
                    if track_id >= 0:
                        unique_ids[1].add(track_id)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Draw label with track ID and confidence
                if track_id >= 0:
                    label = f"{cls_name} #{track_id} {conf_val:.2f}"
                else:
                    label = f"{cls_name} {conf_val:.2f}"

                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 8),
                              (x1 + label_size[0] + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Draw tracking overlay with cumulative counts
        _draw_tracking_overlay(
            annotated, frame_idx + 1, total_frames,
            frame_humans, frame_cars,
            len(unique_ids.get(0, set())),
            len(unique_ids.get(1, set())),
        )

        writer.write(annotated)
        frame_counts.append({"human": frame_humans, "car": frame_cars})

        # Report progress
        if on_progress:
            elapsed = time.time() - start_time
            eta = (elapsed / (frame_idx + 1)) * (total_frames - frame_idx - 1)
            on_progress(frame_idx + 1, total_frames, elapsed, eta)

    writer.release()
    processing_time = time.time() - start_time

    return output_path, {
        "total_frames": total_frames,
        "fps": fps,
        "duration": total_frames / fps,
        "unique_humans": len(unique_ids.get(0, set())),
        "unique_cars": len(unique_ids.get(1, set())),
        "frame_counts": frame_counts,
        "processing_time": processing_time,
    }



# ── SAHI Inference ───────────────────────────────────────────

def detect_image_sahi(
    model_path, image_bgr, conf, iou, imgsz,
    slice_size=640, overlap_ratio=0.2,
):
    """
    Run SAHI sliced inference on a single image for improved small object recall.

    The image is divided into overlapping tiles of `slice_size` × `slice_size`
    pixels with `overlap_ratio` overlap on each axis. Each tile is run through
    the YOLO model at native resolution; results are merged back into the
    original coordinate space using SAHI's built-in NMS.

    Args:
        model_path: Filesystem path to the YOLO weights file (.pt).
        image_bgr: Input image in BGR format (numpy array).
        conf: Confidence threshold (0–1).
        iou: IoU threshold used for post-merge NMS (0–1).
        imgsz: Input resolution passed to the underlying model.
        slice_size: Height and width of each tile in pixels (default 640).
        overlap_ratio: Fractional overlap between adjacent tiles (default 0.2).

    Returns:
        annotated_rgb: Annotated image in RGB format.
        stats: Dict with human_count, car_count, total, detections list.

    Raises:
        ImportError: If the `sahi` package is not installed.
    """
    if not _SAHI_AVAILABLE:
        raise ImportError(
            "sahi is not installed. Run: pip install sahi"
        )

    # Save image to a temporary file — SAHI's get_sliced_prediction
    # accepts a file path or PIL image; a temp file avoids PIL dependency.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, image_bgr)

    try:
        # Build an AutoDetectionModel around the same weights
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",   # compatible with YOLOv11 via ultralytics
            model_path=model_path,
            confidence_threshold=conf,
            device="cuda:0" if _cuda_available() else "cpu",
        )

        result = get_sliced_prediction(
            tmp_path,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            postprocess_match_threshold=iou,
            verbose=0,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Annotate and count ─────────────────────────────────────
    human_count = 0
    car_count = 0
    detections = []
    annotated = image_bgr.copy()

    for pred in result.object_prediction_list:
        cls_id = pred.category.id
        conf_val = pred.score.value
        bbox = pred.bbox.to_xyxy()          # [x1, y1, x2, y2] floats
        x1, y1, x2, y2 = map(int, bbox)

        cls_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))

        if cls_id == 0:
            human_count += 1
        elif cls_id == 1:
            car_count += 1

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background and text
        label = f"{cls_name} {conf_val:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 8),
            (x1 + label_size[0] + 4, y1),
            color, -1,
        )
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        detections.append({
            "class": cls_name,
            "confidence": conf_val,
            "bbox": [x1, y1, x2, y2],
        })

    # Draw the same count overlay used by standard inference
    _draw_count_overlay(annotated, human_count, car_count)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated_rgb, {
        "human_count": human_count,
        "car_count": car_count,
        "total": human_count + car_count,
        "detections": detections,
    }


def _cuda_available():
    """Return True if a CUDA-capable GPU is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Private Helpers ───────────────────────────────────────────

def _draw_count_overlay(frame, human_count, car_count):
    """Draw a count summary overlay on the top-left corner of a frame."""
    y = 30
    for text in [f"Humans: {human_count}", f"Cars: {car_count}",
                 f"Total: {human_count + car_count}"]:
        cv2.rectangle(frame, (5, y - 20), (200, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 30


def _draw_tracking_overlay(frame, current_frame, total_frames,
                           frame_humans, frame_cars,
                           cumul_humans, cumul_cars):
    """Draw a tracking info overlay on the top-left corner of a frame."""
    lines = [
        f"Frame {current_frame}/{total_frames}",
        f"Humans (frame): {frame_humans}",
        f"Cars (frame): {frame_cars}",
        f"Unique Humans: {cumul_humans}",
        f"Unique Cars: {cumul_cars}",
    ]
    y = 25
    for line in lines:
        cv2.rectangle(frame, (5, y - 16), (280, y + 4), (0, 0, 0), -1)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 24
