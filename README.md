# Drone Human Detection and Counting System

[![W&B](https://img.shields.io/badge/Weights_&_Biases-dashboard-yellow?logo=weightsandbiases)](https://wandb.ai/labibkashshaf-islamic-university-of-technology/drone-object-detection)

An end-to-end computer vision pipeline for detecting humans and cars in drone/aerial imagery, counting total humans per frame, and visualizing results. Built on the VisDrone 2019 dataset using YOLOv11 with architectural modifications for small object detection.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Task 01: Dataset Understanding and Preprocessing](#task-01-dataset-understanding-and-preprocessing)
- [Task 02: Model Training](#task-02-model-training)
- [Task 03: Detection and Counting](#task-03-detection-and-counting)
- [Task 04: Object Tracking (Bonus)](#task-04-object-tracking-bonus)
- [Task 05: Evaluation and Visualization](#task-05-evaluation-and-visualization)
- [Project Structure](#project-structure)
- [Setup and Reproduction](#setup-and-reproduction)
- [Strengths, Limitations, and Challenges](#strengths-limitations-and-challenges)
- [References](#references)

---

## Project Overview

This project implements a detection and counting pipeline for drone-captured aerial scenes. The system:

- Detects humans (pedestrians and people) and cars from aerial/drone viewpoints.
- Counts total humans per image.
- Visualizes detections with color-coded bounding boxes and count overlays.
- Applies SAHI (Slicing Aided Hyper Inference) for improved small object detection.
- Tracks objects across video frames using ByteTrack / BotSORT with unique ID assignment.
- Introduces a custom P2 detection head architecture for tiny object recall.

**Model**: YOLOv11m (Ultralytics)  
**Dataset**: VisDrone 2019 DET  
**Training Environment**: Kaggle (Tesla T4 GPU, 15 GB VRAM)

---

## Dataset

**Source**: [VisDrone 2019 DET on Kaggle](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset?resource=download)

The VisDrone 2019 dataset is a large-scale benchmark for object detection in drone-captured images, collected by the AISKYEYE team at Tianjin University. The Kaggle version used in this project has been pre-converted to YOLO annotation format.

| Split | Images | Annotations | Notes |
|-------|--------|-------------|-------|
| Train | 6,471 | 343,205 | Primary training data |
| Validation | 548 | 38,759 | Used for per-epoch evaluation |
| Test-dev | 1,610 | 75,102 | Held-out evaluation set |
| Test-challenge | 1,580 | -- | No labels (challenge set, unused) |
| **Total** | **10,209** | **457,066** | |

The dataset contains 10 object classes. For this project, only 3 are relevant to the task:

| Class ID | Class Name | Relevance |
|----------|------------|-----------|
| 0 | Pedestrian | Target (Human) |
| 1 | People | Target (Human) |
| 3 | Car | Target (Car) |
| 2, 4-9 | Bicycle, Van, Truck, Tricycle, Awning-tricycle, Bus, Motor | Not used |

---

## Task 01: Dataset Understanding and Preprocessing

### Exploratory Data Analysis

A thorough EDA was conducted across 24 analysis cells covering dataset structure, class distributions, bounding box statistics, spatial distributions, annotation quality, and visual inspection. The full EDA notebook is available at `notebooks/01_eda.ipynb`, with a structured summary at `notebooks/eda_results.md`.

Key findings from the analysis:

#### Image Resolutions

The dataset contains images at 11 distinct resolutions ranging from 480×360 to 2000×1500 pixels.

| Split | Unique Resolutions | Width Range | Height Range | MP Range |
|-------|--------------------|-------------|--------------|----------|
| Train | 11 | 480–2000 | 360–1500 | 0.17–3.00 |
| Val | 3 | 960–1920 | 540–1080 | 0.52–2.07 |
| Test | 6 | 960–1920 | 540–1080 | 0.52–2.07 |

Top resolutions: 1400×1050 (2,772 images), 1400×788 (2,232), 1360×765 (1,318), 2000×1500 (772). All splits have 100% image-label pairing with zero missing files.

#### Class Distribution

| Class | ID | Total | Train % | Val % | Test % | Target |
|-------|----|-------|---------|-------|--------|--------|
| Car | 3 | 187,005 | 42.21% | 36.29% | 37.38% | ✅ |
| Pedestrian | 0 | 109,187 | 23.12% | 22.82% | 27.97% | ✅ |
| Motor | 9 | 40,378 | 8.64% | 12.61% | 7.78% | ❌ |
| People | 1 | 38,560 | 7.88% | 13.22% | 8.49% | ✅ |
| Van | 4 | 32,702 | 7.27% | 5.10% | 7.68% | ❌ |
| Truck | 5 | 16,284 | 3.75% | 1.94% | 3.54% | ❌ |
| Bicycle | 2 | 13,069 | 3.05% | 3.32% | 1.73% | ❌ |
| Bus | 8 | 9,117 | 1.73% | 0.65% | 3.91% | ❌ |
| Tricycle | 6 | 6,387 | 1.40% | 2.70% | 0.71% | ❌ |
| Awning-tri | 7 | 4,377 | 0.95% | 1.37% | 0.80% | ❌ |

Target classes (Pedestrian, People, Car) account for 73.2% of all training annotations. The Pedestrian:People ratio is approximately 3:1 across all splits.

#### Small Object Problem (Critical Finding)

This is the dominant challenge in the dataset. The vast majority of human annotations are extremely small by standard detection benchmarks:

| Class | Small (<32×32 px) | Medium (32–96 px) | Large (>96 px) |
|-------|-------------------|--------------------|----------------|
| Pedestrian | 82.2% | 17.4% | 0.4% |
| People | 86.9% | 12.7% | 0.4% |
| Car | 48.2% | 43.5% | 8.2% |

Pixel dimension stats (train):

| Class | W Mean | W Median | H Mean | H Median | Area Median |
|-------|--------|----------|--------|----------|-------------|
| Pedestrian | 16.2 px | 13.0 px | 30.7 px | 25.0 px | 315 px² |
| People | 16.4 px | 13.0 px | 24.7 px | 20.0 px | 273 px² |
| Car | 51.3 px | 38.0 px | 40.6 px | 29.0 px | 1,104 px² |

Tiny object breakdown for pedestrians: 27.0% have width < 8 px, 62.5% < 16 px, 91.0% < 32 px. 4.2% have both dimensions < 8 px, making them effectively undetectable.

#### Object Density

Images contain up to 902 annotated objects, with 703 training images exceeding 100 objects. Mean density is 53 objects per image in the training split.

| Split | Min | Max | Mean | Median | >100 objects | >200 objects |
|-------|-----|-----|------|--------|--------------|--------------|
| Train | 1 | 902 | 53.0 | 42 | 703 | 80 |
| Val | 1 | 317 | 70.7 | 65 | 107 | 8 |
| Test | 1 | 461 | 46.6 | 36 | 127 | 29 |

Human-car co-occurrence: 82.7% of training images contain both humans and cars, 5.2% contain humans only, 12.1% cars only.

#### Aspect Ratios

| Class | Mean W/H | Median W/H |
|-------|----------|------------|
| Pedestrian | 0.37 | 0.31 |
| People | 0.45 | 0.40 |
| Car | 2.92 | 0.83 |

Pedestrians and people are consistently tall and narrow (W/H < 1). Cars have high variance due to diverse viewing angles.

#### Same-Class IoU Overlap

| Class | Overlapping Pairs | Mean IoU | >0.3 IoU | >0.5 IoU |
|-------|-------------------|----------|----------|----------|
| Pedestrian | 753 | 0.143 | 13.3% | 2.5% |
| People | 121 | 0.221 | 34.7% | 11.6% |
| Car | 1,831 | 0.131 | 10.5% | 2.1% |

The People class has the highest overlap (34.7% pairs >0.3 IoU), indicating crowded groups that will challenge NMS.

#### Spatial Distribution

Objects are relatively uniformly distributed across image regions with a slight concentration toward the center, consistent with drone cameras pointing at areas of interest.

#### Correlations

Image resolution has negligible correlation with object count (megapixels vs total objects: 0.082, vs humans: -0.007, vs cars: 0.098). Resolution does not predict density.

#### Annotation Quality

| Check | Result |
|-------|--------|
| Zero/negative dimensions | 1 box |
| Out of bounds | 0 |
| Covers >50% image | 0 |
| Exact duplicates | 4 |
| Micro boxes (area < 1e-6) | 1 |
| Unknown class IDs | 0 |

Only 6 problematic annotations out of 457,066 (0.001%). The dataset is clean and ready for training without corrections.

### Preprocessing Pipeline

The preprocessing stage transforms the 10-class VisDrone dataset into a focused 2-class detection dataset:

**Step 1: Class Remapping**

Pedestrian (class 0) and People (class 1) are merged into a single Human class (new class 0). Car (class 3) is remapped to new class 1. All other 7 classes are discarded.

Rationale for merging Pedestrian and People:
- Near-identical pixel dimensions (median 13×25 vs 13×20 px).
- Same aspect ratio profile (tall and narrow, W/H < 1).
- 3:1 count ratio between the subclasses creates unnecessary imbalance.
- Semantically identical for the counting task.

**Step 2: Dataset Creation**

A new filtered dataset is written to the working directory with the remapped labels. Images are symlinked to avoid duplicating storage. A `data.yaml` configuration file is generated for Ultralytics training.

**Step 3: Split Preservation**

The original VisDrone train/val/test splits are preserved without re-splitting. The dataset authors curated these splits to ensure images from the same drone flight sequence remain in the same split. Re-splitting would risk data leakage from temporally adjacent frames.

After filtering, the processed dataset contains:

| Split | Images | Human Annotations | Car Annotations | Total |
|-------|--------|-------------------|-----------------|-------|
| Train | 6,471 | 106,396 | 144,867 | 251,263 |
| Val | 548 | 13,969 | 14,064 | 28,033 |
| Test | 1,610 | 27,382 | 28,074 | 55,456 |

The resulting class ratio is approximately 1:1.4 (Human:Car), which is a moderate imbalance that does not require special handling.

---

## Task 02: Model Training

### Model Selection

**YOLOv11m** (medium variant) from Ultralytics was selected for the following reasons:

- YOLOv11 is the latest in the YOLO family with improved C3k2 blocks and C2PSA (Cross Stage Partial with Self-Attention) in the backbone, providing better feature representation than YOLOv8.
- The medium variant (20.1M parameters) offers the best balance between detection accuracy and GPU memory constraints on Kaggle T4 hardware.
- The nano/small variants lack capacity for small object detection in aerial imagery. The large/xlarge variants exceed memory at the required high input resolutions.

### Training Strategy

Training followed a three-phase iterative approach:

#### Phase 1: Baseline (640 px)

Establishes a performance floor with standard settings.

| Parameter | Value |
|-----------|-------|
| Model | YOLOv11m (pretrained on COCO) |
| Input resolution | 640 × 640 |
| Epochs | 50 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.001 (cosine decay to 0.01×) |
| Augmentation | Mosaic, MixUp (0.15), HSV jitter, horizontal flip, scale (0.5), rotation (5°) |

#### Phase 2: Optimized (1280 px)

Addresses the small object problem by doubling input resolution. At 1280 px, a 13 px pedestrian occupies approximately 12 px after resize, compared to approximately 6 px at 640 px.

| Parameter | Value |
|-----------|-------|
| Model | YOLOv11m (pretrained on COCO) |
| Input resolution | 1280 × 1280 |
| Epochs | 100 (early stopped at 86) |
| Batch size | 4 (reduced for GPU memory) |
| Optimizer | AdamW |
| Learning rate | 0.0005 (reduced for stability at high resolution) |
| Patience | 20 epochs (early stopping) |
| Augmentation | Same as baseline |

#### Phase 3: P2 Detection Head (1280 px)

A custom architectural modification adding a 4th detection head at stride 4. Standard YOLOv11 detects at strides 8, 16, and 32. The P2 head adds detection at stride 4, producing a 320×320 feature map at 1280 px input where each cell covers 4×4 pixels of the original image.

This approach is inspired by the TPH-YOLOv5 paper (Zhu et al., 2021), which demonstrated that adding a P2 prediction head improves mAP by approximately 2% on drone-captured scenarios.

A custom YAML architecture file (`yolo11m-p2.yaml`) was created by extending the standard YOLOv11 neck with:
- An additional upsample and concatenation to fuse backbone P2 features.
- A C3k2 processing block at P2 resolution.
- An additional bottom-up path from P2 back to P3.
- The Detect layer updated to output across 4 scales instead of 3.

Pretrained YOLOv11m backbone weights are transferred. The new P2 head layers are initialized randomly and trained from scratch.

| Parameter | Value |
|-----------|-------|
| Model | YOLOv11m-P2 (custom architecture) |
| Input resolution | 1280 × 1280 |
| Epochs | 100 |
| Batch size | 2 (further reduced for P2 memory overhead) |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Pretrained | Backbone from yolo11m.pt, P2 head from scratch |

### Training Results

#### Validation Set Metrics (during training)

| Metric | Baseline (640) | Optimized (1280) | Delta |
|--------|---------------|------------------|-------|
| mAP@0.5 | 0.6922 | 0.8265 | +0.1343 |
| mAP@0.5:0.95 | 0.4077 | 0.5136 | +0.1059 |
| Precision | 0.7746 | 0.8337 | +0.0591 |
| Recall | 0.6383 | 0.7708 | +0.1325 |

The optimized model at 1280 px achieved substantial improvements across all metrics. The largest gain was in recall (+13.3 percentage points), directly addressing the small object miss rate identified in the EDA.

Key observations:
- Training and validation losses tracked closely throughout, indicating no overfitting.
- The baseline model was still improving at epoch 50, confirming room for additional training.
- The optimized model converged and early-stopped at epoch 86 out of 100.

#### Test Set Metrics (held-out evaluation)

Evaluated on the test-dev split (1,610 images) using the optimized (1280 px) model:

| Metric | Overall | Human | Car |
|--------|---------|-------|-----|
| AP@0.5 | 0.6197 | 0.4531 | 0.7864 |
| AP@0.5:0.95 | 0.3632 | 0.1985 | 0.5279 |

The gap between validation (0.8265 mAP@0.5) and test (0.6197 mAP@0.5) reflects the greater diversity and difficulty of the test-dev split. The Human class AP remains the bottleneck due to the extreme prevalence of sub-32 px objects.

---

## Task 03: Detection and Counting

### Detection Pipeline

The detection system processes input images through the trained YOLOv11m model and produces:

- Color-coded bounding boxes: red for humans, green for cars.
- Per-box confidence scores.
- A count overlay displaying total humans, total cars, and total objects.

The inference function accepts configurable confidence threshold (default 0.25) and NMS IoU threshold (default 0.45).

### Counting Logic

Human counting is implemented as a direct summation of all detection boxes classified as Human (class 0) with confidence above the threshold. This is a frame-level count — each detected bounding box increments the counter by one.

While simple, this approach is appropriate for single-image analysis. Limitations include potential double-counting of partially occluded individuals and missed counts for humans below the confidence threshold.

### SAHI Integration

To address the small object detection gap, SAHI (Slicing Aided Hyper Inference) is integrated as an alternative inference mode available via the web application's sidebar toggle. SAHI divides the input image into overlapping tiles (configurable size, default 640×640 px with 20% overlap), runs detection independently on each tile at native resolution, then merges all detections back into the original coordinate space using NMS.

This avoids downscaling the full image and preserves fine spatial detail for tiny objects. A 13 px pedestrian remains 13 px within its tile rather than being compressed further during whole-image resize.

The webapp exposes three SAHI controls:
- **Enable/disable toggle** — switches between standard and sliced inference.
- **Tile size** — configurable from 320 to 800 px.
- **Overlap ratio** — 10% to 40% overlap between adjacent tiles.

---

## Task 04: Object Tracking (Bonus)

### Tracking Implementation

Multi-object tracking is implemented in the webapp's Video Tracking tab using Ultralytics' built-in tracker integration. The system supports two tracking algorithms:

- **ByteTrack** — a simple, high-performance tracker that associates detections using both high and low confidence scores, improving tracking continuity for temporarily occluded objects.
- **BotSORT** — combines motion (Kalman filter) and appearance (ReID) cues for more robust association in crowded scenes.

### Tracking Features

The tracking pipeline processes uploaded drone/aerial videos frame-by-frame and provides:

- **Unique ID assignment** — each detected object receives a persistent track ID (e.g., `Human #14`, `Car #7`) that is maintained across frames.
- **Per-frame counts** — current number of humans and cars visible in each frame.
- **Cumulative unique counts** — total unique humans and cars seen across the entire video (using track ID sets).
- **Tracking overlay** — each frame displays frame number, per-frame counts, and cumulative unique counts.
- **Per-frame chart** — an expandable line chart showing human and car counts over time.
- **Output video** — the annotated tracking video is available for download.
- **Progress reporting** — real-time progress bar with elapsed time and ETA.

### Unique Object Counting

Unlike per-frame detection counting, the tracking-based count uses set-based accumulation of track IDs across all frames. This provides a more accurate total count by deduplicating objects that appear in multiple frames.

---

## Task 05: Evaluation and Visualization

### Prediction Outputs

Detection visualizations are generated for sample test images showing bounding boxes, class labels, confidence scores, and per-image human/car counts. Side-by-side comparisons between standard inference and SAHI-enhanced inference demonstrate the improvement in small object recall.

### Counting Accuracy

Predicted human and car counts are compared against ground truth annotation counts on a sample of 200 test images. The analysis includes:

- Mean Absolute Error (MAE) per class.
- Mean signed error (to detect systematic over- or under-counting bias).
- Scatter plots of predicted vs. ground truth counts with a perfect-prediction reference line.

### Metrics Summary

| Model Configuration | mAP@0.5 (val) | mAP@0.5:0.95 (val) | Precision | Recall |
|---------------------|---------------|---------------------|-----------|--------|
| Baseline (YOLOv11m, 640 px) | 0.6922 | 0.4077 | 0.7746 | 0.6383 |
| Optimized (YOLOv11m, 1280 px) | 0.8265 | 0.5136 | 0.8337 | 0.7708 |

Test set evaluation (optimized model):

| Class | AP@0.5 | AP@0.5:0.95 |
|-------|--------|-------------|
| Human | 0.4531 | 0.1985 |
| Car | 0.7864 | 0.5279 |
| Overall | 0.6197 | 0.3632 |

---

## Project Structure

```
drone-object-detection/
│
├── README.md
├── requirements.txt                               # Python dependencies
│
├── assets/                                        # README images and visualizations
│   ├── training_curves.png
│   ├── sahi_comparison.png
│   ├── detection_sample.png
│   └── ...
│
├── docs/
│   └── assessment_brief.pdf                       # Assessment specification
│
├── notebooks/
│   ├── 01_eda.ipynb                               # Exploratory data analysis (24 cells)
│   ├── 02_preprocessing_and_training.ipynb        # Preprocessing, training, inference, SAHI
│   └── eda_results.md                             # Structured summary of EDA findings
│
├── weights/
│   ├── baseline_640/
│   │   └── best.pt                                # Baseline model weights (YOLOv11m, 640px)
│   └── optimized_1280/
│       └── best.pt                                # Optimized model weights (YOLOv11m, 1280px)
│
├── results/
│   ├── baseline_640/
│   │   └── training_log.csv                       # Epoch-by-epoch metrics (50 epochs)
│   └── optimized_1280/
│       └── training_log.csv                       # Epoch-by-epoch metrics (86 epochs)
│
└── streamlit_app/
    ├── app.py                                     # Streamlit application entry point
    ├── detector.py                                # Detection, SAHI inference, and tracking
    ├── config.py                                  # Constants, class maps, model search paths
    ├── components.py                              # Reusable UI components
    ├── styles.py                                  # Custom dark-mode CSS
    ├── requirements.txt                           # App-specific dependencies
    ├── Dockerfile                                 # Container image definition
    ├── .dockerignore                              # Docker build context exclusions
    └── docker-compose.yml                         # One-command container launch
```

---

## Setup and Reproduction

### Downloading Model Weights

> **Note:** Due to their large size, the trained model weights (`.pt` files) are not stored in this GitHub repository. They are hosted on Weights & Biases as versioned artifacts.

1. Go to the [W&B Artifacts Dashboard](https://wandb.ai/labibkashshaf-islamic-university-of-technology/drone-object-detection/artifacts)
2. Download `model-optimized-1280px` and/or `model-baseline-640px`.
3. Place the downloaded `.pt` files in the respective folders under `weights/` in the project root:
   - `weights/optimized_1280/best.pt`
   - `weights/baseline_640/best.pt`

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support (CPU inference is supported but slower)
- ultralytics >= 8.0
- sahi >= 0.11
- streamlit >= 1.30
- numpy, pandas, opencv-python-headless, Pillow

### Running the Web Application

```bash
cd streamlit_app
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app will auto-detect model weights from `weights/optimized_1280/best.pt`. Alternatively, use the **Upload weights** option in the sidebar to load any `.pt` file directly through the browser.

### Running with Docker

The Streamlit app can also be launched in a container with no local Python setup required.

**Using Docker Compose (recommended):**

```bash
cd streamlit_app
docker compose up --build
```

This mounts the project's `weights/` directory automatically. The app will be available at `http://localhost:8501`.

**Using Docker directly:**

```bash
cd streamlit_app
docker build -t drone-detection .
docker run -p 8501:8501 -v ../weights:/app/weights:ro drone-detection
```

> **Note:** The Docker image uses CPU-only PyTorch to keep the image size manageable (~2.5 GB). For GPU inference, replace the base image with an NVIDIA CUDA image and install the GPU variant of PyTorch.

### Kaggle Reproduction

1. Create a new Kaggle notebook with GPU (T4) acceleration.
2. Add the [VisDrone dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset) as input.
3. Run the preprocessing cells to create the filtered 2-class dataset.
4. Run training cells in order. The baseline takes approximately 45 minutes; the optimized run takes 2-4 hours.
5. Run inference and evaluation cells for detection outputs and metrics.

### Using Pretrained Weights

```python
from ultralytics import YOLO

model = YOLO('weights/optimized_1280/best.pt')
results = model.predict(source='image.jpg', conf=0.25, iou=0.45, imgsz=1280)
```

---

## Strengths, Limitations, and Challenges

### Strengths

- Systematic iterative training approach with data-driven decisions at each stage.
- Comprehensive EDA (24 analysis cells) that directly informed preprocessing choices (class merging, resolution selection, augmentation strategy).
- Custom P2 architectural modification demonstrates understanding of the small object detection problem beyond hyperparameter tuning.
- SAHI integration provides a practical inference-time solution for small object recall without retraining.
- Full tracking pipeline with ByteTrack and BotSORT for unique object counting across video frames.
- Production-quality web application with modular architecture, configurable inference settings, and weight upload support.

### Limitations

- Human detection AP on the test set (0.4531) remains significantly lower than car detection (0.7864) due to the extreme small object challenge inherent in aerial imagery.
- The strict IoU metric (mAP@0.5:0.95) penalizes localization error on tiny bounding boxes disproportionately. A 2-pixel offset on a 13-pixel box is a larger IoU penalty than the same offset on a 38-pixel box.
- Counting accuracy degrades in highly dense scenes (100+ humans) where overlapping detections are suppressed by NMS.
- Inference speed with SAHI is approximately 9× slower than standard inference due to per-tile processing.

### Challenges Faced

- **GPU memory constraints**: Training at 1280 px on a Tesla T4 (15 GB VRAM) required reducing batch size to 4 (optimized) and 2 (P2 head). Dense images with 900+ annotations triggered occasional CUDA OOM in the TaskAlignedAssigner, which was handled by automatic CPU fallback.
- **Tiny object detection floor**: Despite doubling resolution and adding a P2 head, objects under 8 px in both dimensions (4.2% of pedestrians) remain effectively undetectable by anchor-based or anchor-free detection architectures.
- **Validation-to-test generalization gap**: The test-dev split produced notably lower metrics than the validation split, reflecting greater scene diversity in the held-out data.

---

## References

- Zhu, X., Lyu, S., Wang, X., & Zhao, Q. (2021). TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-Captured Scenarios. *Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops*.
- Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection. *IEEE International Conference on Image Processing (ICIP)*.
- Ultralytics. (2024). YOLOv11 Documentation. https://docs.ultralytics.com/models/yolo11
- Zhu, P., Wen, L., Du, D., et al. (2021). Detection and Tracking Meet Drones Challenge. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
