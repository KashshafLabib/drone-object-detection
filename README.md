# Drone Human Detection and Counting System

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

A thorough EDA was conducted across 24 analysis cells covering dataset structure, class distributions, bounding box statistics, spatial distributions, annotation quality, and visual inspection. The full EDA notebook is available at `kaggle_eda/drone-object-detection-eda.ipynb`.

Key findings from the analysis:

#### Image Resolutions

The dataset contains images at 11 distinct resolutions ranging from 480x360 to 2000x1500 pixels. The most common resolution is 1400x1050 (2,772 images). All splits have 100% image-label pairing with zero missing files.

#### Class Distribution

Target classes (Pedestrian, People, Car) account for 73.2% of all annotations in the training set. The remaining 26.8% belong to non-target vehicle classes.

| Class | Train Count | Percentage |
|-------|------------|------------|
| Car | 144,867 | 42.2% |
| Pedestrian | 79,337 | 23.1% |
| People | 27,059 | 7.9% |
| Other (7 classes) | 91,942 | 26.8% |

#### Small Object Problem (Critical Finding)

This is the dominant challenge in the dataset. The vast majority of human annotations are extremely small by standard detection benchmarks:

| Class | Small (<32x32 px) | Medium (32-96 px) | Large (>96 px) |
|-------|-------------------|--------------------|--------------------|
| Pedestrian | 82.2% | 17.4% | 0.4% |
| People | 86.9% | 12.7% | 0.4% |
| Car | 48.2% | 43.5% | 8.2% |

Median pixel dimensions: Pedestrian 13x25 px, People 13x20 px, Car 38x29 px. Approximately 27% of pedestrians have a width under 8 pixels and 4.2% have both dimensions under 8 pixels.

#### Object Density

Images contain up to 902 annotated objects, with 703 training images exceeding 100 objects. Mean density is 53 objects per image in the training split. 82.7% of training images contain both humans and cars.

#### Annotation Quality

Out of 457,066 annotations, only 6 were identified as problematic (1 zero-dimension box, 4 exact duplicates, 1 micro-area box). The dataset is clean and ready for training without annotation corrections.

### Preprocessing Pipeline

The preprocessing stage transforms the 10-class VisDrone dataset into a focused 2-class detection dataset:

**Step 1: Class Remapping**

Pedestrian (class 0) and People (class 1) are merged into a single Human class (new class 0). Car (class 3) is remapped to new class 1. All other 7 classes are discarded.

Rationale for merging Pedestrian and People:
- Near-identical pixel dimensions (median 13x25 vs 13x20 px).
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
| Input resolution | 640 x 640 |
| Epochs | 50 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.001 (cosine decay to 0.01x) |
| Augmentation | Mosaic, MixUp (0.15), HSV jitter, horizontal flip, scale (0.5), rotation (5 deg) |

#### Phase 2: Optimized (1280 px)

Addresses the small object problem by doubling input resolution. At 1280 px, a 13 px pedestrian occupies approximately 12 px after resize, compared to approximately 6 px at 640 px.

| Parameter | Value |
|-----------|-------|
| Model | YOLOv11m (pretrained on COCO) |
| Input resolution | 1280 x 1280 |
| Epochs | 100 (early stopped at 86) |
| Batch size | 4 (reduced for GPU memory) |
| Optimizer | AdamW |
| Learning rate | 0.0005 (reduced for stability at high resolution) |
| Patience | 20 epochs (early stopping) |
| Augmentation | Same as baseline |

#### Phase 3: P2 Detection Head (1280 px)

A custom architectural modification adding a 4th detection head at stride 4. Standard YOLOv11 detects at strides 8, 16, and 32. The P2 head adds detection at stride 4, producing a 320x320 feature map at 1280 px input where each cell covers 4x4 pixels of the original image.

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
| Input resolution | 1280 x 1280 |
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

Model weights are available via Google Drive: [Download Link (placeholder -- to be updated)]

---

## Task 03: Detection and Counting

### Detection Pipeline

The detection system processes input images through the trained YOLOv11m model and produces:

- Color-coded bounding boxes: red for humans, green for cars.
- Per-box confidence scores.
- A count overlay displaying total humans, total cars, and total objects.

The inference function accepts configurable confidence threshold (default 0.25) and NMS IoU threshold (default 0.45).

### Counting Logic

Human counting is implemented as a direct summation of all detection boxes classified as Human (class 0) with confidence above the threshold. This is a frame-level count -- each detected bounding box increments the counter by one.

While simple, this approach is appropriate for single-image analysis. Limitations include potential double-counting of partially occluded individuals and missed counts for humans below the confidence threshold.

### SAHI Integration

To address the small object detection gap, SAHI (Slicing Aided Hyper Inference) is integrated as an alternative inference mode. SAHI divides the input image into overlapping tiles (640x640 px with 20% overlap), runs detection independently on each tile at native resolution, then merges all detections back into the original coordinate space using NMS.

This avoids downscaling the full image and preserves fine spatial detail for tiny objects. A 13 px pedestrian remains 13 px within its tile rather than being compressed further during whole-image resize.

---

## Task 04: Object Tracking (Bonus)

*To be implemented.*

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
Drone Human Detection & Counting System/
|
|-- README.md
|
|-- kaggle_eda/
|   |-- cell_01_setup_and_structure.py        # Dataset loading, path config, label parsing, resolution analysis
|   |-- cell_02_class_distribution.py         # Class counts, target class focus, density analysis
|   |-- cell_03_bbox_analysis.py              # Bounding box sizes (normalized and pixel), aspect ratios, COCO-style categories
|   |-- cell_04_cooccurrence_and_overlap.py   # Class co-occurrence, IoU overlap, tiny object deep-dive
|   |-- cell_05_visualizations.py             # Annotated sample images (dense, sparse, human-heavy, car-heavy)
|   |-- cell_06_summary.py                    # Per-class statistics, correlation analysis, EDA summary
|   |-- drone-object-detection-eda.ipynb      # Complete executed EDA notebook with outputs
|   |-- eda_results.md                        # Structured summary of all EDA findings
|
|-- kaggle_training/
|   |-- cell_01_preprocessing.py              # Label remapping, class filtering, data.yaml creation
|   |-- cell_02_baseline_training.py          # YOLOv11m baseline training at 640 px
|   |-- cell_03_optimized_training.py         # YOLOv11m optimized training at 1280 px, comparison
|   |-- cell_04_inference_and_counting.py     # Detection, counting, test set evaluation, counting accuracy
|   |-- cell_05_sahi_and_export.py            # SAHI sliced inference, side-by-side comparison, model export
|   |-- cell_06_p2_head_training.py           # Custom P2 head YAML, architecture verification, P2 training
|   |-- cell_07_sahi_evaluation.py            # SAHI batch evaluation with mAP metrics via torchmetrics
|   |-- weights and results/
|       |-- baseline 640/
|       |   |-- best.pt                       # Baseline model weights
|       |   |-- results.csv                   # Epoch-by-epoch training metrics
|       |-- optimised 1280/
|           |-- best.pt                       # Optimized model weights
|           |-- results.csv                   # Epoch-by-epoch training metrics
```

---

## Setup and Reproduction

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- ultralytics >= 8.0
- sahi
- torchmetrics
- numpy, pandas, matplotlib, seaborn, Pillow, PyYAML, tqdm

### Kaggle Reproduction

1. Create a new Kaggle notebook with GPU (T4) acceleration.
2. Add the [VisDrone dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset) as input.
3. Run the preprocessing cells from `cell_01_preprocessing.py` to create the filtered 2-class dataset.
4. Run training cells in order. The baseline takes approximately 45 minutes; the optimized run takes 2-4 hours.
5. Run inference and evaluation cells for detection outputs and metrics.

### Using Pretrained Weights

Download the trained model weights from [Google Drive (placeholder -- to be updated)] and load directly:

```python
from ultralytics import YOLO

model = YOLO('path/to/best.pt')
results = model.predict(source='image.jpg', conf=0.25, iou=0.45, imgsz=1280)
```

---

## Strengths, Limitations, and Challenges

### Strengths

- Systematic iterative training approach with data-driven decisions at each stage.
- Comprehensive EDA that directly informed preprocessing choices (class merging, resolution selection, augmentation strategy).
- Custom P2 architectural modification demonstrates understanding of the small object detection problem beyond hyperparameter tuning.
- SAHI integration provides a practical inference-time solution for small object recall without retraining.

### Limitations

- Human detection AP on the test set (0.4531) remains significantly lower than car detection (0.7864) due to the extreme small object challenge inherent in aerial imagery.
- The strict IoU metric (mAP@0.5:0.95) penalizes localization error on tiny bounding boxes disproportionately. A 2-pixel offset on a 13-pixel box is a larger IoU penalty than the same offset on a 38-pixel box.
- Counting accuracy degrades in highly dense scenes (100+ humans) where overlapping detections are suppressed by NMS.
- Inference speed with SAHI is approximately 9x slower than standard inference due to per-tile processing.

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
