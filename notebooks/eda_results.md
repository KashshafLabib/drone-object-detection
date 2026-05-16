# VisDrone Dataset — EDA Results & Analysis

> Reference document for all future sessions on this project.

## 1. Dataset Structure

| Split | Images | Labels | Annotations | Matched |
|-------|--------|--------|-------------|---------|
| Train | 6,471 | 6,471 | 343,205 | ✅ 100% |
| Val | 548 | 548 | 38,759 | ✅ 100% |
| Test-dev | 1,610 | 1,610 | 75,102 | ✅ 100% |
| Test-challenge | 1,580 | — | — | No labels |
| **Total** | **10,209** | **8,629** | **457,066** | |

- **Format:** YOLO (`class_id x_center y_center width height`, normalized)
- **Kaggle path:** `/kaggle/input/datasets/banuprasadb/visdrone-dataset/VisDrone_Dataset/`
- No missing labels, 0 empty label files, 0 images without annotations

---

## 2. Image Resolutions

| Split | Unique Res | Width Range | Height Range | MP Range |
|-------|-----------|-------------|--------------|----------|
| Train | 11 | 480–2000 | 360–1500 | 0.17–3.00 |
| Val | 3 | 960–1920 | 540–1080 | 0.52–2.07 |
| Test | 6 | 960–1920 | 540–1080 | 0.52–2.07 |

**Top resolutions:**
- 1400×1050 — 2,772 images (most common)
- 1400×788 — 2,232 images
- 1360×765 — 1,318 images
- 2000×1500 — 772 images

---

## 3. Class Distribution (All Splits)

| Class | ID | Total | Train % | Val % | Test % | Target? |
|-------|-----|-------|---------|-------|--------|---------|
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

### Target Class Breakdown

| Split | Target Annots | % of Total | Humans | Cars |
|-------|--------------|------------|--------|------|
| Train | 251,263 | 73.2% | 106,396 | 144,867 |
| Val | 28,033 | 72.3% | 13,969 | 14,064 |
| Test | 55,456 | 73.8% | 27,382 | 28,074 |

### Pedestrian vs People Ratio

| Split | Pedestrian | People | Ped % |
|-------|-----------|--------|-------|
| Train | 79,337 | 27,059 | 74.6% |
| Val | 8,844 | 5,125 | 63.3% |
| Test | 21,006 | 6,376 | 76.7% |

---

## 4. Object Density Per Image

| Split | Min | Max | Mean | Median | Std | >100 | >200 |
|-------|-----|-----|------|--------|-----|------|------|
| Train | 1 | 902 | 53.0 | 42 | 43.8 | 703 | 80 |
| Val | 1 | 317 | 70.7 | 65 | 46.0 | 107 | 8 |
| Test | 1 | 461 | 46.6 | 36 | 44.1 | 127 | 29 |

### Target Objects Per Image (Train)
- **All targets:** Mean=38.8, Max=896
- **Humans:** Mean=18.7, Max=888, Present in 5,684/6,471 images (87.8%)
- **Cars:** Present in ~6,133/6,471 images (94.8%)

### Background Images
- Train: 3 images with no target objects
- Val: 0
- Test: 7 images with no target objects

---

## 5. Bounding Box Size Analysis (CRITICAL)

### Normalized Area Stats (All Classes, Train)
- Mean: 0.001535, Median: 0.000462
- Min: ~0, Max: 0.302962

### COCO-Style Size Categories (Train, Target Classes)

| Class | Small (<32²) | Medium (32²-96²) | Large (>96²) |
|-------|-------------|-------------------|--------------|
| **Pedestrian** | **82.2%** | 17.4% | 0.4% |
| **People** | **86.9%** | 12.7% | 0.4% |
| **Car** | **48.2%** | 43.5% | 8.2% |

### Pixel Dimension Stats (Train)

| Class | W_px Mean | W_px Med | H_px Mean | H_px Med | Area_px Med |
|-------|-----------|----------|-----------|----------|-------------|
| Pedestrian | 16.2 | 13.0 | 30.7 | 25.0 | 315 |
| People | 16.4 | 13.0 | 24.7 | 20.0 | 273 |
| Car | 51.3 | 38.0 | 40.6 | 29.0 | 1,104 |

### Tiny Object Breakdown (Train, Pedestrian)
- **27.0%** have width < 8px
- **62.5%** have width < 16px
- **91.0%** have width < 32px
- **4.2%** have BOTH dimensions < 8px (essentially invisible)

---

## 6. Aspect Ratios (Train)

| Class | Mean | Median | Std |
|-------|------|--------|-----|
| Pedestrian | 0.37 | 0.31 | 0.22 |
| People | 0.45 | 0.40 | 0.19 |
| Car | 2.92 | 0.83 | 750.63 |

- Pedestrians/People are **tall & narrow** (W/H < 1)
- Cars have **high variance** (viewed from many angles, some extreme outlier aspect ratios)

---

## 7. Co-occurrence & Overlap

### Human-Car Co-occurrence (Train)
- Both Human & Car: 5,349 images (82.7%)
- Human only: 335 (5.2%)
- Car only: 784 (12.1%)
- Neither: 3 (0.0%)

### Same-Class IoU Overlap (Sampled)

| Class | Overlapping Pairs | Mean IoU | >0.3 IoU | >0.5 IoU |
|-------|-------------------|----------|----------|----------|
| Pedestrian | 753 | 0.143 | 13.3% | 2.5% |
| People | 121 | 0.221 | 34.7% | 11.6% |
| Car | 1,831 | 0.131 | 10.5% | 2.1% |

- **People class has highest overlap** — 34.7% pairs >0.3 IoU (crowded groups)

---

## 8. Spatial Distribution
- Objects are **relatively uniformly distributed** across image regions
- Slight concentration in center regions (drone typically points at area of interest)

---

## 9. Annotation Quality

| Check | Result |
|-------|--------|
| Zero/negative dimensions | 1 box |
| Out of bounds | 0 |
| Covers >50% image | 0 |
| Exact duplicates | 4 |
| Micro boxes (area < 1e-6) | 1 (Car) |
| Unknown class IDs | 0 |

**Quality verdict:** Excellent — only 6 problematic annotations out of 457,066 (0.001%)

---

## 10. Correlations
- Megapixels vs total objects: **0.082** (negligible)
- Megapixels vs humans: **-0.007** (none)
- Megapixels vs cars: **0.098** (negligible)

→ Image resolution does NOT predict object count

---

## 11. Key Findings & Decisions

### 🔴 Critical: Small Object Problem
- **82-87% of humans** are COCO-small (<32×32px)
- Median pedestrian is only **13×25 pixels**
- This is the dominant challenge — standard detectors will struggle

### 🟡 Class Imbalance
- Car (42%) dominates over Pedestrian (23%) and People (8%)
- Pedestrian:People ratio is ~3:1
- Merging Ped+People → "Human" creates a more balanced Human (31%) vs Car (42%) split

### 🟢 Dataset Health
- Clean annotations, no missing files, consistent splits
- 73% of annotations are target classes — most of the data is relevant
- 83% of images contain both humans and cars

### ⚡ Object Density
- Max 902 objects in one image
- 703 training images have >100 objects — NMS tuning critical
- High overlap in People class (crowded groups)
