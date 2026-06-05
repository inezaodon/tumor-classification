# Deliverable 2 — Brain Tumor MRI Classification
**AME 34351 | Machine Learning for Engineers | Dublin Summer 2026**

**Team Members:**
| Name | Primary Role |
|---|---|
| Marco Basile | Dashboard & artifact design |
| Jeremy Kelly | Data exploration & baseline models |
| Odon Ineza | Data cleaning & preprocessing pipeline |
| Tyler Asmussen | Model training, validation & evaluation |

---

## 1. Updated Problem Statement

*What are we trying to do, in one tight paragraph.*

- Narrow from D1's broad "medical software tool" framing to the specific 4-class task
- State the dataset, the classes, and the educational prototype framing
- End with what success looks like: honest evaluation that identifies where the model fails, not a diagnostic claim

---

## 2. Confirmed Data Source

*Everything McClarren asked for, in one place.*

- Dataset name and Kaggle URL
- Total image count (7,200)
- Class labels: glioma, meningioma, pituitary, notumor
- Per-class counts for train (1,400 each) and test (400 each)
- Note that image dimensions are non-uniform → standardization required
- Note V2 dataset guarantees: no duplicates, balanced classes, no patient overlap between train and test

---

## 3. Data Cleaning & Preprocessing

*What the dataset already handled vs. what we still need to do.*

**Already done by dataset (V2):**
- Duplicates removed
- Classes balanced
- Train/test split separated by patient (no leakage)

**Our preprocessing steps:**
- Resize all images to a fixed resolution (e.g. 224×224)
- Convert to grayscale — MRI is single-channel; color adds noise
- Normalize pixel values to [0, 1]
- Flatten to 1D vectors for baseline models (CNNs will use the 2D structure)

---

## 4. Visualization 1 — Class Balance

*Bar chart: train vs. test counts per class.*

**What to show:** Grouped bar chart, one group per class, two bars (train/test) each.

**What it tells us:** The dataset is perfectly balanced — each class has equal representation. This means accuracy is a meaningful metric here (though not sufficient on its own for a medical task). It also means no class-weighting tricks are needed during training.

---

## 5. Visualization 2 — Sample Images Per Class

*Grid: 4 classes × 5 example images.*

**What to show:** A 4-row grid with class labels on the left, 5 representative MRI images per row.

**What it tells us:** Makes the classification challenge concrete. Glioma and meningioma in particular share overlapping intensity patterns and spatial structure — visually similar even to a non-expert. The "notumor" class is visually distinct, which hints that binary detection (tumor/no tumor) may be easier than 4-class discrimination.

---

## 6. Visualization 3 — Raw Image Dimension Distribution

*Histograms of raw image widths and heights before preprocessing.*

**What to show:** Two side-by-side histograms (width distribution, height distribution) across the training set.

**What it tells us:** Images vary substantially in size, which is why standardization is a required preprocessing step — not an optional one. This motivates and justifies the resize operation in Section 3.

---

## 7. Key Challenge — High Dimensionality & Inter-Class Similarity

*The most interesting obstacle in the data, and our plan to address it.*

**The challenge:** After resizing to 224×224, each image has 50,176 features. With ~4,500 training images, any pixel-space model is severely underdetermined. Worse, the high-dimensional pixel space doesn't encode the spatial hierarchies (edges → textures → shapes) that actually distinguish tumor types. Glioma and meningioma have overlapping pixel distributions — a model operating on raw pixels will conflate them.

**How we plan to address it:** A CNN learns spatially-aware, hierarchical features rather than treating pixels independently. For Deliverable 3, we will compare a CNN trained from scratch against a transfer-learning model (e.g. ResNet-18 pretrained on ImageNet), which already encodes general visual features and requires far less data to fine-tune.

---

## 8. Updated ML Task Definition

*Precise feature and label descriptions.*

| | Description |
|---|---|
| **Task type** | Multi-class classification |
| **Input features** | Pixel values of a standardized grayscale MRI image (224×224 = 50,176 features for baselines; 2D array for CNN) |
| **Output labels** | One of: glioma, meningioma, pituitary, notumor |
| **Loss function** | Cross-entropy (multi-class) |
| **Evaluation metrics** | Accuracy, macro precision, macro recall, macro F1, confusion matrix |

*Note: We report all five metrics, not accuracy alone — standard practice for medical classification tasks.*

---

## 9. Train / Validation / Test Split Plan

*How data is partitioned and why it's leak-free.*

| Split | Size | Source |
|---|---|---|
| **Test** | 1,600 images (400/class) | Fixed by dataset — never touched during training |
| **Validation** | ~1,120 images (~280/class) | 20% stratified holdout from training set |
| **Train (fit)** | ~4,480 images (~1,120/class) | Remaining 80% of training set |

**Why this is leak-free:** The dataset's V2 release separated patients between the train and test directories. Our validation split is carved from within the already-separated training set using stratified sampling, so no patient's images appear in more than one split.

---

## 10. Baseline Models

*Three baselines to beat before a CNN is meaningful.*

### Baseline 1 — Majority Class Predictor
Always predicts the most frequent class. Because the dataset is balanced, this gives 25% accuracy. Sets the floor — any real model must beat this comfortably.

### Baseline 2 — Logistic Regression on Pixel Features
Flatten and normalize each image, then fit a logistic regression with L2 regularization. This is a linear model in pixel space — it can find global brightness/contrast patterns but has no sense of spatial structure. We expect moderate accuracy, with the model struggling most on glioma vs. meningioma.

### Baseline 3 — Brightness Threshold Heuristic
Compute mean pixel intensity per image and use a threshold to separate "notumor" from "has tumor" (binary). Threshold is tuned on the validation set. This is intentionally binary — a simple intensity rule cannot distinguish tumor *types*, only tumor *presence*. Reports binary accuracy, precision, recall, and F1.

---

## 11. Results Summary

*To be filled in after running the notebook.*

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Majority Class | — | — | — | — |
| Logistic Regression | — | — | — | — |
| Brightness Threshold (binary) | — | — | — | — |
| *CNN (Deliverable 3)* | — | — | — | — |
| *Transfer Learning (Deliverable 3)* | — | — | — | — |

Confusion matrices for each baseline will be plotted here.

---

## 12. Limitations & Clinical Disclaimer

This notebook is an **educational prototype** built on a public benchmark dataset. It is **not clinically validated** and must not be used for diagnosis. Reported performance reflects controlled benchmark conditions — real-world MRI data varies in scanner type, slice orientation, acquisition protocol, and patient population in ways this dataset does not capture. Additional data collection, prospective validation, and regulatory review would be required before anything like this could be used in a clinical setting.
