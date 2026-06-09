# 🌍 Glacier Lake Detection using Deep Learning for GLOF Risk Assessment

## GLOFeagles '26 Challenge Submission

This repository contains our complete solution for the **GLOFeagles '26 Challenge – Automated Detection of Glacial Lakes from Multi-Temporal Satellite Imagery**, organized under **NCVPRIPG 2026**.

The objective of this project is to accurately identify and segment glacial lakes from satellite imagery while minimizing false detections caused by snow, ice, terrain shadows, debris-covered glaciers, and other visually similar features.

Our approach utilizes a **U-Net++ semantic segmentation architecture with an EfficientNet-B3 encoder**, enhanced with extensive data augmentation, weighted sampling, custom loss functions, threshold optimization, and multi-scale Test-Time Augmentation (TTA) to achieve robust and reliable glacial lake segmentation.

---

# 📂 Repository Structure

```text
ICE_KATTI_GLOF_Challenge/

├── Trained model files/
│   ├── glacier_lake_model_final.pth
│   └── glacier_lake_checkpoint.pth
│
├── Source Code/
│   ├── train.py
│   ├── inference.py
│   ├── model_architecture.py
│   └── utils.py
│   └──Lake_Glacier_Detection_full_source_code
├── Python Notebook/
│   └── Glacier_Lake_Detection.ipynb
│───SNU GLOF 2026 Dataset
|   └── Label_Subset
|       └── Class_Labels
|       └── Ground truth
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── Segmentation masks for all images/
│   ├── 1_mask.png
│   ├── 2_mask.png
│   └── ...
│
├── Technical and Evaluation Reports/
│   ├── Technical_Report.pdf
│   └── Evaluation_Report.pdf
│
├── README.md
└── requirements.txt
```

---

# 🎯 Problem Statement

Glacial lakes are rapidly expanding due to glacier retreat caused by climate change. These lakes can trigger catastrophic **Glacial Lake Outburst Floods (GLOFs)**, posing serious risks to downstream communities and infrastructure.

The primary goal of this challenge is to automatically detect glacial lakes from satellite imagery and generate accurate segmentation masks that can support GLOF risk assessment, hazard monitoring, and early warning systems.

---

# 🏗️ Proposed Architecture

Our solution is based on a semantic segmentation pipeline designed specifically for glacial lake extraction from high-altitude satellite imagery.

### Architecture Components

- U-Net++ Decoder
- EfficientNet-B3 Encoder (ImageNet Pretrained)
- Advanced Data Augmentation
- Weighted Sampling Strategy
- Focal + Dice + Boundary Loss
- Threshold Optimization
- Multi-Scale Test-Time Augmentation
- Morphological Post Processing

---

# 🔄 Workflow

## Overall Pipeline
<p align="center">
  <img src="https://github.com/user-attachments/assets/8f016660-708c-4fbf-8f37-30fcc179681b"  width="60%">
</p>


The complete workflow follows the steps below:

1. Dataset Preparation
2. Data Augmentation
3. Model Training
4. Validation
5. Threshold Optimization
6. Test-Time Augmentation
7. Post Processing
8. Segmentation Mask Generation

---

# 📊 Dataset Preparation



The provided challenge dataset consists of satellite images and corresponding ground-truth segmentation masks for glacial lake segmentation. The dataset was divided into training and validation subsets using stratified sampling to maintain a balanced representation of different glacial lake categories.

### Dataset Split

| Dataset Split | Percentage | Number of Images |
|--------------|------------|------------------|
| Training Set | 80% | 48 |
| Validation Set | 20% | 12 |

### Label Subsets

The dataset is organized into six label subsets representing different environmental conditions and challenges commonly encountered in glacial lake detection:

- Cloud Cover
- Debris Cover
- Moraine
- Snow Cover
- Terrain Shadow
- Turbidity

### Data Structure

For each label subset, the challenge organizers provided:

1. **Original Satellite Images** – Remote sensing images containing glacial lakes under varying environmental conditions.
2. **Ground-Truth Segmentation Masks** – Pixel-wise annotated masks identifying the exact boundaries of glacial lakes.

The original images are used as inputs to the segmentation model, while the corresponding masks serve as ground-truth labels during training and validation. This image-mask pairing enables supervised learning for accurate glacial lake segmentation.

The inclusion of diverse label subsets such as cloud-covered, debris-covered, snow-covered, and turbid lakes ensures that the trained model can generalize effectively across a wide range of real-world glacial lake scenarios.
# 🔧 Data Augmentation

To improve robustness against varying environmental conditions, extensive augmentation techniques were applied.

### Geometric Transformations

- Horizontal Flip
- Vertical Flip
- Random Rotation
- Shift Scale Rotate
- Elastic Transform
- Grid Distortion

### Photometric Transformations

- Random Brightness and Contrast
- Hue-Saturation Adjustment
- CLAHE Enhancement
- Random Gamma
- Gaussian Noise
- Random Shadow Simulation

### Occlusion Simulation

- Coarse Dropout

### Normalization

ImageNet Mean and Standard Deviation Normalization

---

# 🧠 Model Architecture

## U-Net++ with EfficientNet-B3 Encoder

| Component | Configuration |
|------------|---------------|
| Architecture | U-Net++ |
| Encoder | EfficientNet-B3 |
| Input Size | 512 × 512 |
| Output | Binary Segmentation Mask |
| Framework | PyTorch |

### Why U-Net++?

- Better feature fusion
- Dense skip connections
- Improved segmentation accuracy
- Better localization of small glacial lakes

### Why EfficientNet-B3?

- Efficient feature extraction
- Strong transfer learning capability
- Improved performance on satellite imagery

---

# 📉 Loss Function

To improve segmentation performance and boundary detection, a custom hybrid loss function was used.

Loss Function:

Focal Loss (40%)
+
Dice Loss (40%)
+
Boundary Loss (20%)

### Focal Loss

Handles class imbalance and difficult samples.

### Dice Loss

Improves overlap between prediction and ground truth masks.

### Boundary Loss

Improves segmentation around lake boundaries.

---

# ⚙️ Training Configuration

| Parameter | Value |
|------------|---------|
| Image Size | 512 × 512 |
| Batch Size | 4 |
| Epochs | 100 |
| Learning Rate | 0.001 |
| Optimizer | AdamW |
| Scheduler | Cosine Annealing Learning Rate |
| Encoder Weights | ImageNet |

---

# ⚖️ Class Imbalance Handling

Certain glacial lake categories, especially debris-covered lakes, contained fewer samples.

To address this challenge:

- Weighted Random Sampling was used.
- Debris-related classes were oversampled.
- Class-frequency-based weighting was incorporated during training.

This improved learning for underrepresented categories.

---

# 🚀 Test-Time Augmentation (TTA)

To improve robustness during inference, predictions were generated using:

### Spatial Augmentations

- Original Image
- Horizontal Flip
- Vertical Flip
- 90° Rotation
- 180° Rotation
- 270° Rotation

### Multi-Scale Inference

- 384 × 384
- 512 × 512
- 640 × 640

Predictions from all scales and augmentations were averaged to generate the final probability map.

---

# 🧹 Post Processing

After inference, the probability maps undergo:

- Threshold Optimization
- Morphological Opening
- Morphological Closing
- Connected Component Filtering

This removes noise and improves final segmentation quality.

---

# 📈 Evaluation Metrics

The following metrics were used for evaluation:

- Intersection over Union (IoU)
- Precision
- Recall
- F1 Score
- Dice Score
- Specificity
- Cohen’s Kappa

---

# 📊 Quantitative Results

## Performance Metrics

<img width="2688" height="1470" alt="Figure_A_Metrics" src="https://github.com/user-attachments/assets/aa74a25e-e759-467d-af5c-1552455d40dd" />


**Figure A. Quantitative performance metrics of the proposed glacial lake segmentation model evaluated on the validation dataset.**

| Metric | Score (%) |
|---------|---------:|
| IoU | 51.61 |
| Precision | 68.99 |
| Recall | 67.19 |
| F1 Score | 68.08 |
| Dice Score | 68.08 |
| Specificity | 99.52 |
| Cohen's Kappa | 67.58 |

The model achieved an **Intersection over Union (IoU) of 51.61%**, which serves as the primary evaluation metric for the challenge. Precision and Recall values remain well balanced, resulting in an F1 Score of 68.08%. The high specificity of 99.52% indicates strong discrimination between glacial lakes and background regions, while a Cohen's Kappa score of 67.58% demonstrates substantial agreement between predictions and ground-truth annotations.

---

## Confusion Matrix Analysis


<img width="796" height="565" alt="4587ed14-7521-4604-a15f-2bf437e42bab" src="https://github.com/user-attachments/assets/b07b7cef-e28b-4231-b1e8-69e3a1903f21" />

**Figure B. Pixel-level confusion matrix evaluated on the validation dataset at a probability threshold of 0.4.**

| Metric | Count |
|---------|---------:|
| True Positive (TP) | 32,920 |
| False Positive (FP) | 14,794 |
| False Negative (FN) | 16,076 |
| True Negative (TN) | 3,081,938 |

### Interpretation

- **True Positives (32,920):** Lake pixels correctly identified as glacial lakes.
- **True Negatives (3,081,938):** Background pixels correctly classified as non-lake regions.
- **False Positives (14,794):** Background pixels incorrectly classified as lakes.
- **False Negatives (16,076):** Actual lake pixels missed by the model.

The confusion matrix indicates that the model successfully captures the majority of glacial lake pixels while maintaining a very low false positive rate relative to the large background class. This behavior is particularly important for operational glacial lake monitoring, where minimizing false alarms while preserving detection capability is critical.


---

# 🏆 Results and Robustness Analysis

The proposed framework successfully detects glacial lakes across a variety of challenging environments.

### Successfully Handles

✅ Small Lakes

✅ Large Lakes

✅ Terrain Shadows

✅ Snow-Covered Areas

✅ Debris-Covered Surfaces

✅ Complex Mountain Terrain

The combination of advanced augmentation, weighted sampling, boundary-aware loss functions, and test-time augmentation significantly improves robustness and reduces false detections.

---

# 🖼️ Qualitative Results

## Validation Examples

![Validation Results](<img width="1108" height="293" alt="image" src="https://github.com/user-attachments/assets/e09b3e5d-f2bb-4acc-833b-771da2c58a27" />
)


## Example 1

### Input Image

![Online Image 1](images/online_test_1.png)

### Predicted Mask

![Prediction 1](images/prediction_1.png)

---

## Example 2

### Input Image

![Online Image 2](images/online_test_2.png)

### Predicted Mask

![Prediction 2](images/prediction_2.png)

---

# 💻 Setup Instructions

## Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/TeamName_GLOF_Challenge.git

cd TeamName_GLOF_Challenge
```

## Step 2: Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / MacOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

## Step 4: Dataset Placement

Place the challenge dataset as:

```text
Dataset/

├── Class_Labels/
├── Ground_truth/
└── Test_Images/
```

## Step 5: Configure Paths

Open the source files and update the following paths according to your system.

```python
BASE_PATH = "YOUR_DATASET_PATH"

IMG_BASE = BASE_PATH + "/Class_Labels"

GT_BASE = BASE_PATH + "/Ground_truth"

MODEL_SAVE_PATH = "YOUR_MODEL_SAVE_PATH"

OUTPUT_DIR = "YOUR_OUTPUT_DIRECTORY"
```

## Step 6: Train the Model

```bash
python train.py
```

## Step 7: Run Inference

```bash
python inference.py
```

Generated masks will automatically be saved inside:

```text
Segmentation masks for all images/
```

---

# 🔁 Reproducibility

To reproduce our results:

1. Use the provided trained weights.
2. Use only the official challenge dataset.
3. Maintain the same image resolution.
4. Use the optimized threshold value.
5. Follow the provided folder structure.

---

# 🎥 Explanation Video

YouTube Link:

PASTE_YOUR_UNLISTED_YOUTUBE_LINK_HERE

---


# 👥 Team Members

| Name | GitHub 🐙| LinkedIn 💼|
|------|---------|----------|
| **Santosh S (Team Lead)** | [GitHub](https://github.com/Sxnthxsh-S3107) | [LinkedIn](https://www.linkedin.com/in/santhosh-s-553436322) |
| Rashmika | [GitHub](https://github.com/rashmikaishere) | [LinkedIn](https://www.linkedin.com/in/rashmika-m-s) |
| Visweshwaran K | [GitHub](https://github.com/vizarrd) | [LinkedIn](https://linkedin.com/in/your-linkedin) |

---


---

# 🙏 Acknowledgements

This work was developed as part of the GLOFeagles '26 Challenge under NCVPRIPG 2026, organized by Shiv Nadar University Chennai.

We thank the organizers for providing the dataset, evaluation protocol, and challenge platform that enabled this work.

---

**Team ICE KATTI**  
**Automated Detection of Glacial Lakes from Multi-Temporal Satellite Imagery**  
**NCVPRIPG 2026**
