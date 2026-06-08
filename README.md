\# Automated Glacial Lake Detection from Multi-Temporal Satellite Imagery



\## GLOFeagles '26 Challenge Submission

\*\*NCVPRIPG 2026 – LNMIIT Jaipur\*\*



\---



\## Overview



This repository contains our submission for the \*\*GLOFeagles '26 Challenge: Automated Detection of Glacial Lakes from Multi-Temporal Satellite Imagery\*\*.



The objective is to accurately identify and segment glacial lakes from satellite imagery while minimizing confusion with visually similar features such as:



\- Snow

\- Clean Ice

\- Debris-Covered Ice

\- Terrain Shadows

\- Cloud Cover



To address this challenge, we developed a deep-learning-based segmentation framework using \*\*U-Net++ with an EfficientNet-B3 encoder\*\*, combined with advanced augmentation techniques, threshold optimization, and test-time augmentation (TTA).



\---



\## Methodology



\### Model Architecture



| Component | Description |

|------------|------------|

| Architecture | U-Net++ |

| Encoder | EfficientNet-B3 |

| Pretrained Weights | ImageNet |

| Input Size | 512 × 512 |

| Output | Binary Segmentation Mask |



\---



\### Data Augmentation



To improve robustness in complex mountainous terrain, the following augmentations were applied:



\#### Geometric Augmentations

\- Horizontal Flip

\- Vertical Flip

\- Random Rotation

\- Shift-Scale-Rotate

\- Elastic Transform

\- Grid Distortion



\#### Photometric Augmentations

\- Brightness and Contrast Adjustment

\- Hue-Saturation-Value Transformation

\- CLAHE

\- Gaussian Noise

\- Gamma Correction

\- Shadow Simulation



\#### Occlusion Handling

\- Coarse Dropout



\---



\## Loss Function



The final loss function combines multiple objectives:



\### Combined Loss



\- 40% Focal Loss

\- 40% Dice Loss

\- 20% Boundary Loss



This combination improves:



\- Lake boundary delineation

\- Small object detection

\- Class imbalance handling



\---



\## Training Configuration



| Parameter | Value |

|------------|------------|

| Image Size | 512 × 512 |

| Batch Size | 4 |

| Epochs | 100 |

| Learning Rate | 1e-3 |

| Optimizer | AdamW |

| Scheduler | Cosine Annealing LR |



\---



\## Inference Pipeline



\### Multi-Scale Inference



Predictions are generated at multiple resolutions:



\- 384 × 384

\- 512 × 512

\- 640 × 640



\### Test-Time Augmentation (TTA)



The following transformations are applied during inference:



\- Original Image

\- Horizontal Flip

\- Vertical Flip

\- 90° Rotation

\- 180° Rotation

\- 270° Rotation



Predictions from all transformations are averaged to improve robustness.



\### Post-Processing



The generated masks are refined using:



\- Morphological Opening

\- Morphological Closing

\- Connected Component Analysis

\- Small Region Removal



\---



\## Threshold Optimization



A threshold sweep was performed on the validation dataset to determine the optimal segmentation threshold that maximizes Intersection over Union (IoU).



The selected threshold was used for final mask generation on the test dataset.



\---



\## Evaluation Metrics



The model is evaluated using:



\- Intersection over Union (IoU)

\- Precision

\- Recall

\- F1 Score

\- Dice Score

\- Cohen's Kappa



\---



\## Repository Structure



```text

.

├── train.py

├── inference.py

├── model\_architecture.py

├── utils.py

├── Lake\_Glacier\_Detection.ipynb

├── requirements.txt

├── README.md

│

├── models/

│   ├── glacier\_lake\_model\_final.pth

│   └── glacier\_lake\_checkpoint.pth

│

├── segmentation\_masks/

│

├── reports/

│   ├── technical\_report.pdf

│   └── evaluation\_report.pdf

│

└── assets/

```



\---



\## Installation



```bash

pip install -r requirements.txt

```



\---



\## Training



```bash

python train.py

```



\---



\## Inference



```bash

python inference.py

```



Generated segmentation masks will be saved automatically to the configured output directory.



\---



\## Results



| Metric | Score |

|----------|----------|

| IoU | To Be Updated |

| Precision | To Be Updated |

| Recall | To Be Updated |

| F1 Score | To Be Updated |

| Cohen's Kappa | To Be Updated |



\---



\## Challenges Addressed



The proposed framework is designed to improve performance under challenging conditions including:



\- Terrain Shadows

\- Debris-Covered Ice

\- Snow-Covered Regions

\- Illumination Variations

\- Small and Irregularly Shaped Lakes



\---



\## Reproducibility



This repository includes:



\- Trained Model Weights (.pth)

\- Source Code

\- Python Notebook

\- Segmentation Outputs

\- Technical Report

\- Evaluation Report

\- README

\- Requirements File



allowing evaluators to reproduce the submitted results directly.



\---



\## Submission Components



✔ Trained Model Weights



✔ Source Code



✔ Python Notebook



✔ Segmentation Masks



✔ Technical Report



✔ Evaluation Report



✔ README



✔ requirements.txt



✔ Demonstration Video



\---



\## Demonstration Video



\*\*YouTube Link:\*\*  

\[Add Your Unlisted YouTube Link Here]



\---



\---



\## Team



\### Team Members



1\. \*\*Santosh S\*\* \*(Team Lead / Corresponding Member)\*

2\. \*\*Rashmika\*\*

3\. \*\*Visweshwaran K\*\*



\---



\## Acknowledgements



This work was developed for the \*\*GLOFeagles '26 Challenge\*\* organized by \*\*Shiv Nadar University Chennai\*\* under \*\*NCVPRIPG 2026\*\*.



The challenge aims to advance automated glacial lake monitoring systems for improved GLOF risk assessment, climate resilience, and disaster preparedness.

