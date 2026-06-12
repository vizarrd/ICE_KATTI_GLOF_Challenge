Justification of Key Design Choices

The proposed framework incorporates several design choices aimed at improving glacial lake segmentation under challenging environmental conditions. These decisions were motivated by the characteristics of the provided dataset and the need for robust generalization.

EfficientNet-B3 Encoder:

EfficientNet-B3 was selected as the encoder backbone due to its strong feature extraction capability and favorable accuracy-to-parameter ratio. Pretrained ImageNet weights enabled effective transfer learning despite the limited size of the labeled dataset.

U-Net++ Architecture:

U-Net++ was chosen because its nested skip connections improve multi-scale feature fusion and help preserve fine boundary information. This is particularly important for accurately delineating glacial lake contours.

Hybrid Loss Function:

A combination of Focal Loss, Dice Loss, and Boundary Loss was adopted.

Focal Loss addresses class imbalance by focusing on difficult samples.
Dice Loss directly optimizes region overlap.
Boundary Loss improves segmentation around lake edges.

This combination provides balanced optimization of both regional accuracy and boundary quality.

Weighted Sampling:

The dataset contains environmental categories with varying sample distributions. Weighted sampling was used to increase the representation of challenging debris-covered regions during training and reduce class bias.

Data Augmentation:

Extensive geometric, photometric, and occlusion-based augmentations were applied to improve robustness against variations in illumination, shadows, snow cover, viewing conditions, and terrain appearance.

Threshold Optimization:

Instead of using a fixed threshold, multiple thresholds were evaluated on the validation dataset. The threshold producing the highest validation IoU was selected for final inference.

Test-Time Augmentation:

Multi-scale and flip-based Test-Time Augmentation was employed during inference to improve prediction stability and reduce sensitivity to image orientation and scale variations.

