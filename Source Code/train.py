import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

# Import your custom modules
from model_architecture import get_model
from utils import LakeDataset, combined_loss, compute_iou, compute_dice

# Configurations
BASE_PATH = "./Label_Subset" # Update path based on your local/drive setup
IMG_BASE        = os.path.join(BASE_PATH, "Class_Labels")
GT_BASE         = os.path.join(BASE_PATH, "Ground truth")
MODEL_SAVE_PATH = "./glacierlake_model.pth"

IMG_SIZE   = 512
BATCH_SIZE = 4
EPOCHS     = 100
LR         = 1e-3
THRESHOLD  = 0.32

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # 1. Gather Image/Mask Pairs
    pairs = []
    for cls in sorted(os.listdir(IMG_BASE)):
        img_folder  = os.path.join(IMG_BASE, cls)
        mask_folder = os.path.join(GT_BASE, cls)
        if not os.path.isdir(img_folder): continue
        for fname in os.listdir(img_folder):
            img_path  = os.path.join(img_folder, fname)
            mask_path = os.path.join(mask_folder, fname)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path, cls))

    # 2. Train/Val Split
    pair_data    = [(img, mask) for img, mask, _ in pairs]
    pair_classes = [cls for _, _, cls in pairs]

    train_pairs, val_pairs, train_classes, _ = train_test_split(
        pair_data, pair_classes, test_size=0.20, random_state=42, stratify=pair_classes
    )

    # 3. Augmentations
    train_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.15, rotate_limit=30, border_mode=0, p=0.6),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.40, contrast_limit=0.40, p=0.7),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.5),
        A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=0.5),
        A.GaussNoise(var_limit=(10.0, 80.0), p=0.4),
        A.RandomGamma(gamma_limit=(60, 140), p=0.4),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, p=0.4),
        A.CoarseDropout(max_holes=12, max_height=32, max_width=32, min_holes=4, min_height=8, min_width=8, fill_value=0, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 4. Data Loaders & Class Weighting
    train_ds = LakeDataset(train_pairs, transform=train_transform)
    val_ds   = LakeDataset(val_pairs,   transform=val_transform)

    class_freq = Counter(train_classes)
    debris_keywords = ["debris", "Debris", "covered", "Covered"]

    def get_weight(cls):
        if any(k in cls for k in debris_keywords):
            return 3.0 / class_freq[cls]
        return 1.0 / class_freq[cls]

    sample_weights = torch.tensor([get_weight(c) for c in train_classes], dtype=torch.float)
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    # 5. Initialization
    model = get_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 6. Training Loop
    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss  = combined_loss(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = v_iou = v_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds  = model(imgs)
                v_loss += combined_loss(preds, masks).item()
                v_iou  += compute_iou(preds, masks, THRESHOLD)
                v_dice += compute_dice(preds, masks, THRESHOLD)

        avg_t = t_loss / len(train_loader)
        avg_v = v_loss / len(val_loader)
        avg_i = v_iou  / len(val_loader)
        avg_d = v_dice / len(val_loader)
        scheduler.step()

        if avg_v < best_val_loss:
            best_val_loss = avg_v
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Epoch {epoch:3d}/{EPOCHS} │ Val Loss: {avg_v:.4f} │ IoU: {avg_i:.3f} - SAVED")
        elif epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{EPOCHS} │ Val Loss: {avg_v:.4f} │ IoU: {avg_i:.3f}")

if __name__ == "__main__":
    main()