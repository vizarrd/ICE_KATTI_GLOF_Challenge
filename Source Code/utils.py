import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp

# --- DATASET ---
class LakeDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs     = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = cv2.imread(img_path)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"]

        return img.float(), mask.float().unsqueeze(0)

# --- LOSS FUNCTIONS ---
dice_loss_fn  = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
focal_loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE)

def boundary_loss(pred_logits, target, weight=0.2):
    """Penalises errors near lake boundaries more heavily."""
    pred    = torch.sigmoid(pred_logits)
    kernel  = torch.ones(1, 1, 5, 5, device=target.device) / 25.0
    dilated = F.conv2d(target, kernel, padding=2).clamp(0, 1)
    boundary_region = dilated - target # ring around lake edge
    bce      = F.binary_cross_entropy(pred, target, reduction='none')
    weighted = bce * (1.0 + weight * boundary_region * 10)
    return weighted.mean()

def combined_loss(pred, target, alpha=0.4):
    """40% Focal + 40% Dice + 20% Boundary"""
    return (alpha * focal_loss_fn(pred, target)
          + 0.4   * dice_loss_fn(pred, target)
          + 0.2   * boundary_loss(pred, target))

# --- METRICS ---
def compute_iou(pred_logits, target, thr=0.32):
    pred  = (torch.sigmoid(pred_logits) > thr).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()

def compute_dice(pred_logits, target, thr=0.32):
    pred  = (torch.sigmoid(pred_logits) > thr).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2 * inter + 1e-6) / (union + 1e-6)).mean().item()

def compute_all_metrics(model, loader, threshold, device="cpu"):
    model.eval()
    TP = FP = FN = TN = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            probs  = torch.sigmoid(model(imgs))
            preds  = (probs > threshold).float()

            preds_f = preds.view(-1)
            masks_f = masks.view(-1)

            TP += ((preds_f == 1) & (masks_f == 1)).sum().item()
            FP += ((preds_f == 1) & (masks_f == 0)).sum().item()
            FN += ((preds_f == 0) & (masks_f == 1)).sum().item()
            TN += ((preds_f == 0) & (masks_f == 0)).sum().item()

    iou         = TP / (TP + FP + FN + 1e-6)
    precision   = TP / (TP + FP + 1e-6)
    recall      = TP / (TP + FN + 1e-6)
    f1          = 2 * precision * recall / (precision + recall + 1e-6)
    dice        = 2 * TP / (2 * TP + FP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)

    total = TP + FP + FN + TN
    po    = (TP + TN) / total
    pe    = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (total ** 2 + 1e-6)
    kappa = (po - pe) / (1 - pe + 1e-6)

    return dict(iou=iou, precision=precision, recall=recall, f1=f1,
                dice=dice, specificity=specificity, kappa=kappa,
                TP=TP, FP=FP, FN=FN, TN=TN)