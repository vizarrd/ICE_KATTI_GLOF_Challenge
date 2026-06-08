import os
import glob
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import model architecture
from model_architecture import get_model

# Constants
BASE_DIR = "./SNUC GLOFeagles 2026 challenge datasets" # Update
OUTPUT_DIR = "./Test_Outputs" # Update
MODEL_SAVE_PATH = "./glacierlake_model.pth"
THRESHOLD = 0.32
device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_lake_mask_tta(model, image_path, device, threshold=0.32, min_area=50):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    scales = [384, 512, 640]
    probs_accum = np.zeros((orig_h, orig_w), dtype=np.float32)

    tta_images = [
        img_rgb,
        np.fliplr(img_rgb),
        np.flipud(img_rgb),
        np.rot90(img_rgb, 1),
        np.rot90(img_rgb, 2),
        np.rot90(img_rgb, 3),
    ]

    model.eval()
    with torch.no_grad():
        for scale in scales:
            scale_probs = np.zeros((orig_h, orig_w), dtype=np.float32)

            preprocess = A.Compose([
                A.Resize(scale, scale),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

            for i, aug_img in enumerate(tta_images):
                x = preprocess(image=aug_img.copy())["image"].unsqueeze(0).float().to(device)
                prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()

                if i == 1: prob = np.fliplr(prob)
                elif i == 2: prob = np.flipud(prob)
                elif i == 3: prob = np.rot90(prob, -1)
                elif i == 4: prob = np.rot90(prob, -2)
                elif i == 5: prob = np.rot90(prob, -3)

                prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                scale_probs += prob

            probs_accum += scale_probs / len(tta_images)

    prob_map = probs_accum / len(scales)
    binary = (prob_map > threshold).astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == lbl] = 1

    return img_rgb, prob_map, filtered

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model = get_model(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    
    image_paths = sorted(glob.glob(os.path.join(BASE_DIR, "*.png")))
    print(f"Processing {len(image_paths)} images...")

    for img_path in image_paths:
        try:
            _, _, pred_mask = predict_lake_mask_tta(model, img_path, device, threshold=THRESHOLD)
            mask_to_save = (pred_mask > 0).astype(np.uint8) * 255
            
            filename = os.path.basename(img_path).replace(".png", "_mask.png")
            save_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(save_path, mask_to_save)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Failed on {img_path}: {e}")

if __name__ == "__main__":
    main()