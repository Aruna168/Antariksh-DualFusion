import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from models.dual_sr_net import DualSRNet

# ========== Config ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_model.pth"  # Update if using epoch-specific

# Get user input
IMG_NAME = input("🔍 Enter image file name (e.g., 0.jpg): ").strip()
LR1_PATH = f"data/processed/lr1/{IMG_NAME}"
LR2_PATH = f"data/processed/lr2/{IMG_NAME}"
HR_PATH  = f"data/processed/hr/{IMG_NAME}"  # Optional ground truth

SAVE_PATH = f"outputs/result_{IMG_NAME}"
os.makedirs("outputs", exist_ok=True)

# ========== Transforms ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # ✅ RGB Normalization
])
denorm = lambda t: t * 0.5 + 0.5  # [-1, 1] to [0, 1]

# ========== Load Model ==========
model = DualSRNet().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ========== Load Inputs ==========
def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

lr1 = load_image(LR1_PATH)
lr2 = load_image(LR2_PATH)

# Optional: Load GT HR image
has_gt = os.path.exists(HR_PATH)
hr = load_image(HR_PATH) if has_gt else None

# ========== Inference ==========
with torch.no_grad():
    pred_hr = model(lr1, lr2)
    pred_hr = denorm(pred_hr)

# ========== Save Output ==========
save_image(pred_hr, SAVE_PATH)

# ========== Side-by-side Comparison ==========
if has_gt:
    grid = make_grid([
        denorm(lr1[0]),
        denorm(lr2[0]),
        pred_hr[0].cpu(),
        denorm(hr[0])
    ], nrow=4)

    save_image(grid, f"outputs/grid_{IMG_NAME}")
    print(f"📸 Saved side-by-side result → outputs/grid_{IMG_NAME}")
else:
    print("⚠️ Ground truth not found. Skipping side-by-side comparison.")

print(f"✅ Super-resolved image saved at: {SAVE_PATH}")
