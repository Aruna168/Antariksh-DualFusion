import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.utils import save_image

from models.dual_sr_net import DualSRNet
from dataset.dual_sr_dataset import DualSRDataset

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def evaluate_psnr_ssim(pred, gt):
    pred_img = pred.squeeze().cpu().numpy().transpose(1, 2, 0)
    gt_img = gt.squeeze().cpu().numpy().transpose(1, 2, 0)
    return psnr(gt_img, pred_img, data_range=1.0), ssim(gt_img, pred_img, channel_axis=-1, data_range=1.0)

# ========== Config ==========
EPOCHS = 30
BATCH_SIZE = 4  # Adjust if you face memory issues with 512x512 inputs
LR = 1e-4
MAX_SAMPLES = 1000  # Increase as per dataset size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== Transforms ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB normalization
])

# ========== Dataset & Loaders ==========
full_dataset = DualSRDataset(
    lr1_dir="data/processed/lr1",
    lr2_dir="data/processed/lr2",
    hr_dir="data/processed/hr",
    transform=transform
)

subset_indices = list(range(min(len(full_dataset), MAX_SAMPLES)))
subset_dataset = Subset(full_dataset, subset_indices)

train_size = int(0.9 * len(subset_dataset))
val_size = len(subset_dataset) - train_size
train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========== Model ==========
model = DualSRNet(upscale_factor=2).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

best_val_loss = float('inf')

# ========== Training Loop ==========
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for lr1, lr2, hr in train_loader:
        lr1, lr2, hr = lr1.to(DEVICE), lr2.to(DEVICE), hr.to(DEVICE)

        optimizer.zero_grad()
        pred = model(lr1, lr2)
        loss = criterion(pred, hr)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ========== Validation ==========
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lr1, lr2, hr in val_loader:
            lr1, lr2, hr = lr1.to(DEVICE), lr2.to(DEVICE), hr.to(DEVICE)
            pred = model(lr1, lr2)
            loss = criterion(pred, hr)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        print("💾 Saved best model.")

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"dualsr_epoch{epoch+1}.pth"))

    # Save sample predictions
    with torch.no_grad():
        sample_lr1, sample_lr2, _ = next(iter(val_loader))
        sample_lr1, sample_lr2 = sample_lr1.to(DEVICE), sample_lr2.to(DEVICE)
        sample_pred = model(sample_lr1, sample_lr2)
        save_image(sample_pred * 0.5 + 0.5, f"pred_epoch{epoch+1}.png")

print("✅ Training completed with validation support!")

