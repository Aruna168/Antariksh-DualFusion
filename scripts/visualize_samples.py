import os
from PIL import Image
import matplotlib.pyplot as plt

LR1_DIR = "data/processed/lr1"
LR2_DIR = "data/processed/lr2"
HR_DIR  = "data/processed/hr"

# Pick a sample image by index
sample_idx = "0"  # or "1", "2", etc.
filename = f"{sample_idx}.jpg"

# Load images
lr1 = Image.open(os.path.join(LR1_DIR, filename))
lr2 = Image.open(os.path.join(LR2_DIR, filename))
hr  = Image.open(os.path.join(HR_DIR, filename))

# Plot them side-by-side
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(lr1)
axs[0].set_title("Low-Res View 1")
axs[1].imshow(lr2)
axs[1].set_title("Low-Res View 2")
axs[2].imshow(hr)
axs[2].set_title("High-Res Ground Truth")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()
