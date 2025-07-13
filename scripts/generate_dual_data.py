import os
from PIL import Image
import random
from torchvision.transforms.functional import adjust_brightness, adjust_contrast

RAW_DIR = "data/raw/landuse-scene-classification"
OUT_LR1 = "data/processed/lr1"
OUT_LR2 = "data/processed/lr2"
OUT_HR  = "data/processed/hr"

# Create output dirs
for d in [OUT_LR1, OUT_LR2, OUT_HR]:
    os.makedirs(d, exist_ok=True)

def process_image(img_path, img_name):
    try:
        img = Image.open(img_path).convert("RGB")
        hr = img.resize((512, 512), Image.BICUBIC)  # ✅ Changed from 256 to 512

        # 🌤️ Add slight lighting and contrast variations
        hr = adjust_brightness(hr, random.uniform(0.9, 1.1))
        hr = adjust_contrast(hr, random.uniform(0.9, 1.1))

        # 📉 LR1: downscale-upscale (blur)
        lr1 = hr.resize((256, 256), Image.BICUBIC)

        # 🔁 LR2: slight random rotation
        angle = random.uniform(-2, 2)
        rotated = hr.rotate(angle)
        lr2 = rotated.resize((256, 256), Image.BICUBIC)

        # 💾 Save processed images
        hr.save(os.path.join(OUT_HR, img_name))
        lr1.save(os.path.join(OUT_LR1, img_name))
        lr2.save(os.path.join(OUT_LR2, img_name))

    except Exception as e:
        print(f"❌ Failed to process {img_path}: {e}")

if __name__ == "__main__":
    count = 0
    for cls_folder in os.listdir(RAW_DIR):
        cls_path = os.path.join(RAW_DIR, cls_folder)
        if os.path.isdir(cls_path):
            for img_file in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_file)
                process_image(img_path, f"{count}.jpg")
                count += 1

    print(f"✅ Processed {count} images into dual low-res format.")
