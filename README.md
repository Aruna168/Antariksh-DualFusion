# 🛰️ Antariksh DualFusion — Dual Image Super-Resolution Network

> **Bharatiya Antariksh Hackathon 2025**  
> Challenge: *"Dual Image Super Resolution for High-Resolution Optical Satellite Imagery and its Blind Evaluation"*

A deep learning pipeline that fuses **two complementary low-resolution satellite images** of the same scene into a single **super-resolved high-resolution output** — going beyond traditional single-image SR by exploiting cross-view information.

---

## 🧠 Why Dual-Input?

Single-image super-resolution hits a ceiling — one blurry frame can only recover so much detail. When two LR images of the same scene exist (different angles, lighting, or capture times), their complementary information can be fused to reconstruct detail that neither image contains alone.

**DualSRNet** learns to align, attend, and merge these two perspectives into one sharp HR output.

---

## 🏗️ Architecture — DualSRNet

```
LR Image 1 ──→ [ CNN Feature Extractor ]──┐
                                           ├──→ [ Cross-Input Attention ] ──→ [ Upsampler + Skip Connections ] ──→ SR Output
LR Image 2 ──→ [ CNN Feature Extractor ]──┘
```

| Module | Role |
|---|---|
| CNN Feature Extractor | Extracts spatial features independently from each LR input |
| Cross-Input Attention | Lets the model learn which regions in LR2 help reconstruct LR1, and vice versa |
| Upsampling + Skip Connections | Progressively reconstructs full resolution while preserving low-level detail |
| Supervised Training | Trained on dual-input / HR triplets with pixel-level reconstruction loss |

---

## 🖥️ Streamlit App

A full comparison UI built with Streamlit:

- Upload two LR satellite images of the same scene
- Optionally upload a Ground Truth HR image for metric evaluation
- Generate and download the super-resolved output
- View side-by-side comparison grid: **LR1 | LR2 | SR | HR**

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

## 🗂️ Repository Structure

```
Antariksh-DualFusion/
│
├── models/
│   ├── __init__.py
│   └── dual_sr_net.py          # DualSRNet architecture
│
├── scripts/
│   ├── generate_dual_data.py   # LR/HR triplet pair generation from dataset
│   ├── train.py                # Training pipeline
│   ├── inference.py            # Run SR inference on new image pairs
│   └── visualize_samples.py   # LR1 | LR2 | SR | HR comparison grids
│
├── app.py                      # Streamlit web app
├── requirements.txt
└── SETUP.md
```

---

## 📦 Dataset

- Multi-class aerial/satellite imagery from Kaggle (~2–4 GB)
- Scene categories: aeroplanes, agricultural land, apartments, forests, roads, and more
- Preprocessed into LR1 / LR2 / HR triplets using `generate_dual_data.py`

> Dataset not included due to size. See `SETUP.md` for download and preparation instructions.

---

## 📸 Sample Output

> SR output vs ground truth HR — model trained on limited compute; output quality scales with training duration and batch size.

| LR Input 1 | LR Input 2 | SR Output | Ground Truth HR |
|---|---|---|---|
| *<img width="412" height="631" alt="Screenshot 2025-07-02 185035" src="https://github.com/user-attachments/assets/a3d966c1-e8bc-4168-958c-1a00bc4663fe" />* | *<img width="390" height="620" alt="Screenshot 2025-07-02 185115" src="https://github.com/user-attachments/assets/d0b05a02-cc05-4827-84ac-9aad771079e9" />* | *<img width="393" height="619" alt="Screenshot 2025-07-02 185250" src="https://github.com/user-attachments/assets/fd73f133-d813-4edb-9374-909ce0698505" />* | *<img width="393" height="619" alt="Screenshot 2025-07-02 185250" src="https://github.com/user-attachments/assets/f1156ea4-8080-4ece-876b-a66b3bf66b3a" />* |

---

## 🔍 What I Learned

- Dual-input feature fusion using cross-attention — how models learn to align and leverage complementary views
- Trade-offs between PSNR/SSIM (pixel accuracy) and perceptual sharpness
- Building supervised training pipelines for image-to-image tasks
- End-to-end deployment: data prep → training → inference → interactive Streamlit UI

---

## 🛠️ Tech Stack

`Python` · `TensorFlow / PyTorch` · `OpenCV` · `Streamlit` · `NumPy`

---

## 🏆 Context

Built for the **Bharatiya Antariksh Hackathon 2025** — an ISRO-affiliated national challenge focused on applying ML and computer vision to satellite and space imagery problems.

---

**Arunaprabha K N** · [GitHub](https://github.com/Aruna168) · [LinkedIn](https://linkedin.com/in/aruna-rayan06)
