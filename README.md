# 🚀 Antariksh DualFusion: Dual Image Super-Resolution Network

A Deep Learning–based solution designed to enhance satellite imagery by fusing **two complementary low-resolution (LR)** inputs into a single **super-resolved high-resolution (HR)** output.

---

## 📌 Problem Statement

This project was built for the **Bharatiya Antariksh Hackathon 2025**, under the challenge:  
**“Dual Image Super Resolution for High-Resolution Optical Satellite Imagery and its Blind Evaluation”**.

---

## 🧠 Core Idea

Traditional single-image enhancement struggles to recover fine details from blurry inputs.  
Our model leverages **two different perspectives of the same scene** — for example, images taken under different lighting or angles — to fuse their information and reconstruct a clearer, sharper version.

---

## 🏗️ Model Architecture

The model (`DualSRNet`) is a custom deep neural network combining:

- ✨ Feature extraction layers (CNNs)
- 🔄 Cross-input attention modules
- 📶 Upsampling + skip connections
- 🔍 Supervised training on dual-input–HR triplets

---

## 🖥️ Streamlit App

We built a complete UI using **Streamlit**, where users can:

- Upload two LR satellite images (of the same scene)
- (Optionally) upload a Ground Truth HR image for comparison
- Generate and download the Super-Resolved image
- View comparison grids (LR1 | LR2 | SR | HR)

Run the app locally:

```bash
streamlit run app.py
# Antariksh-DualFusion