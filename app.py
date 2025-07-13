import streamlit as st
import torch
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
from io import BytesIO
from models.dual_sr_net import DualSRNet

# ================== Configuration ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_model.pth"

# Image transformation pipeline (for model input)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Denormalize output tensor to [0, 1]
def denorm(tensor):
    return tensor * 0.5 + 0.5

# =============== Load Model with Caching ===============
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        model = DualSRNet().to(DEVICE)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model checkpoint not found at {MODEL_PATH}. Please train the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# =============== Helper Functions ===============
def preprocess_image(img: Image.Image, size=None) -> torch.Tensor:
    if size:
        img = img.resize(size, Image.BICUBIC)
    return transform(img).unsqueeze(0).to(DEVICE)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = denorm(tensor).clamp(0, 1)
    np_img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)

def save_image_to_bytes(tensor: torch.Tensor, format="PNG") -> BytesIO:
    img = tensor_to_pil(tensor)
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer

def create_comparison_grid(images: list, nrow=4) -> Image.Image:
    grid = make_grid(images, nrow=nrow)
    return tensor_to_pil(grid.unsqueeze(0))

# ================== Streamlit Interface ==================
st.set_page_config(
    page_title="Dual Image Super-Resolution",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Dual Image Super-Resolution")
st.markdown("""
Enhance two complementary low-resolution satellite images into one clear high-resolution output using deep learning.

Just upload, generate, and download.
""")

# Sidebar Instructions
with st.sidebar.expander("📌 How It Works", expanded=True):
    st.markdown("""
    - Upload **two complementary LR images** (e.g. different angles, slight shifts).
    - (Optional) Upload a **Ground Truth HR image** for visual comparison.
    - Click **Generate** to create a super-resolved image.
    - Download the output or compare with original inputs.

    ✅ No need for HR image during actual usage  
    ⚙️ Works even without internet (locally)  
    ⚡ GPU-accelerated if available
    """)

# Upload Inputs
col1, col2 = st.columns(2)
with col1:
    lr1_file = st.file_uploader("🖼️ Upload LR Image 1", type=["jpg", "jpeg", "png"])
with col2:
    lr2_file = st.file_uploader("🖼️ Upload LR Image 2", type=["jpg", "jpeg", "png"])

hr_file = st.file_uploader("🔍 Upload HR Image (optional)", type=["jpg", "jpeg", "png"])

# Generate Button
generate = st.button("🚀 Generate Super-Resolved Image")

if generate:
    if not model:
        st.error("Model not loaded. Cannot generate images.")
    elif not lr1_file or not lr2_file:
        st.warning("Please upload both LR images to proceed.")
    else:
        try:
            # Load and resize inputs
            lr1 = Image.open(lr1_file).convert("RGB")
            lr2 = Image.open(lr2_file).convert("RGB")
            common_size = (min(lr1.width, lr2.width), min(lr1.height, lr2.height))
            lr1 = lr1.resize(common_size, Image.BICUBIC)
            lr2 = lr2.resize(common_size, Image.BICUBIC)

            # Preprocess
            lr1_tensor = preprocess_image(lr1)
            lr2_tensor = preprocess_image(lr2)

            with torch.no_grad():
                pred_tensor = model(lr1_tensor, lr2_tensor).clamp(-1, 1)

            pred_img = tensor_to_pil(pred_tensor)
            pred_bytes = save_image_to_bytes(pred_tensor)

            # Display Input Images
            st.subheader("🧿 Low-Resolution Inputs")
            st.image([lr1, lr2], caption=["LR Image 1", "LR Image 2"], width=300)

            if hr_file:
                hr = Image.open(hr_file).convert("RGB").resize(pred_img.size, Image.BICUBIC)
                hr_tensor = preprocess_image(hr)
                st.subheader("📊 Comparison Grid")
                grid = create_comparison_grid([
                    lr1_tensor.squeeze(0).cpu(),
                    lr2_tensor.squeeze(0).cpu(),
                    pred_tensor.squeeze(0).cpu(),
                    hr_tensor.squeeze(0).cpu()
                ])
                st.image(grid, caption="LR1 | LR2 | Predicted HR | Ground Truth HR", use_column_width=True)
            else:
                st.subheader("🖼️ Super-Resolved Output")
                st.image(pred_img, use_column_width=True)

            st.download_button(
                label="⬇️ Download Output (PNG)",
                data=pred_bytes,
                file_name="super_resolved.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Error occurred: {e}")
else:
    st.info("📥 Upload inputs and click Generate to begin.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using PyTorch + Streamlit · Dual-Image SR Prototype · [GitHub](https://github.com/your-repo)")
