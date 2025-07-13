# Setup Instructions

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model (required):**
   ```bash
   python scripts/train.py
   ```
   This will create model checkpoints in the `checkpoints/` folder.

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Note
- The model checkpoint (`checkpoints/best_model.pth`) is not included in this repo due to file size.
- You must train the model first before using the Streamlit app.
- Training data should be placed in `data/processed/` folders as described in README.md.