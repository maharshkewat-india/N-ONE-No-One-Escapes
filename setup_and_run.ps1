# ------------------------------------------------------------
# One‑command installer & launcher for the N‑ONE surveillance app
# ------------------------------------------------------------

# Create venv if it doesn't exist
if (-Not (Test-Path "venv")) {
    python -m venv venv
}

# Activate venv
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip and install required packages
pip install --upgrade pip
pip install streamlit pandas opencv-python-headless scipy pillow

# Run the app
streamlit run app.py