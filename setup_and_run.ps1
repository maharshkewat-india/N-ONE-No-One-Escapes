# ------------------------------------------------------------
# One‑command installer & launcher for the N‑ONE surveillance app
# ------------------------------------------------------------

# Create venv if it doesn't exist
if (-Not (Test-Path "venv")) {
    python -m venv venv
}

# Activate venv
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip and install the base runtime. This works on Python 3.14 using
# the built-in OpenCV fallback; see requirements-deepface.txt for full AI mode.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run the app
streamlit run app.py
