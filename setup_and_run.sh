#!/usr/bin/env bash
# ------------------------------------------------------------
# One‑command installer & launcher for the N‑ONE surveillance app
# ------------------------------------------------------------

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip and install the base runtime. This works on Python 3.14 using
# the built-in OpenCV fallback; see requirements-deepface.txt for full AI mode.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run the app
streamlit run app.py
