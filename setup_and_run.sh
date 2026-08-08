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

# Upgrade pip and install required packages
pip install --upgrade pip
pip install streamlit pandas opencv-python-headless scipy pillow

# Run the app
streamlit run app.py