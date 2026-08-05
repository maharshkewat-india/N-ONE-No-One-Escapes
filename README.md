# PROJECT N-ONE: Advanced AI Surveillance Platform

PROJECT N-ONE is a Streamlit-based surveillance console for identity-aware monitoring, member attendance logging, and threat contour heuristics. The system supports local webcam input, uploaded video files, and RTSP/IP camera streams.

## Features

- RBAC login with Administrator and Operator roles
- Admin-only face registration for lost-person / victim profiles and staff profiles
- Multi-mode surveillance workflow:
  - Lost Person Search (target alert mode)
  - Member Attendance Logger
  - Threat & Weapon Detection Mode
- Live frame visualization and status updates
- Append-only audit logging to CSV
- CSV export for investigation records

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

## RTSP / IP Camera Setup

Use an RTSP URI in the form:

```text
rtsp://username:password@camera-ip:554/stream_path
```

For better compatibility with OpenCV, the app uses the RTSP TCP transport path when the IP Camera Stream source is selected.

## Directory Layout

```text
project_n_one/
├── app.py
├── requirements.txt
├── README.md
├── BRAIN.md
├── ARCHITECTURE.md
├── .gitignore
├── registered_faces/
├── detection_logs/
└── temp_current_frame.jpg
```

## Notes

- Registered face images are stored in the registered_faces directory.
- Audit logs are appended to detection_logs/system_audit_logs.csv.
- For hardware-limited environments, the app gracefully falls back to lightweight histogram matching when DeepFace is unavailable.
