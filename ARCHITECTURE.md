# ARCHITECTURE.md

## Integrated Component Architecture (Update 2026-08-13)

```text
+-------------------------+
| Streamlit UI            |
| - RBAC login            |
| - Registration module  |
| - Camera face capture |
| - Mode selector         |
| - Camera controls       |
| - Settings panel        |
+-----------+-------------+
            |
            v
+-------------------------+
| Video Ingestion         |
| - Browser camera/WebRTC |
| - OpenCV webcam/files   |
| - RTSP/HTTP IP streams  |
| - Frame resize/diagnostics|
+-----------+-------------+
            |
            v
+-------------------------+
| Processing Layer        |
| - DeepFace matching     |
| - OpenCV fallback       |
| - Contour heuristics    |
| - CASCADE_IAFACE_ANALYSIS|
| - Status annotations    |
+-----------+-------------+
            |
            v
+-------------------------+
| Storage Layer           |
| - registered_faces/     |
| - unknown_faces/        |
| - detection_logs/       |
| - temp files            |
+-------------------------+
```

## Enhanced Data Flow

1. **Authentication Gate** → Sidebar RBAC enforces `Administrator`/`Operator` roles.
2. **Camera & Mode Selection** → Browser WebRTC captures browser camera frames; OpenCV captures local webcams, files, and IP streams.
3. **Cascaded Processing Pipeline**:
   - **DeepFace Recognition** (configurable model/backend when TensorFlow is available)
   - **OpenCV fallback** (frontal/profile detection plus HOG/CLAHE embedding when DeepFace is unavailable)
   - **Contour Heuristics** for threat detection
   - **Anti‑Spoofing Detection** (optional)
   - **Facial Attribute Analysis** (age, gender, emotion, race)
4. **Event Annotation** → Annotated frames returned to UI for real‑time feedback.
5. **Persistent Logging** → All events appended to `detection_logs/system_audit_logs.csv`.
6. **Unknown Person Handling** → Automated re‑identification, previous sighting lookup, current DB update, and live date/time/location feedback.
7. **Configuration Management** → Sidebar settings control models, backends, thresholds, and feature toggles.

## Directory Layout (Updated)

```text
project_n_one/
├── app.py                         # Main Streamlit entry point
├── requirements.txt               # Dependency list
├── README.md                      # Project overview and setup guide
├── ARCHITECTURE.md                # This design document
├── SYNOPSIS.md                    # Project synopsis
├── BRAIN.md                       # Core logic specification
├── DEEPFACE_FEATURES.md           # DeepFace integration details
├── .gitignore                     # Ignored files and folders
├── registered_faces/              # Images of registered subjects
├── unknown_faces/                 # Auto‑captured unknown faces
├── detection_logs/                # CSV audit logs
│   └── system_audit_logs.csv
├── unknown_person_db.csv          # Database of unique unknown persons
├── unknown_sighting_log.csv       # Sighting timestamp/location log
└── temp_current_frame.jpg         # Temporary frame storage (cleared on reset)
```

*All configuration toggles are accessible via the sidebar settings panel, enabling runtime adjustments without code changes.*

## Current role and matching rules (2026-08-13)

Both `Administrator` and `Operator` sessions can view the dashboard inventory: total registered faces, total unknown faces, and registered profiles filtered by `All`, `Staff`, or `Victim`. Only Administrators can register profiles or change model settings.

The same dashboard provides a read-only photo viewer with `Registered`, `Victim`, and `Unknown` categories. Selecting an entry displays the stored face image and the relevant profile or sighting metadata.

`1. Lost Person Search` requires one Victim target. The processing layer restricts known-face matching to that target profile. Other registered faces and unknown faces are not shown as match results; unmatched faces are still saved or re-identified in the unknown store. A successful target is shown in a separate Victim Found card with image, profile ID, location, distance, timestamp, active configuration, and history. The operator-provided camera location is saved with unknown sightings and successful Victim sightings.

`2. Member Attendance Logger` compares against all registered Staff and Victim profiles. Known identities are displayed and unknown faces are re-identified or registered with visible unknown details. Use `Facenet512` + `retinaface` + `cosine` with a calibrated `0.30–0.40` starting threshold when the full DeepFace runtime is installed.

`3. Threat & Weapon Detection` uses contour/heuristic threat processing and does not depend on the face model selection.

The recommended live source is `Browser Webcam (WebRTC)`. The server-side `Laptop Webcam` option remains useful when the camera is attached to the machine running Streamlit; it can show a black frame when the app is remote, permission is denied, or another process owns the device.

Victim location history is stored in `detection_logs/victim_sighting_log.csv`. Unknown data remains in `unknown_person_db.csv`, `unknown_sighting_log.csv`, and `unknown_faces/`.
