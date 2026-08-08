# ARCHITECTURE.md

## Integrated Component Architecture (Update 2026-08-07)

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
| Processing Layer        |
| - OpenCV frame capture  |
| - DeepFace matching     |
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
2. **Camera & Mode Selection** → OpenCV captures frames based on user choice.
3. **Cascaded Processing Pipeline**:
   - **DeepFace Recognition** (configurable model/backend)
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
