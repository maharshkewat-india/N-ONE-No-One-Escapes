# ARCHITECTURE.md

## Integrated Component Architecture (Update 2026-08-09)

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

## Current role and matching rules (2026-08-09)

Both `Administrator` and `Operator` sessions can view the dashboard inventory: total registered faces, total unknown faces, and registered profiles filtered by `All`, `Staff`, or `Victim`. Only Administrators can register profiles or change model settings.

`1. Lost Person Search` requires one Victim target. The processing layer restricts known-face matching to that target profile. Other registered faces and unknown faces are not shown as match results; unmatched faces are still saved or re-identified in the unknown store. The operator-provided camera location is saved with unknown sightings and successful Victim sightings.

Victim location history is stored in `detection_logs/victim_sighting_log.csv`. Unknown data remains in `unknown_person_db.csv`, `unknown_sighting_log.csv`, and `unknown_faces/`.
