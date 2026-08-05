# ARCHITECTURE.md

## Component Architecture

```text
+-------------------------+
| Streamlit UI            |
| - RBAC login            |
| - registration module  |
| - mode selector         |
| - camera controls       |
+-----------+-------------+
            |
            v
+-------------------------+
| Processing Layer        |
| - OpenCV frame capture  |
| - DeepFace matching     |
| - contour heuristics    |
| - status annotations    |
+-----------+-------------+
            |
            v
+-------------------------+
| Storage Layer           |
| - registered_faces/    |
| - unknown_faces/       |
| - detection_logs/      |
| - temp files            |
+-------------------------+
```

## Data Flow

1. User authenticates through the sidebar.
2. The selected camera source is opened by OpenCV.
3. Each frame is processed by the active surveillance mode.
4. Matching results or threat contours are annotated on the frame. In "Member Attendance" mode, unknown faces are logged and re-identified.
5. Events are appended to the CSV audit log and reflected in the UI.

## RBAC Permissions Matrix

| Role | Login | Register Profiles | Start/Stop Stream | View Logs | Export Logs |
|------|-------|-------------------|-------------------|-----------|-------------|
| Administrator | Allowed | Allowed | Allowed | Allowed | Allowed |
| Operator | Allowed | Denied | Allowed | Allowed | Allowed |

## Directory Layout

```text
project_n_one/
├── app.py                # main Streamlit app entry point
├── requirements.txt      # package dependencies
├── README.md             # user documentation
├── BRAIN.md              # system logic specification
├── ARCHITECTURE.md       # system design documentation
├── .gitignore            # ignore rules for temp and generated files
├── registered_faces/     # user-registered face images
├── unknown_faces/        # auto-registered images of unknown persons
├── detection_logs/       # CSV audit logs
└── unknown_person_db.csv # Database of unknown persons
└── unknown_sighting_log.csv # Log of unknown person sightings
```
