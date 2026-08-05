# BRAIN.md

## System Logic Overview

PROJECT N-ONE uses a layered surveillance pipeline:

1. Authentication Gate
   - The sidebar enforces RBAC using the admin/admin123 and operator/op123 credentials.
   - Administrators may register new face profiles; operators can view and operate the console.

2. Registration and Identity Storage
   - Face images are saved in the registered_faces directory with a naming prefix such as Lost_name.jpg or Member_name.jpg.
   - Each saved profile becomes a candidate match target for later frame analysis.

3. Frame Processing Loop
   - The active video source is read frame by frame.
   - Each frame passes through the selected surveillance mode logic:
     - Lost Person Search: checks for matches against registered lost-person identities.
     - Member Attendance Logger: checks for registered staff/member identities. If a face is detected that does not match a registered member, it proceeds to the Unknown Person Tracking logic.
     - Threat & Weapon Detection Mode: applies contour-based heuristics for suspicious elongated shapes.

4. Unknown Person Tracking
    - This workflow is triggered in "Member Attendance" mode for faces that do not match any registered profiles.
    - The detected face is compared against a separate database of previously seen unknown individuals stored in the `unknown_faces/` directory.
    - **If a match is found:** The system re-identifies the person, updates their `last_seen_timestamp` in the `unknown_person_db.csv`, and logs the new sighting in `unknown_sighting_log.csv`. The UI displays the person's unique ID and their first-seen timestamp.
    - **If no match is found:** A new unique ID is generated. The person's face is saved to the `unknown_faces/` directory, and a new record is created in `unknown_person_db.csv` and `unknown_sighting_log.csv`.

5. DeepFace Matching
   - When the DeepFace library is available, the app uses DeepFace.find against the registered_faces folder.
   - The matching result returns the closest identity profile, which is then annotated on the frame.
   - In environments where DeepFace is unavailable or fails, the app falls back to a lightweight OpenCV histogram comparison approach so the UI remains usable.

5. Threat Heuristic Logic
   - A Canny edge transform and contour extraction identify objects with suspicious elongated or irregular geometry.
   - Bounding boxes are drawn around candidate shapes and marked as threat alerts.
   - These events are logged to the CSV audit trail.

6. Audit Logging Workflow
   - Every significant event is appended to detection_logs/system_audit_logs.csv.
   - The log table is shown in the Streamlit UI and can be downloaded as CSV.
