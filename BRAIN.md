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
     - Member Attendance Logger: checks for registered staff/member identities.
     - Threat & Weapon Detection Mode: applies contour-based heuristics for suspicious elongated shapes.

4. DeepFace Matching
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
