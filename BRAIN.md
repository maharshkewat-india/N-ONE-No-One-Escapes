# BRAIN.md

## Current system logic (Update 2026-08-13)

This document describes the behavior implemented in `app.py`. The application uses local CSV/image storage and does not claim blockchain, IPFS, BIP-46, or end-to-end encryption features.

1. **Authentication and state**
   - Login credentials are read from environment variables or Streamlit Secrets.
   - `Administrator` can register Staff/Victim profiles, configure models, clear registered profiles, and reset audit data.
   - `Operator` can monitor feeds, run searches, and review inventory/logs without administrator controls.
   - Streamlit session state stores the active role, selected mode, selected Victim target, camera location, and the latest result.

2. **Profile registration**
   - The Administrator uploads an image or captures one through the browser camera.
   - Exactly one face is required; the system saves only the padded face crop.
   - New files use `Staff_<name>.jpg` or `Victim_<name>.jpg`.
   - Legacy `Member_` files normalize to Staff and `Lost_` files normalize to Victim.

3. **Dashboard inventory**
   - Both authenticated roles see counts for registered faces and unique unknown faces.
   - The registered inventory can be filtered to `All`, `Staff`, or `Victim`.
   - A read-only photo viewer lets authenticated users select and view a Registered, Victim, or Unknown face image with available metadata.

4. **Normal recognition modes**
   - Each frame is processed by the configured DeepFace backend when TensorFlow is available; otherwise `OpenCVFaceBackend` uses frontal/profile Haar detection plus a HOG/CLAHE fallback embedding.
   - Known profiles are matched against the configured similarity threshold.
   - If no known profile matches, the frame is checked against the unknown-face store.
   - A new unknown is saved to `unknown_faces/`; a repeat unknown updates its last-seen time and location.

5. **Victim-only search mode**
   - `1. Lost Person Search` requires the operator to select one Victim profile.
   - The known-face cache is restricted to that profile; Staff and other Victims cannot produce a visible match.
   - Unmatched faces continue through unknown registration/re-identification, but their IDs, boxes, and images are not shown in the Victim-search result panel.
   - On a target match, the system records `Victim Found`, displays the Victim name and camera location, and shows the latest record for each location.
   - The result is also rendered in a dedicated `VICTIM FOUND` card with the saved Victim image, profile ID, distance, timestamp, active model/backend/metric, and sighting history.

6. **Mode-specific model rules**
   - Victim Search: use `Facenet512` + `retinaface` + `cosine` in a verified full DeepFace runtime; start with distance threshold `0.30–0.40` and calibrate against real footage.
   - Member Attendance Logger: use the same configuration for accuracy, or `Facenet` for lower CPU cost. Compare against all Staff/Victim profiles and show known plus unknown results.
   - Threat Detection: uses contour/heuristic threat logic; face recognition model settings do not control it.

7. **Location-aware persistence**
   - The operator enters a camera location before starting a feed.
   - Unknown sightings use `unknown_person_db.csv` and `unknown_sighting_log.csv`.
   - Victim sightings use `detection_logs/victim_sighting_log.csv`, with short duplicate suppression so every video frame does not create a new history row.

8. **Video source and camera behavior**
   - `Browser Webcam (WebRTC)` is the recommended live source. The browser captures the camera and the WebRTC worker annotates frames without requiring server-side camera access.
   - `Laptop Webcam`, recorded files, and IP streams use OpenCV. A black local webcam frame means permission, device contention, backend, or remote-host limitations; use WebRTC for a browser/remote camera.
   - Closing or refreshing the browser during streaming can produce Windows `ConnectionResetError: [WinError 10054]`; this is a client connection reset, not a recognition result.
