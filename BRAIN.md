# BRAIN.md

## Current system logic (Update 2026-08-09)

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

4. **Normal recognition modes**
   - Each frame is represented by the configured DeepFace/OpenCV backend.
   - Known profiles are matched against the configured similarity threshold.
   - If no known profile matches, the frame is checked against the unknown-face store.
   - A new unknown is saved to `unknown_faces/`; a repeat unknown updates its last-seen time and location.

5. **Victim-only search mode**
   - `1. Lost Person Search` requires the operator to select one Victim profile.
   - The known-face cache is restricted to that profile; Staff and other Victims cannot produce a visible match.
   - Unmatched faces continue through unknown registration/re-identification, but their IDs, boxes, and images are not shown in the Victim-search result panel.
   - On a target match, the system records `Victim Found`, displays the Victim name and camera location, and shows the latest record for each location.

6. **Location-aware persistence**
   - The operator enters a camera location before starting a feed.
   - Unknown sightings use `unknown_person_db.csv` and `unknown_sighting_log.csv`.
   - Victim sightings use `detection_logs/victim_sighting_log.csv`, with short duplicate suppression so every video frame does not create a new history row.

7. **Operational caveat**
   - The live feed currently uses a synchronous OpenCV loop. Closing or refreshing the browser during streaming can produce Windows `ConnectionResetError: [WinError 10054]`; this is a client connection reset, not a recognition result.
