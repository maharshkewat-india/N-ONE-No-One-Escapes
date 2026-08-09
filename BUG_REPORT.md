# Bug Report: N-ONE Surveillance Dashboard

## Summary
This report covers the current issues identified in `app.py` and the fixes applied.

## Identified Bugs

1. Duplicate audit log entries
   - Every frame with a matched detection was appended to the CSV log.
   - Result: repeated log rows for the same person while the video stream remained active.

2. Video file input displayed as an image-style frame
   - The dashboard uses `st.image` for processed frames only.
   - Uploaded video files were not previewed as a video, so users saw a static-like feed instead of video context.

3. Face match bounding boxes were incorrect and too large
   - Identity matching was performed on the full image, not on individual detected faces.
   - The resulting annotation box covered a broad central region instead of matching actual face positions.

4. Detection mode behavior was inconsistent
   - The active mode display and event logging were not tied to discrete detection events.
   - This made it hard to use the mode switching and streaming controls reliably.

5. Dashboard title needed a more polished presentation
   - The title was generic and did not present a polished command center UI.

## Fixes Applied

- Added real face detection and per-face identity matching.
- Switched profile matching to operate on face ROIs instead of whole frames.
- Added Victim sighting throttling so repeated frames at the same camera do not create a new Victim-history row every frame.
- Added registered-face inventory counts and `All`/`Staff`/`Victim` dashboard filtering.
- Added target-only Victim matching and silent unknown tracking for Lost Person Search.
- Added camera-location persistence for unknown and Victim sightings.
- Updated the dashboard title and subtitle for a cleaner command center presentation.

## Files Changed

- `app.py`

## Validation Notes

- The updated app now detects faces in each frame and annotates them with correct bounding boxes.
- The selected Victim is the only visible identity result in Lost Person Search.
- Unknown faces continue to populate the unknown store without appearing as Victim matches.
- Windows `WinError 10054` is documented as a browser/Streamlit connection reset during live streaming.

## Current behavior and troubleshooting (2026-08-09)

### Victim search behavior

- `1. Lost Person Search` requires a selected Victim target.
- Only the selected Victim can produce a visible identity match.
- Other faces are still saved/re-identified as unknowns, but unknown IDs and boxes are intentionally hidden in this mode.
- The operator-provided camera location is stored with unknown events and Victim sightings.
- Successful Victim matches appear with the name, current location, and latest sighting for each previously recorded location.

### Windows `ConnectionResetError: [WinError 10054]`

This traceback comes from Python's asyncio/Proactor network layer when the browser closes or resets the Streamlit WebSocket. It is commonly triggered by refreshing/closing the browser during the synchronous OpenCV streaming loop, stopping a feed, or losing an IP-camera connection. It is not evidence that face matching failed. Stop the feed before refreshing; if the dashboard disconnects repeatedly, the streaming loop should be migrated to a non-blocking Streamlit fragment or dedicated video component.

### Validation

Run the project-level tests and syntax check from the repository root:

```powershell
python -m pytest tests -q
python -m py_compile app.py deepface_adapter.py
```
