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
- Added a duplicate-suppression key to only log new detection events once per change.
- Added preview support for uploaded video files using `st.video(...)`.
- Updated the dashboard title and subtitle for a cleaner command center presentation.
- Improved the surveillance loop so it logs only when detection state changes.

## Files Changed

- `app.py`

## Validation Notes

- The updated app now detects faces in each frame and annotates them with correct bounding boxes.
- Video uploads show a video preview in the dashboard.
- Audit records are no longer duplicated for continuous detections.
