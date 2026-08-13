# Contributing to N-ONE Surveillance Platform

First off, thank you for considering contributing to N-ONE! It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a feature request, please open a new issue. It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

### Fork & create a branch

If you're ready to contribute, fork the repository and create a new branch with a descriptive name.

```sh
# A good branch name would be (where issue #123 is the ticket you're working on)
git checkout -b 123-fix-bug-in-retinaface
```

## How to contribute

### Reporting Bugs

This section guides you through submitting a bug report for N-ONE. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

*   **Use a clear and descriptive title** for the issue to identify the problem.
*   **Describe the exact steps which reproduce the problem** in as much detail as possible.
*   **Provide specific examples to demonstrate the steps.** Include copy/pasteable snippets if possible.
*   **Include screenshots and animated GIFs** which show you following the described steps and clearly demonstrate the problem.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for N-ONE, including completely new features and minor improvements to existing functionality.

*   **Provide a step-by-step description of the suggested enhancement** in as much detail as possible.
*   **Explain why this enhancement would be useful** to most N-ONE users.

### Pull Request Process

1.  Ensure your code adheres to the project's coding standards.
2.  Update the `README.md` or other relevant documentation with details of changes, if any.
3.  Ensure your pull request has a clear title and description of the changes.

Thank you for your contribution!

## Current feature and documentation expectations

When changing recognition or dashboard behavior, update the relevant root-level documentation as well as `app.py`. In particular, keep these rules synchronized:

- Registered profiles use `Staff` and `Victim`; retain compatibility with `Member_` and `Lost_` files unless a migration is explicitly planned.
- Both Administrator and Operator dashboards expose registered/unknown counts and the registered-face inventory filters.
- Victim Search must remain target-specific: only the selected Victim can be displayed as a visible match.
- Unknown faces must continue to be stored/re-identified silently during Victim Search.
- Camera location belongs in unknown sightings and Victim sighting history.
- `Browser Webcam (WebRTC)` is the recommended live camera path; local webcam/file/IP paths use OpenCV. Document camera permission, black-frame, and Windows `WinError 10054` caveats when changing streaming behavior.
- Document the difference between full DeepFace/TensorFlow and `OpenCVFaceBackend` fallback. Sidebar model names must not be described as active neural models when the fallback is running.
- Keep the mode matrix accurate: Victim Search is selected-target-only with a separate Victim Found result; Attendance compares all registered profiles and shows unknown details; Threat Detection uses contour/heuristic logic.
- When adding a model setting, record the recommended model, detector backend, metric, threshold range, runtime dependency, and whether the setting affects each mode.

Before opening a pull request, run:

```powershell
python -m pytest tests -q
python -m py_compile app.py deepface_adapter.py
```

## Current operator contract (2026-08-13)

| User/task | Required action |
|---|---|
| Administrator | Register Staff/Victim face-only profiles, preferably with five angles; verify exactly one face is captured; configure and validate the AI runtime. |
| Operator — Victim Search | Select one Victim, enter location, use WebRTC for browser/remote cameras, and confirm the separate `VICTIM FOUND` card before acting. |
| Operator — Attendance Logger | Use all Staff/Victim profiles, run a short threshold calibration, and review known plus unknown IDs/details. |
| Operator — Threat Detection | Select Threat mode and review heuristic alerts manually; face settings are not used for weapon/threat decisions. |

Recommended full-runtime recognition configuration: `Facenet512` + `retinaface` + `cosine`, threshold `0.30–0.40` as a starting range. This recommendation requires TensorFlow/DeepFace to load successfully.
